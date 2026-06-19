import copy

from typing import Any, List, Optional, Tuple
import random
import numpy as np
import warnings

import torch

from pyMAISE import settings
from pyMAISE.methods.nn._nn_hypermodel import nnHyperModel

class DeepEnsemble:
    """
    Deep Ensemble class for managing M indepependently trained models.
    Plain python class.
    """

    def __init__(self, models: List, heteroscedastic: bool = False) -> None:
        """
        Parameters
        ----------
        models: List
            List of individual Skortch NeuralNetRegressor models.
        heteroscedastic: bool, default=False
            Whether to use heteroscedastic uncertainty quantification.

        Returns
        -------
        None
        """
        self.ensemble_models = models
        self.heteroscedastic = heteroscedastic

    def fit(self, x: Any, y: Any, **kwargs) -> "DeepEnsemble":
        """
        Trains each model in the ensemble to the provided data.

        Parameters
        ----------
        x: Any
            Input features/samples to fit on.
        y: Any
            Target values for the input features.
        **kwargs: dict
            Extra keyword arguments for model fitting.

        Returns
        -------
        self: DeepEnsemble
            The fitted DeepEnsemble instance.
        """
        for member in self.ensemble_models:
            member.fit(x, y, **kwargs)
        return self
    
    def _predict_stacked(self, x: Any) -> np.ndarray:
        """
        Method returning stacked predictions from all ensemble members.

        Parameters
        ----------
        x: Any
            Input features/samples to predict on.
        
        Returns
        -------
        predictions: np.ndarray
            Stacked predictions of shape (n_models, n_samples, n_targets) in standard mode,
            or (n_models, n_samples, 2*n_targets) in heteroscedastic NLL mode.
            Used internally by predict_with_uncertainty. Not meant to be
            called directly by users.
        """
        return np.stack(
            [model.predict(x) for model in self.ensemble_models],
            axis=0,
            )

    def predict(self, x: Any) -> np.ndarray:
        """
        Generates mean prediction across all ensemble members.

        Parameters
        ----------
        x: Any
            Input features/samples to predict on.

        Returns
        -------
        predictions: np.ndarray
            Mean prediction of shape (n_samples, n_targets), consistent
            with standard pyMAISE models. For full uncertainty
            decomposition use predict_with_uncertainty() instead.
        """
        return np.mean(self._predict_stacked(x), axis=0)

    def predict_with_uncertainty(self, x: Any) -> dict:
        """
        Generates predictions for the input samples using all models including variances.

        Aleatoric variance supported when loss ="nll" is set in compile_params.

        Parameters
        ----------
        x: Any
            Input features/samples.
        **kwargs: dict
            Extra keyword arguments for model prediction.

        Returns
        -------
        results: dict
            - **predictions** (np.ndarray, array-like): Stacked raw predictions from
              each ensemble member.

              Shapes:
                - Regression: ``(n_models, n_samples, n_outputs)``
                - Classification: ``(n_models, n_samples, n_classes)``

            - **mean_preds** (np.ndarray, array-like): Ensemble mean predictions.

              Shapes:
                - Regression: ``(n_samples, n_outputs)`` (average predicted target values)
                - Classification: ``(n_samples, n_classes)`` (average predicted class probability distributions)

            - **epistemic_var** (np.ndarray, array-like): Epistemic uncertainty.

              Shapes:
                - Regression: ``(n_samples, n_outputs)`` (variance of predictions across ensemble models)
                - Classification: ``(n_samples,)`` (predictive entropy over the class probabilities, computed as ``-sum(p * log(p))``)

            - **aleatoric_var** (np.ndarray or None): Aleatoric uncertainty.
            Returns None if heteroscedastic=False (default). Returns aleatoric
            variance if heteroscedastic=True (loss ="nll" in compile_params).

            Shapes:
                - Regression (heteroscedastic=True): ``(n_samples, n_outputs)``
                - Regression (heteroscedastic=False): ``None``
                - Classification: ``None`` (NLL not supported for classification in current implementation)
        """
        predictions = self._predict_stacked(x)

        if self.heteroscedastic:
            n_targets = predictions.shape[-1] // 2
            assert predictions.shape[-1] % 2 == 0, "heteroscedastic mode expects 2*n_targets outputs"
            member_means = predictions[:, :, :n_targets]
            member_vars = predictions[:, :, n_targets:]
            mean_preds = np.mean(member_means, axis=0)
            aleatoric_var = np.mean(member_vars, axis=0)
        else:
            mean_preds = np.mean(predictions, axis=0)
            aleatoric_var = None


        if settings.values.problem_type == settings.ProblemType.REGRESSION:
            if self.heteroscedastic:
                # For heteroscedastic regression, epistemic variance is the variance of the mean predictions across ensemble members.
                epistemic_var = np.var(member_means, axis=0)
            else:
                epistemic_var = np.var(predictions, axis=0)
        elif settings.values.problem_type == settings.ProblemType.CLASSIFICATION:
            # For classification, use predictive entropy as the uncertainty measure.
            # entropy = -sum(p * log(p)), higher entropy = more uncertain
            epistemic_var = -np.sum(mean_preds * np.log(mean_preds + 1e-10), axis=-1)
        else:
            raise ValueError("DeepEnsemble.predict_with_uncertainty() only supports regression and classification problems.")
        
        return {
            "predictions": predictions,
            "mean": mean_preds,
            "epistemic_var": epistemic_var,
            "aleatoric_var": aleatoric_var
        }


def _set_random_state(state: int):
    settings.values.random_state = state
    random.seed(state)
    np.random.seed(state)
    torch.manual_seed(state)
    torch.cuda.manual_seed_all(state)


class DeepEnsembleHyperModel(nnHyperModel):
    """
    HyperModel for Deep Ensembles extending pyMAISE's nnHyperModel.
    """

    def __init__(
        self,
        parameters: dict,
        input_shape: Tuple,
        name: str,
        num_models: int = 5,
        tune_ensemble: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        parameters: dict
            Hyperparameter space configuration options.
        input_shape: Tuple
            Shape of the input features.
        name: str
            Name identifier for this model.
        num_models: int, default=5
            Number of models in the ensemble.
        tune_ensemble: bool, default=False
            Whether to tune the ensemble as a whole.

        Returns
        -------
        None
        """
        super(DeepEnsembleHyperModel, self).__init__(parameters, input_shape, name)
        self.num_models = num_models
        self.tune_ensemble = tune_ensemble

    def build(self, trial: Any) -> Any:
        """
        Builds the model(s) for the Deep Ensemble.

        This method is primarily to be called internally by Optuna.

        Parameters
        ----------
        trial: Any
            Trial object from Optuna.

        Returns
        -------
        model: Model
            A Skorch NerualNetRegressor model (either DeepEnsembleWrapper or a single member).
        """
        # Build each individual model and wrap them in a DeepEnsembleModel
        if self.tune_ensemble:
            return self.build_ensemble(trial)

        # Build the single model (best for normal tuning)
        return super(DeepEnsembleHyperModel, self).build(trial)


    def build_ensemble(self, trial: Any) -> DeepEnsemble:
        """
        Build each model in the ensemble, returning the wrapped ensemble model.

        This method is to be called by the user, and it can be called internally
        when `tune_ensemble=True`.

        Parameters
        ----------
        trial: Any
            HyperParameters object

        Returns
        -------
        ensemble_model: DeepEnsemble
            The deep ensemble model.
        """
        # NLL loss not supported for classification
        if self._compilation_params.get("loss") == "nll" and settings.values.problem_type == settings.ProblemType.CLASSIFICATION:
            raise ValueError("NLL loss is not currently supported for classification problems in DeepEnsembleHyperModel.")
        
        models = []
        seed = settings.values.random_state

        # A random state must be set
        if seed is None:
            # Use numpy's random state acorss pyMAISE
            seed = np.random.randint(0, 2**32)

            # Set global random state
            # TODO I have pyMAISE global verbosity set to 1, which will ignore this warning. I'm assuming
            #      that is okay, because this warning is minor.
            warnings.warn(f"Required for building deep ensemble: Random State is None, setting to {seed}.")

            # TODO If we would like to bypass warning settings and always display the seed, this is how
            #      we would do it. (less desirable IMO)
            # with warnings.catch_warnings():
            #     warnings.simplefilter("always")
            #     warnings.warn(f"Required for building deep ensemble: Random State is None, setting to {seed}.")

            _set_random_state(seed)

        for i in range(self.num_models):
            # Set the random seed across all frameworks
            _set_random_state(seed + i)

            # Generate the model based on the new seed
            models.append(super(DeepEnsembleHyperModel, self).build(trial))

        # reset the random state
        _set_random_state(seed)

        # Wrap them in the DeepEnsembleModel
        ensemble_model = DeepEnsemble(
            models=models,
            heteroscedastic=self._compilation_params.get("loss") == "nll"
        )
        return ensemble_model

