import warnings
from typing import Any, List, Tuple

import numpy as np
import torch

from pyMAISE import settings
from pyMAISE.methods.nn._nn_hypermodel import nnHyperModel


class DeepEnsembleWrapper:
    """
    Wrapper class for Deep Ensembles.
    Acts as a single model that manages the ensemble of PyTorch/skorch models.
    """

    def __init__(self, models: List[Any]) -> None:
        """
        Parameters
        ----------
        models: List[Any]
            List of individual skorch models to manage in the ensemble.
        """
        self.ensemble_models = models
        # Maintain a history structure similar to skorch to satisfy postprocessor expectations
        self.history = []

    def fit(self, X: Any, y: Any, **fit_params) -> "DeepEnsembleWrapper":
        """
        Custom training step that trains all models in the ensemble on the batch.

        Parameters
        ----------
        X: Any
            Training data inputs.
        y: Any
            Training data targets.

        Returns
        -------
        self
        """
        all_histories = []
        for model in self.ensemble_models:
            model.fit(X, y, **fit_params)

            # Extract history to mimic single skorch model behavior
            train_loss = list(model.history[:, "train_loss"])
            try:
                val_loss = list(model.history[:, "valid_loss"])
            except KeyError:
                val_loss = []

            all_histories.append({"train_loss": train_loss, "valid_loss": val_loss})

        # Average the metrics across all models so PostProcessor can plot a single curve
        if all_histories:
            # We assume all models trained for the same number of epochs
            n_epochs = len(all_histories[0]["train_loss"])
            avg_train_loss = [
                np.mean([m["train_loss"][i] for m in all_histories])
                for i in range(n_epochs)
            ]

            avg_valid_loss = []
            if all_histories[0]["valid_loss"]:
                avg_valid_loss = [
                    np.mean([m["valid_loss"][i] for m in all_histories])
                    for i in range(n_epochs)
                ]

            # Replicate skorch's array-like history format minimally required by fit() wrapper
            class DummyHistory:
                def __init__(self, t_loss, v_loss):
                    self.t_loss = t_loss
                    self.v_loss = v_loss
                def __getitem__(self, item):
                    if item == (slice(None, None, None), "train_loss"):
                        return self.t_loss
                    if item == (slice(None, None, None), "valid_loss"):
                        if not self.v_loss:
                            raise KeyError("valid_loss")
                        return self.v_loss
                    raise NotImplementedError(f"DummyHistory does not support item {item}")
                    
            self.history = DummyHistory(avg_train_loss, avg_valid_loss)

        return self

    def _predict_stacked(self, x: Any, **kwargs) -> np.ndarray:
        """
        Method returning stacked predictions from all ensemble members.

        Parameters
        ----------
        x: Any
            Input features/samples to predict on.
        **kwargs: dict
            Extra keyword arguments for model prediction.

        Returns
        -------
        predictions: np.ndarray
            Stacked predictions of shape (n_models, n_samples, n_targets).
            Used internally by predict_with_uncertainty.
        """
        # Skorch predict expects torch tensors or arrays, and handles device movements
        return np.stack(
            [model.predict(x, **kwargs) for model in self.ensemble_models],
            axis=0,
        )

    def predict(self, x: Any, **kwargs) -> np.ndarray:
        """
        Generates mean prediction across all ensemble members.

        Parameters
        ----------
        x: Any
            Input features/samples to predict on.
        **kwargs: dict
            Extra keyword arguments for model prediction.

        Returns
        -------
        predictions: np.ndarray
            Mean prediction of shape (n_samples, n_targets).
        """
        return np.mean(self._predict_stacked(x, **kwargs), axis=0)

    def predict_with_uncertainty(self, x: Any, **kwargs) -> dict:
        """
        Generates predictions for the input samples using all models including variances.

        Parameters
        ----------
        x: Any
            Input features/samples.
        **kwargs: dict
            Extra keyword arguments for model prediction.

        Returns
        -------
        results: dict
            - **predictions** (np.ndarray): Stacked raw predictions from
              each ensemble member.
            - **mean** (np.ndarray): Ensemble mean predictions.
            - **epistemic_var** (np.ndarray): Epistemic uncertainty.
            - **aleatoric_var** (None): Aleatoric uncertainty (unsupported currently).
        """
        predictions = self._predict_stacked(x, **kwargs)
        mean_preds = np.mean(predictions, axis=0)

        if settings.values.problem_type == settings.ProblemType.REGRESSION:
            epistemic_var = np.var(predictions, axis=0)
        elif settings.values.problem_type == settings.ProblemType.CLASSIFICATION:
            # For classification, use predictive entropy as the uncertainty measure
            epistemic_var = -np.sum(mean_preds * np.log(mean_preds + 1e-10), axis=-1)
        else:
            raise ValueError(
                "DeepEnsembleWrapper.predict_with_uncertainty() only supports regression and classification problems."
            )

        aleatoric_var = None  # NLL not supported in current nnHyperModel

        return {
            "predictions": predictions,
            "mean": mean_preds,
            "epistemic_var": epistemic_var,
            "aleatoric_var": aleatoric_var,
        }


def _set_random_state(state: int):
    settings.values.random_state = state
    np.random.seed(state)
    torch.manual_seed(state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(state)


class DeepEnsembleHyperModel(nnHyperModel):
    """
    HyperModel for Deep Ensembles extending pyMAISE's PyTorch nnHyperModel.
    """

    def __init__(
        self,
        parameters: dict,
        input_shape: Tuple,
        name: str,
        num_models: int = 5,
        ensemble_mode: bool = False,
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
        ensemble_mode: bool, default=False
            If True, `.build()` will return the full DeepEnsemble wrapper (used during PostProcessing).
            If False, `.build()` will return a single neural network (used during fast tuning).
        """
        super(DeepEnsembleHyperModel, self).__init__(parameters, input_shape, name)
        self.num_models = num_models
        self.ensemble_mode = ensemble_mode

    @classmethod
    def inject_for_postprocessing(
        cls,
        tuner_results: dict,
        model_name: str,
        parameters: dict,
        input_shape: Tuple,
        num_models: int = 5,
    ) -> "DeepEnsembleHyperModel":
        """
        Alternative constructor that creates a DeepEnsembleHyperModel in ensemble_mode
        and automatically injects it into a pyMAISE Tuner results dictionary.

        Parameters
        ----------
        tuner_results: dict
            The output dictionary from pyMAISE's Tuner search.
        model_name: str
            The key inside the tuner_results dictionary to target.
        parameters: dict
            Hyperparameter space configuration options.
        input_shape: Tuple
            Shape of the input features.
        num_models: int, default=5
            Number of models in the ensemble.

        Returns
        -------
        DeepEnsembleHyperModel
            The instantiated model wrapper.
        """
        if model_name not in tuner_results:
            raise KeyError(f"Model name '{model_name}' not found in tuner_results.")

        # Extract the top configurations found by the Tuner
        top_configs, _ = tuner_results[model_name]

        # "Ensemble Mode" instance
        instance = cls(
            parameters=parameters,
            input_shape=input_shape,
            name=model_name,
            num_models=num_models,
            ensemble_mode=True,
        )

        # Mutate the dictionary so pyMAISE PostProcessor sees the ensemble
        tuner_results[model_name] = (top_configs, instance)

        return instance

    def build(self, trial: Any) -> Any:
        """
        Builds the model(s) for the Deep Ensemble.

        Parameters
        ----------
        trial: Any
            Trial object from Optuna/Keras Tuner.

        Returns
        -------
        model: Any
            A compiled DeepEnsembleWrapper or a single skorch NeuralNetRegressor member.
        """
        # Build each individual model and wrap them in a DeepEnsembleModel
        if self.ensemble_mode:
            return self.build_ensemble(trial)

        # Build the single model (best for normal tuning)
        return super(DeepEnsembleHyperModel, self).build(trial)

    def build_ensemble(self, trial: Any) -> DeepEnsembleWrapper:
        """
        Build each model in the ensemble, returning the wrapped ensemble model.

        Parameters
        ----------
        trial: Any
            Trial object from Optuna.

        Returns
        -------
        ensemble_model: DeepEnsembleWrapper
            The configured deep ensemble wrapper model.
        """
        models = []
        seed = settings.values.random_state

        # A random state must be set for repeatable divergence
        if seed is None:
            seed = np.random.randint(0, 2**32)
            warnings.warn(
                f"Required for building deep ensemble: Random State is None, setting to {seed}."
            )
            _set_random_state(seed)

        for i in range(self.num_models):
            # Set a slightly different random seed for each framework so initialization diverges
            _set_random_state(seed + i)
            # Generate the model based on the new seed
            models.append(super(DeepEnsembleHyperModel, self).build(trial))

        # Reset the random state
        _set_random_state(seed)

        # Wrap them in the DeepEnsembleModel
        ensemble_model = DeepEnsembleWrapper(models=models)

        return ensemble_model
