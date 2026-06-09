import absl.logging

# This hides the annoying red warnings caused by Tensor Flow; since pyMAISE is shifting
# to Pytorch, I am not going to worry about finding a way to make this less hack-y.
absl.logging.set_verbosity(absl.logging.ERROR)

from typing import Any, Dict, List, Optional, Tuple
import random
import numpy as np
import warnings

import tensorflow as tf
from tensorflow.keras import Model # type: ignore
from pyMAISE import settings
# TODO: change back to pyMAISE import once NLL changes are merged into pyMAISE
# from pyMAISE.methods.nn._nn_hypermodel import nnHyperModel
from ._nn_hypermodel import nnHyperModel, gaussian_nll_loss  # local copy for NLL development


class DeepEnsembleWrapper(Model):
    """
    Wrapper class for Deep Ensembles extending tf.keras.Model.
    Acts as a single Keras model that manages the ensemble.
    """

    def __init__(self, models: List[Model], heteroscedastic: bool = False, **kwargs) -> None:
        """
        Wrapper class for Deep Ensembles extending tf.keras.Model.
        Acts as a single Keras model that manages the ensemble.

        Parameters
        ----------
        models: List[Model]
            List of individual Keras models to manage in the ensemble.
        heteroscedastic: bool, default=False
            Whether to use heteroscedastic uncertainty quantification.
        **kwargs: dict
            Standard keyword arguments for tf.keras.Model.

        Returns
        -------
        None
        """
        super(DeepEnsembleWrapper, self).__init__(**kwargs)
        self.ensemble_models = models
        self.heteroscedastic = heteroscedastic

    def compile(self, **kwargs) -> None:
        """
        Compiles all underlying models in the ensemble.

        Parameters
        ----------
        **kwargs: dict
            Compilation configurations such as optimizer, loss, and metrics.

        Returns
        -------
        None
        """
        # Compile the single parent model.
        wrapper_kwargs = kwargs.copy()
        # Single Keras Model does not expect loss and metrics
        wrapper_kwargs.pop("loss", None)
        wrapper_kwargs.pop("metrics", None)
        super(DeepEnsembleWrapper, self).compile(**wrapper_kwargs)

        # Compile the individual models.
        optimizer = kwargs.get("optimizer")
        for model in self.ensemble_models:
            model_kwargs = kwargs.copy()
            if optimizer is not None and hasattr(optimizer, "from_config"):
                model_kwargs["optimizer"] = optimizer.__class__.from_config(optimizer.get_config())
            model.compile(**model_kwargs)

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Forward pass for the ensemble.

        Parameters
        ----------
        inputs: tf.Tensor
            Input tensor for the models.
        training: bool, default=False
            Whether the models are called in training mode.

        Returns
        -------
        predictions: tf.Tensor
            Stacked predictions from all ensemble models.
        """
        predictions = tf.stack(
            [model(inputs, training=training) for model in self.ensemble_models],
            axis=0,
        )
        return predictions

    def train_step(self, data: Tuple) -> Dict[str, tf.Tensor]:
        """
        Custom training step that trains all models in the ensemble on the batch.

        Parameters
        ----------
        data: Tuple
            Training data batch containing inputs and targets.

        Returns
        -------
        avg_metrics: Dict[str, tf.Tensor]
            Average metrics across all ensemble models.
        """
        all_metrics = []
        for model in self.ensemble_models:
            all_metrics.append(model.train_step(data))

        # Average the metrics across all models
        avg_metrics = {}
        if all_metrics:
            for key in all_metrics[0].keys():
                avg_metrics[key] = tf.reduce_mean([m[key] for m in all_metrics])
        return avg_metrics

    def test_step(self, data: Tuple) -> Dict[str, tf.Tensor]:
        """
        Custom test/evaluation/validation step that evaluates all models in the ensemble.

        Parameters
        ----------
        data: Tuple
            Test/evaluation data batch containing inputs and targets.

        Returns
        -------
        avg_metrics: Dict[str, tf.Tensor]
            Average metrics across all ensemble models.
        """
        all_metrics = []
        for model in self.ensemble_models:
            all_metrics.append(model.test_step(data))

        # Average the metrics across all models
        avg_metrics = {}
        if all_metrics:
            for key in all_metrics[0].keys():
                avg_metrics[key] = tf.reduce_mean([m[key] for m in all_metrics])
        return avg_metrics

    def save_weights(self, filepath: str, *args, **kwargs) -> None:
        """
        Saves the weights of all underlying models in the ensemble.

        Parameters
        ----------
        filepath: str
            Base filepath for saving weights.
        *args: tuple
            Extra positional arguments for weight saving.
        **kwargs: dict
            Extra keyword arguments for weight saving.

        Returns
        -------
        None
        """
        for i, model in enumerate(self.ensemble_models):
            # ensure both Keras 2 and 3 compatibility
            if str(filepath).endswith(".weights.h5"):
                p = str(filepath)[:-11] + f"_model_{i}.weights.h5"
            elif str(filepath).endswith(".h5"):
                p = str(filepath)[:-3] + f"_model_{i}.h5"
            else:
                p = f"{filepath}_model_{i}"
            model.save_weights(p, *args, **kwargs)

    def load_weights(self, filepath: str, *args, **kwargs) -> None:
        """
        Loads the weights of all underlying models in the ensemble.

        Parameters
        ----------
        filepath: str
            Base filepath for loading weights.
        *args: tuple
            Extra positional arguments for weight loading.
        **kwargs: dict
            Extra keyword arguments for weight loading.

        Returns
        -------
        None
        """
        for i, model in enumerate(self.ensemble_models):

            # Ensure stricter Keras 3 compatibility, which always expects ".weights.h5" filename
            if str(filepath).endswith(".weights.h5"):
                p = str(filepath)[:-11] + f"_model_{i}.weights.h5"
            elif str(filepath).endswith(".h5"):
                p = str(filepath)[:-3] + f"_model_{i}.h5"
            else:
                p = f"{filepath}_model_{i}"
            status = model.load_weights(p, *args, **kwargs)
            if status is not None:
                status.expect_partial()
    
    def _predict_stacked(self, x: Any, **kwargs) -> tf.Tensor:
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
        predictions: tf.Tensor
            Stacked predictions of shape (n_models, n_samples, n_targets).
            Used internally by predict_with_uncertainty. Not meant to be
            called directly by users.
        """
        return tf.stack(
            [model.predict(x, **kwargs) for model in self.ensemble_models],
            axis=0,
            )

    def predict(self, x: Any, **kwargs) -> tf.Tensor:
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
        predictions: tf.Tensor
            Mean prediction of shape (n_samples, n_targets), consistent
            with standard Keras and pyMAISE models. For full uncertainty
            decomposition use predict_with_uncertainty() instead.
        """
        return tf.reduce_mean(self._predict_stacked(x, **kwargs), axis=0)

    def predict_with_uncertainty(
        self, x: Any, **kwargs
    ) -> dict:
        """
        Generates predictions for the input samples using all models including variances.

        Aleatoric variance supported when loss_func="nll" is set in compile_params.

        Parameters
        ----------
        x: Any
            Input features/samples.
        **kwargs: dict
            Extra keyword arguments for model prediction.

        Returns
        -------
        results: dict
            - **predictions** (tf.Tensor, array-like): Stacked raw predictions from
              each ensemble member.

              Shapes:
                - Regression: ``(n_models, n_samples, n_outputs)``
                - Classification: ``(n_models, n_samples, n_classes)``

            - **mean_preds** (tf.Tensor, array-like): Ensemble mean predictions.

              Shapes:
                - Regression: ``(n_samples, n_outputs)`` (average predicted target values)
                - Classification: ``(n_samples, n_classes)`` (average predicted class probability distributions)

            - **epistemic_var** (tf.Tensor, array-like): Epistemic uncertainty.

              Shapes:
                - Regression: ``(n_samples, n_outputs)`` (variance of predictions across ensemble models)
                - Classification: ``(n_samples,)`` (predictive entropy over the class probabilities, computed as ``-sum(p * log(p))``)

            - **aleatoric_var** (tf.Tensor or None): Aleatoric uncertainty.
            Returns None if heteroscedastic=False (default). Returns aleatoric
            variance if heteroscedastic=True (loss_func="nll" in compile_params).

            Shapes:
                - Regression (heteroscedastic=True): ``(n_samples, n_outputs)``
                - Regression (heteroscedastic=False): ``None``
                - Classification: ``None``
        """
        predictions = self._predict_stacked(x, **kwargs)

        if self.heteroscedastic:
            n_targets = predictions.shape[-1] // 2
            member_means = predictions[:, :, :n_targets]
            member_vars = predictions[:, :, n_targets:]
            mean_preds = tf.reduce_mean(member_means, axis=0)
            aleatoric_var = tf.reduce_mean(member_vars, axis=0)
        else:
            mean_preds = tf.reduce_mean(predictions, axis=0)
            aleatoric_var = None


        if settings.values.problem_type == settings.ProblemType.REGRESSION:
            if self.heteroscedastic:
                # For heteroscedastic regression, epistemic variance is the variance of the mean predictions across ensemble members.
                epistemic_var = tf.math.reduce_variance(member_means, axis=0)
            else:
                epistemic_var = tf.math.reduce_variance(predictions, axis=0)
        elif settings.values.problem_type == settings.ProblemType.CLASSIFICATION:
            # For classification, use predictive entropy as the uncertainty measure.
            # entropy = -sum(p * log(p)), higher entropy = more uncertain
            epistemic_var = -tf.reduce_sum(mean_preds * tf.math.log(mean_preds + 1e-10), axis=-1)
        else:
            raise ValueError("DeepEnsembleWrapper.predict_with_uncertainty() only supports regression and classification problems.")
        
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
    tf.random.set_seed(state)


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

    def build(self, hp: Any) -> Model:
        """
        Builds the model(s) for the Deep Ensemble.

        This method is primarily to be called internally by Keras Tuner.

        Parameters
        ----------
        hp: Any
            HyperParameters object from Keras Tuner.

        Returns
        -------
        model: Model
            A compiled Keras Model (either DeepEnsembleWrapper or a single member).
        """
        # Build each individual model and wrap them in a DeepEnsembleModel
        if self.tune_ensemble:
            return self.build_ensemble(hp)

        # Build the single model (best for normal tuning)
        return super(DeepEnsembleHyperModel, self).build(hp)


    def build_ensemble(self, hp: Any) -> DeepEnsembleWrapper:
        """
        Build each model in the ensemble, returning the wrapped ensemble model.

        This method is to be called by the user, and it can be called internally by
        Keras Tuner when `tune_ensemble=True`.

        Parameters
        ----------
        hp: Any
            HyperParameters object from Keras Tuner.

        Returns
        -------
        ensemble_model: DeepEnsembleWrapper
            The compiled deep ensemble wrapper model.
        """
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
            models.append(super(DeepEnsembleHyperModel, self).build(hp))

        # reset the random state
        _set_random_state(seed)

        # Wrap them in the DeepEnsembleModel
        ensemble_model = DeepEnsembleWrapper(
            models=models,
            name=f"{self._name}_ensemble",
            heteroscedastic=self._compilation_params.get("loss_func") == "nll"
        )

        # Compile the ensemble — strip loss_func (pyMAISE-only key) and inject
        # gaussian_nll_loss so Keras doesn't receive an unknown kwarg.
        compile_params = {k: v for k, v in self._compilation_params.items() if k != "loss_func"}
        if self._compilation_params.get("loss_func") == "nll":
            compile_params["loss"] = gaussian_nll_loss
        compile_params["optimizer"] = self._get_optimizer(hp)
        ensemble_model.compile(**compile_params)

        return ensemble_model

