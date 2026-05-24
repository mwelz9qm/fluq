import absl.logging
# This hides the annoying red warnings caused by Tensor Flow; since pyMAISE is shifting
# to Pytorch, I am not going to worry about finding a way to make this less hack-y.
absl.logging.set_verbosity(absl.logging.ERROR)

import tensorflow as tf
from tensorflow.keras import Model

from pyMAISE.methods.nn._nn_hypermodel import nnHyperModel


class DeepEnsembleWrapper(Model):
    """
    Wrapper class for Deep Ensembles extending tf.keras.Model.
    Acts as a single Keras model that manages the ensemble.
    """

    def __init__(self, models: list, **kwargs):
        super(DeepEnsembleWrapper, self).__init__(**kwargs)
        self.ensemble_models = models

    def compile(self, **kwargs):
        """
        Compiles all underlying models in the ensemble.
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

    def call(self, inputs, training=False):
        """
        Forward pass for the ensemble.
        Returns the stacked predictions from all models.
        """
        predictions = tf.stack(
            [model(inputs, training=training) for model in self.ensemble_models],
            axis=0,
        )
        return predictions

    def train_step(self, data):
        """
        Custom training step that trains all models in the ensemble on the batch.
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

    def test_step(self, data):
        """
        Custom test/evaluation/validation step that evaluates all models in the ensemble.
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

    def save_weights(self, filepath, *args, **kwargs):
        """
        Saves the weights of all underlying models in the ensemble.
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

    def load_weights(self, filepath, *args, **kwargs):
        """
        Loads the weights of all underlying models in the ensemble.
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

    def predict(self, x, **kwargs):
        """
        Generates predictions for the input samples using all models.
        Returns a tensor of stacked predictions.
        """
        predictions = tf.stack(
            [model.predict(x, **kwargs) for model in self.ensemble_models],
            axis=0,
        )
        return predictions

    def predict_uq(self, x, **kwargs):
        """
        TODO:
            I am not sure if this is the correct place to put this, since all of these methods
            are just being internally called by TF/Keras. It may make more sense to alter
            the PostProcessor to accept the UQ values when it calls the predict method.

        TODO NOTE:
            At the moment, it is not possible to predict aleatoric variance, because the underlying
            pyMAISE HyperModel does not use NLL, and only supports MAE. That will have to be changed.
        """
        pass


class DeepEnsembleHyperModel(nnHyperModel):
    """
    HyperModel for Deep Ensembles extending pyMAISE's nnHyperModel.
    """

    def __init__(self, parameters: dict, input_shape, name: str, num_models: int = 5):
        super(DeepEnsembleHyperModel, self).__init__(parameters, input_shape, name)
        self.num_models = num_models

    def build(self, hp):
        """
        Builds the DeepEnsembleModel consisting of multiple individual models.
        """
        # Build individual models using the parent nnHyperModel's build logic
        models = [
            super(DeepEnsembleHyperModel, self).build(hp)
            for _ in range(self.num_models)
        ]

        # Wrap them in the DeepEnsembleModel
        ensemble_model = DeepEnsembleWrapper(
            models=models,
            name=f"{self._name}_ensemble",
        )

        # Compile the ensemble
        self._compilation_params["optimizer"] = self._get_optimizer(hp)
        ensemble_model.compile(**self._compilation_params)

        return ensemble_model
