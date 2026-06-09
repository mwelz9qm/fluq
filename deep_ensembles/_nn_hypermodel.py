import copy
import re

import numpy as np
import tensorflow as tf
from keras_tuner import HyperModel
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import (
    SGD,
    Adadelta,
    Adafactor,
    Adagrad,
    Adam,
    Adamax,
    AdamW,
    Ftrl,
    Nadam,
    RMSprop,
)

import pyMAISE.settings as settings
from pyMAISE.methods.nn._conv1d import Conv1DLayer
from pyMAISE.methods.nn._conv2d import Conv2DLayer
from pyMAISE.methods.nn._conv3d import Conv3DLayer
from pyMAISE.methods.nn._dense import DenseLayer
from pyMAISE.methods.nn._dropout import DropoutLayer
from pyMAISE.methods.nn._flatten import FlattenLayer
from pyMAISE.methods.nn._gru import GRULayer
from pyMAISE.methods.nn._lstm import LSTMLayer
from pyMAISE.methods.nn._max_pooling_1d import MaxPooling1DLayer
from pyMAISE.methods.nn._max_pooling_2d import MaxPooling2DLayer
from pyMAISE.methods.nn._max_pooling_3d import MaxPooling3DLayer
from pyMAISE.methods.nn._reshape import ReshapeLayer
from pyMAISE.methods.nn._variational import VariationalLayer
from pyMAISE.utils.hyperparameters import Choice, HyperParameters


_KL_SUPPORTED_SCHEDULES = ("sigmoid_decay", "sigmoid_growth", "linear")


class _KLAnnealingCallback(tf.keras.callbacks.Callback):
    """Keras callback that updates the KL-divergence weight at the start of each epoch.

    The weight is stored in a shared mutable list (``kl_weight_container``) that is
    captured by the divergence-function lambdas inside every ``VariationalLayer``.
    Updating ``kl_weight_container[0]`` therefore immediately changes the effective
    KL weight used during the forward pass of the next batch.

    Schedules
    ---------
    sigmoid_decay
        KL weight starts near ``max_weight`` and decays to ``min_weight`` by
        roughly 15 % of the way through training.  Matches the schedule used in
        the reference viAL implementation.
    sigmoid_growth
        KL weight starts near ``min_weight`` and grows to ``max_weight`` by
        roughly 70 % of the way through training (classic KL annealing).
    linear
        KL weight ramps linearly from ``min_weight`` to ``max_weight`` over all
        epochs.

    Parameters
    ----------
    kl_weight_container : list[float]
        A one-element list shared with the variational-layer divergence functions.
    schedule : str
        One of ``_KL_SUPPORTED_SCHEDULES``.
    total_epochs : int
        Total number of training epochs (used to parameterise the schedule).
    max_weight : float
        Maximum / target KL weight (typically ``1 / n_train``).
    min_weight : float, optional
        Minimum KL weight.  Defaults to ``max_weight * 0.01``.
    """

    def __init__(self, kl_weight_container, schedule, total_epochs, max_weight, min_weight=None):
        super().__init__()
        self._container = kl_weight_container
        self._schedule = schedule
        self._n = total_epochs
        self._max = max_weight
        self._min = min_weight if min_weight is not None else max_weight * 0.01

    def on_epoch_begin(self, epoch, logs=None):
        self._container[0] = self._compute(epoch)

    def _compute(self, epoch):
        e, n = epoch, self._n
        lo, hi = self._min, self._max
        if self._schedule == "sigmoid_decay":
            return hi / (1.0 + np.exp(2.0 * (e - 0.15 * n))) + lo
        elif self._schedule == "sigmoid_growth":
            return hi / (1.0 + np.exp(-2.0 * (e - 0.7 * n))) + lo
        else:  # linear
            return lo + (hi - lo) * min(e / max(n - 1, 1), 1.0)

def gaussian_nll_loss(y_true, y_pred):
    """
    Gaussian Negative Log Likelihood loss function for heteroscedastic regression.

    Used when loss_func="nll" is specified in compile_params. Trains the model
    to predict both a mean and a variance for each output, enabling aleatoric
    variance estimation.

    Parameters
    ----------
    y_true : tf.Tensor, shape (n_samples, n_targets)
        True target values.
    y_pred : tf.Tensor, shape (n_samples, 2*n_targets)
        Predicted means and variances concatenated. First half is means,
        second half is variances.

    Returns
    -------
    loss : tf.Tensor
        Mean NLL loss across all samples and targets.
    """
    #split predictions into mean and variance
    n_targets = tf.shape(y_pred)[-1] // 2
    mean = y_pred[:, :n_targets]
    variance = y_pred[:, n_targets:]

    #small epsilon for numerical stability
    variance = variance + 1e-6

    #compute NLL loss
    #log(variance) penalizes overconfidence
    #(true - mean)^2 / variance penalizes inaccurate means
    return 0.5 * tf.reduce_mean(
        tf.math.log(variance) + tf.math.square(y_true - mean) / variance
    )

class nnHyperModel(HyperModel):
    # Dictionary of supported Layers
    layer_dict = {
        "Dense": DenseLayer,
        "Dropout": DropoutLayer,
        "LSTM": LSTMLayer,
        "GRU": GRULayer,
        "Conv1D": Conv1DLayer,
        "Conv2D": Conv2DLayer,
        "Conv3D": Conv3DLayer,
        "MaxPooling1D": MaxPooling1DLayer,
        "MaxPooling2D": MaxPooling2DLayer,
        "MaxPooling3D": MaxPooling3DLayer,
        "Flatten": FlattenLayer,
        "Reshape": ReshapeLayer,
        "Variational": VariationalLayer,
    }

    # Dictionary of supported optimizers
    optimizer_dict = {
        "SGD": SGD,
        "RMSprop": RMSprop,
        "Adam": Adam,
        "AdamW": AdamW,
        "Adadelta": Adadelta,
        "Adagrad": Adagrad,
        "Adamax": Adamax,
        "Adafactor": Adafactor,
        "Nadam": Nadam,
        "Ftrl": Ftrl,
    }

    def __init__(self, parameters: dict, input_shape, name):
        # Structure/Architectural hyperparameters
        self._structural_params = parameters["structural_params"]

        # Optimizers and their hyperparameters
        if parameters["optimizer"]:
            self._optimizer = parameters["optimizer"]

            self._optimizer_params = {}

            if isinstance(self._optimizer, Choice):
                for optimizer in self._optimizer.values:
                    assert parameters[optimizer]
                    self._optimizer_params[optimizer] = parameters[optimizer]
            else:
                assert parameters[self._optimizer]
                self._optimizer_params[self._optimizer] = parameters[self._optimizer]
        else:
            raise RuntimeError("Optimizer was not given in `optimizer` key")

        # Model compilation hyperparameters
        self._compilation_params = parameters["compile_params"]

        # Model fitting hyperparameters — strip pyMAISE-specific KL schedule keys
        # so they are never forwarded to model.fit() as unexpected kwargs.
        raw_fitting = parameters["fitting_params"]
        self._kl_schedule = raw_fitting.get("kl_schedule", None)
        self._kl_schedule_min_weight = raw_fitting.get("kl_schedule_min_weight", None)
        self._fitting_params = {
            k: v
            for k, v in raw_fitting.items()
            if k not in ("kl_schedule", "kl_schedule_min_weight")
        }

        # When a KL schedule is requested, create a shared mutable container
        # that will be captured by VariationalLayer divergence-function lambdas
        # and updated each epoch by KLAnnealingCallback.
        if self._kl_schedule is not None:
            if self._kl_schedule not in _KL_SUPPORTED_SCHEDULES:
                raise ValueError(
                    f"Unknown kl_schedule {self._kl_schedule!r}. "
                    f"Choose from {_KL_SUPPORTED_SCHEDULES}."
                )
            self._kl_weight_container = [self._find_base_kl_weight(parameters["structural_params"])]
        else:
            self._kl_weight_container = None

        # Input data shape
        self._input_shape = input_shape

        # Model name
        self._name = name

    # ==========================================================================
    # Methods
    def build(self, hp):
        # Sequential keras neural network
        model = Sequential(name=self._name)

        # Add input layer
        model.add(Input(shape=self._input_shape, name=self._name + "_Input"))

        # Iterating though archetecture
        for layer_name in self._structural_params.keys():
            model = self._build_tree(
                model, layer_name, self._structural_params[layer_name], hp
            )
        
        # Check if NLL loss is requested
        compile_params = copy.deepcopy(self._compilation_params)
        if compile_params.get("loss_func", None) == "nll":
            # Sequential can't branch, so convert to a Functional model.
            # Mean and variance heads both connect to the last hidden layer
            # (not stacked on each other) so they predict independently.
            # We rebuild by piping the existing layers through a new Input.
            n_targets = model.output_shape[-1]
            nll_input = Input(shape=self._input_shape, name=self._name + "_Input")
            x = nll_input
            for layer in model.layers[:-1]:   # all except the user's output Dense
                x = layer(x)
            mean_out = model.layers[-1](x)    # user's Dense becomes the mean head
            var_out = Dense(n_targets, activation="softplus", name="variance_head")(x)
            combined = tf.keras.layers.Concatenate(name="nll_output")([mean_out, var_out])
            model = tf.keras.Model(inputs=nll_input, outputs=combined, name=self._name)
            compile_params["loss"] = gaussian_nll_loss
            compile_params.pop("loss_func")

        # Compile Model
        compile_params["optimizer"] = self._get_optimizer(hp)
        model.compile(**compile_params)
        return model

    def _build_tree(self, model, layer_name, structural_params, hp):
        # Get layer object
        layer = copy.deepcopy(self._get_layer(layer_name, structural_params))

        # Run through all number of layers
        for _ in range(layer.num_layers(hp)):
            # Check if there's a wrapper (TimeDistributed, Bidirectional)
            wrapper_data = layer.wrapper()
            if wrapper_data is not None:
                model.add(wrapper_data[0](layer.build(hp), **wrapper_data[1]))
            else:
                model.add(layer.build(hp))

            # Check for a sublayer
            sublayer_data = layer.sublayer(hp)
            if sublayer_data is not None and sublayer_data[1]:
                model = self._build_tree(model, sublayer_data[0], sublayer_data[1], hp)

            # Increment current layer
            layer.increment_layer()

        # Reset the layer object
        layer.reset()

        return model

    # Fit function for keras-tuner to allow hyperparameter tuning of fitting parameters
    def fit(self, hp, model, x, y, **kwargs):
        fitting_params = copy.deepcopy(self._fitting_params)
        for key, value in self._fitting_params.items():
            if isinstance(value, HyperParameters):
                fitting_params[key] = value.hp(hp, key)

        callbacks = list(kwargs.pop("callbacks", None) or [])
        if self._kl_schedule is not None and self._kl_weight_container is not None:
            epochs = fitting_params.get("epochs", 100)
            callbacks.append(
                _KLAnnealingCallback(
                    kl_weight_container=self._kl_weight_container,
                    schedule=self._kl_schedule,
                    total_epochs=epochs,
                    max_weight=self._find_base_kl_weight(self._structural_params),
                    min_weight=self._kl_schedule_min_weight,
                )
            )

        return model.fit(
            x,
            y,
            **fitting_params,
            callbacks=callbacks if callbacks else None,
            verbose=settings.values.verbosity,
            **kwargs,
        )

    # Update parameters after tuning, a common use case is increasing
    # the number of epochs
    def set_params(self, parameters: dict = None):
        if "structural_params" in parameters:
            for key, value in parameters["structural_params"].items():
                assert key in self._structural_params
                for param_key, param_value in value.items():
                    self._structural_params[key][param_key] = param_value

        elif "optimizer" in parameters:
            self._optimizer = parameters["optimizer"]
            if parameters[self._optimizer]:
                for key, value in parameters[self._optimizer].items():
                    self._optimizer_params[self._optimizer][key] = value

        elif "compile_params" in parameters:
            for key, value in parameters["compile_params"].items():
                self._compilation_params[key] = value

        elif "fitting_params" in parameters:
            for key, value in parameters["fitting_params"].items():
                self._fitting_params[key] = value

    def _get_layer(self, layer_name, structural_params):
        # Search through supported layers dictionary to find layer,
        # if multiple then take the first as the layer
        layer = None
        position = None
        for key, value in self.layer_dict.items():
            match_idx = re.search(key, layer_name)
            if match_idx is not None and (
                position is None or match_idx.span()[0] > position
            ):
                layer = value
                position = match_idx.span()[0]

        if layer is not None:
            if layer is VariationalLayer and self._kl_weight_container is not None:
                # Inject the shared container so the divergence lambdas can be
                # updated each epoch by KLAnnealingCallback.
                structural_params = {**structural_params, "_kl_weight_container": self._kl_weight_container}
            return layer(layer_name, structural_params)
        else:
            # If not found we throw an error
            raise RuntimeError(f"Layer ({layer_name}) is not supported")

    @staticmethod
    def _find_base_kl_weight(structural_params):
        """Return the kl_weight from the first Variational layer found, or 1.0."""
        for layer_name, params in structural_params.items():
            if re.search("Variational", layer_name):
                return params.get("kl_weight", 1.0)
        return 1.0

    def _get_optimizer(self, hp):
        # Get optimizer name
        optimizer = copy.deepcopy(self._optimizer)
        if isinstance(self._optimizer, Choice):
            optimizer = optimizer.hp(hp, "optimizer")

        # Make sure the optimizer parameters were given by the user
        assert self._optimizer_params[optimizer]

        # Copy data and sample hyperparameters
        sampled_data = copy.deepcopy(self._optimizer_params[optimizer])
        for key, value in sampled_data.items():
            if isinstance(value, HyperParameters):
                sampled_data[key] = value.hp(hp, "_".join([optimizer, key]))

        # Search for support optimizer
        if optimizer in self.optimizer_dict:
            return self.optimizer_dict[optimizer](**sampled_data)

        # If the optimizer name doesn't exit in supported optimizer
        # dictionary throw error
        raise RuntimeError(f"Optimizer ({optimizer}) is not supported")

    def get_hyperparameters(self):
        hps = []

        def search_dict(d):
            for _, v in d.items():
                if isinstance(v, HyperParameters):
                    hps.append(v)

                elif isinstance(v, dict):
                    search_dict(v)

        for d in [
            self._structural_params,
            self._compilation_params,
            self._fitting_params,
            self._optimizer_params,
        ]:
            search_dict(d)

        return hps

    @property
    def is_variational(self):
        """Return True if any layer in the architecture is a Variational layer."""
        for key in self._structural_params.keys():
            if re.search("Variational", key) is not None:
                return True
        return False
