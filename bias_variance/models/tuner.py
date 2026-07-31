"""Hyperparameter tuning utilities for bias-variance studies.

This module contains the tuner objects used to create a TrainingConfig when
the user does not provide one directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import optuna
import torch

from bias_variance.models.evaluation import (
    MetricName,
    get_model_predictions,
    get_model_scores,
)
from bias_variance.models.training import Trainer, TrainingConfig

if TYPE_CHECKING:
    from bias_variance.analyzer import RunBaseline

@dataclass(frozen=True, slots=True)
class TunerConfig:
    """Configuration for the hyperparameter tuning step.

    Attributes
    ----------
    n_trials : int
        Number of Optuna trials to run.
    direction : str
        Optimization direction. Usually "minimize" for loss, RMSE, MSE, or MAE.
    metric : str
        Metric used to choose the best trial.
    optimizer_choices : tuple[str, ...]
        Optimizers that Optuna can choose from.
    learning_rate_range : tuple[float, float]
        Minimum and maximum learning rate values.
    loss_choices : tuple[str, ...]
        Loss functions that Optuna can choose from.
    epoch_choices : tuple[int, ...]
        Epoch values that Optuna can choose from.
    batch_size_choices : tuple[int, ...]
        Batch size values that Optuna can choose from.
    """
    n_trials: int = 10
    direction: str = "minimize"
    metric: str = "rmse"

    optimizer_choices: tuple[str, ...] = ("adam",)
    learning_rate_range: tuple[float, float] = (1e-4, 1e-2)
    loss_choices: tuple[str, ...] = ("mse",)
    epoch_choices: tuple[int, ...] = (50, 100, 150)
    batch_size_choices: tuple[int, ...] = (8, 16, 32)

    def __post_init__(self) -> None:
        """Validate tuner configuration values."""
        self._validate_n_trials()
        self._validate_direction()
        self._validate_metric()
        self._validate_optimizer_choices()
        self._validate_learning_rate_range()
        self._validate_loss_choices()
        self._validate_epoch_choices()
        self._validate_batch_size_choices()

    def _validate_n_trials(self) -> None:
        """Validate that n_trials is a positive integer."""
        if (
                not isinstance(self.n_trials, int)
                or isinstance(self.n_trials, bool)
        ):
            raise TypeError("n_trials must be an integer.")

        if self.n_trials <= 0:
            raise ValueError("n_trials must be greater than 0.")

    def _validate_direction(self) -> None:
        """Validate the Optuna optimization direction."""
        if not isinstance(self.direction, str):
            raise TypeError("direction must be a string.")

        if self.direction not in ("minimize", "maximize"):
            raise ValueError(
                "direction must be either 'minimize' or 'maximize'."
            )

    def _validate_metric(self) -> None:
        """Validate the metric used to select the best trial."""
        if not isinstance(self.metric, str):
            raise TypeError("metric must be a string.")

        supported_metrics = {"rmse", "mse", "mae", "r2"}

        if self.metric not in supported_metrics:
            raise ValueError(
                f"metric must be one of {sorted(supported_metrics)}."
            )

    def _validate_optimizer_choices(self) -> None:
        """Validate optimizer choices."""
        supported_optimizers = {"adam", "sgd"}

        self._validate_string_choices(
            "optimizer_choices",
            self.optimizer_choices,
            supported_optimizers,
        )

    def _validate_learning_rate_range(self) -> None:
        """Validate the learning rate search range."""
        if not isinstance(self.learning_rate_range, tuple):
            raise TypeError("learning_rate_range must be a tuple.")

        if len(self.learning_rate_range) != 2:
            raise ValueError(
                "learning_rate_range must contain exactly two bounds."
            )

        lower, upper = self.learning_rate_range

        if (
                not isinstance(lower, float)
                or isinstance(lower, bool)
                or not isinstance(upper, float)
                or isinstance(upper, bool)
        ):
            raise TypeError("learning_rate_range bounds must be floats.")

        if lower <= 0:
            raise ValueError(
                "learning_rate_range lower bound must be greater than 0."
            )

        if lower >= upper:
            raise ValueError(
                "learning_rate_range lower bound must be less than "
                "its upper bound."
            )

    def _validate_loss_choices(self) -> None:
        """Validate loss function choices."""
        supported_losses = {"mse", "mae"}

        self._validate_string_choices(
            "loss_choices",
            self.loss_choices,
            supported_losses,
        )

    def _validate_epoch_choices(self) -> None:
        """Validate epoch choices."""
        self._validate_positive_integer_choices(
            "epoch_choices",
            self.epoch_choices,
        )

    def _validate_batch_size_choices(self) -> None:
        """Validate batch size choices."""
        self._validate_positive_integer_choices(
            "batch_size_choices",
            self.batch_size_choices,
        )

    @staticmethod
    def _validate_string_choices(
            name: str,
            value: tuple[str, ...],
            supported_values: set[str],
    ) -> None:
        """Validate a tuple of supported string choices."""
        if not isinstance(value, tuple):
            raise TypeError(f"{name} must be a tuple.")

        if not value:
            raise ValueError(f"{name} must not be empty.")

        for choice in value:
            if not isinstance(choice, str):
                raise TypeError(f"{name} must contain only strings.")

            if choice not in supported_values:
                raise ValueError(
                    f"{name} contains unsupported value {choice!r}. "
                    f"Expected one of {sorted(supported_values)}."
                )

    @staticmethod
    def _validate_positive_integer_choices(
            name: str,
            value: tuple[int, ...],
    ) -> None:
        """Validate a tuple of positive integer choices."""
        if not isinstance(value, tuple):
            raise TypeError(f"{name} must be a tuple.")

        if not value:
            raise ValueError(f"{name} must not be empty.")

        for choice in value:
            if not isinstance(choice, int) or isinstance(choice, bool):
                raise TypeError(f"{name} must contain only integers.")

            if choice <= 0:
                raise ValueError(
                    f"{name} must contain only positive integers."
                )


class Tuner:
    """Runs hyperparameter tuning and creates a TrainingConfig.

    The tuner is meant to be used as an optional pre-step before running bias
    studies. If the user does not provide a TrainingConfig, the workflow can use
    a Tuner to search for training settings and return a TrainingConfig.
    """

    def __init__(self, config: TunerConfig | None = None) -> None:
        """Initialize the tuner.

        Parameters
        ----------
        config : TunerConfig | None
            Tuner configuration. If None, a default TunerConfig is used.
        """
        self.config = config or TunerConfig()

    def tune(
            self,
            baseline: "RunBaseline",
            test_metrics: frozenset[MetricName],
            random_state: int | None = None,
    ) -> TrainingConfig:
        """Run tuning and return the best TrainingConfig.

        Parameters
        ----------
        baseline : RunBaseline
            Baseline data and architecture used during tuning.
        test_metrics : frozenset[MetricName]
            Metrics used to score each tuning trial.
        random_state : int | None, default = None
            Random seed used during training, if provided.

        Returns
        -------
        TrainingConfig
            Training configuration created from the best tuning result.
        """
        study = optuna.create_study(direction=self.config.direction)
        study.optimize(
            lambda trial: self._objective(
                trial,
                baseline,
                test_metrics,
                random_state,
            ),
            n_trials=self.config.n_trials,
        )

        return self._build_training_config_from_params(study.best_params)

    def _objective(
            self,
            trial,
            baseline: "RunBaseline",
            test_metrics: frozenset[MetricName],
            random_state: int | None,
    ) -> float:
        """Evaluate one tuning trial.

        This method builds one candidate TrainingConfig from the Optuna trial,
        trains a model with the baseline architecture and baseline training
        split, predicts on the baseline test inputs, scores those predictions
        against the baseline test outputs, and returns the selected metric value
        to Optuna.

        Parameters
        ----------
        trial
            Optuna trial object containing suggested hyperparameter values.
        baseline : RunBaseline
            Baseline data split and architecture used for tuning.
        test_metrics : frozenset[MetricName]
            Metrics used to score the candidate model.
        random_state : int | None
            Random seed used during training, if provided.

        Returns
        -------
        float
            Score for the selected tuning metric.
        """
        candidate_config = self._build_training_config_from_trial(trial)

        trainer = Trainer(candidate_config)
        trainer.set_fnn_model_builder(
            baseline.inputs.shape[1],
            baseline.outputs.shape[1],
        )

        x_train = self._to_tensor(baseline.split.x_train)
        y_train = self._to_tensor(baseline.split.y_train)
        x_test = baseline.split.x_test.to_numpy(dtype=np.float32, copy=True)
        y_test = baseline.split.y_test.to_numpy(dtype=np.float32, copy=True)

        trained_model = trainer.train(
            architecture=baseline.architecture,
            x_train=x_train,
            y_train=y_train,
            random_state=random_state,
        )

        predictions = get_model_predictions(
            model=trained_model,
            x_test=x_test,
            resolved_device=candidate_config.resolved_device,
        )

        scores = get_model_scores(
            predictions=predictions,
            y_test=y_test,
            metrics=test_metrics,
        )

        return scores[self.config.metric]

    @staticmethod
    def _to_tensor(data) -> torch.Tensor:
        """Convert tabular data into a float32 torch tensor.

        Parameters
        ----------
        data
            DataFrame or array-like object to convert.

        Returns
        -------
        torch.Tensor
            Float32 tensor created from the provided data.
        """
        return torch.from_numpy(
            data.to_numpy(dtype=np.float32, copy=True)
        )

    def _build_training_config_from_params(
        self,
        params: dict[str, object],
    ) -> TrainingConfig:
        """Create a TrainingConfig from Optuna's best parameters.

        Parameters
        ----------
        params : dict[str, object]
            Best parameter values selected by Optuna.

        Returns
        -------
        TrainingConfig
            Training configuration built from the best parameter values.
        """
        return TrainingConfig(
            optimizer=params["optimizer"],
            learning_rate=params["learning_rate"],
            loss=params["loss"],
            epochs=params["epochs"],
            batch_size=params["batch_size"],
        )

    def _build_training_config_from_trial(self, trial) -> TrainingConfig:
        """Create a TrainingConfig from one Optuna trial.

        Parameters
        ----------
        trial
            Optuna trial object containing suggested hyperparameter values.

        Returns
        -------
        TrainingConfig
            Training configuration built from the trial suggestions.
        """
        optimizer = trial.suggest_categorical(
            "optimizer",
            self.config.optimizer_choices,
        )
        learning_rate = trial.suggest_float(
            "learning_rate",
            self.config.learning_rate_range[0],
            self.config.learning_rate_range[1],
            log=True,
        )
        loss = trial.suggest_categorical(
            "loss",
            self.config.loss_choices,
        )
        epochs = trial.suggest_categorical(
            "epochs",
            self.config.epoch_choices,
        )
        batch_size = trial.suggest_categorical(
            "batch_size",
            self.config.batch_size_choices,
        )

        return TrainingConfig(
            optimizer=optimizer,
            learning_rate=learning_rate,
            loss=loss,
            epochs=epochs,
            batch_size=batch_size,
        )