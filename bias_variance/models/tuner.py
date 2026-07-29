"""Hyperparameter tuning utilities for bias-variance studies.

This module contains the tuner objects used to create a TrainingConfig when
the user does not provide one directly.
"""

from __future__ import annotations

from dataclasses import dataclass

from bias_variance.models.training import TrainingConfig


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

    def tune(self) -> TrainingConfig:
        """Run tuning and return the best TrainingConfig.

        Returns
        -------
        TrainingConfig
            Training configuration created from the best tuning result.
        """
        raise NotImplementedError

    def _objective(self, trial) -> float:
        """Evaluate one tuning trial.

        This method should eventually:
        1. Build a candidate TrainingConfig from the trial.
        2. Train a model using that candidate config.
        3. Evaluate the model.
        4. Return the selected score for Optuna.

        The full implementation depends on the Trainer prediction/evaluation
        workflow.
        """
        candidate_config = self._build_training_config_from_trial(trial)

        raise NotImplementedError(
            "_objective() still needs trainer, data split, architecture, "
            "prediction, and scoring workflow."
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