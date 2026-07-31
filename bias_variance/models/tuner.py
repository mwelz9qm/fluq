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