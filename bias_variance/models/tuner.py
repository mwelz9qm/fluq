from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import optuna

from bias_variance.models.evaluation import (
    MetricName,
    get_model_predictions,
    get_model_scores,
)
from bias_variance.models.training import Trainer, TrainingConfig
from bias_variance.models.utils import _to_tensor

if TYPE_CHECKING:
    from bias_variance.config import RunBaseline

@dataclass(frozen=True, slots=True)
class TunerConfig:
    n_trials: int = 10
    metric: str = "rmse"

    optimizer_choices: tuple[str, ...] = ("adam",)
    learning_rate_range: tuple[float, float] = (1e-4, 1e-2)
    loss_choices: tuple[str, ...] = ("mse",)
    epoch_choices: tuple[int, ...] = (50, 100, 150)
    batch_size_choices: tuple[int, ...] = (8, 16, 32)

    def __post_init__(self) -> None:
        self._validate_n_trials()
        self._validate_metric()
        self._validate_optimizer_choices()
        self._validate_learning_rate_range()
        self._validate_loss_choices()
        self._validate_epoch_choices()
        self._validate_batch_size_choices()

    def _validate_n_trials(self) -> None:
        if (
                not isinstance(self.n_trials, int)
                or isinstance(self.n_trials, bool)
        ):
            raise TypeError("n_trials must be an integer.")

        if self.n_trials <= 0:
            raise ValueError("n_trials must be greater than 0.")

    def _validate_metric(self) -> None:
        if not isinstance(self.metric, str):
            raise TypeError("metric must be a string.")

        supported_metrics = {"rmse", "mse", "mae", "r2"}

        if self.metric not in supported_metrics:
            raise ValueError(
                f"metric must be one of {sorted(supported_metrics)}."
            )

    @property
    def direction(self) -> str:
        if self.metric == "r2":
            return "maximize"

        return "minimize"

    def _validate_optimizer_choices(self) -> None:
        supported_optimizers = {"adam", "sgd"}

        self._validate_string_choices(
            "optimizer_choices",
            self.optimizer_choices,
            supported_optimizers,
        )

    def _validate_learning_rate_range(self) -> None:
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
        supported_losses = {"mse", "mae"}

        self._validate_string_choices(
            "loss_choices",
            self.loss_choices,
            supported_losses,
        )

    def _validate_epoch_choices(self) -> None:
        self._validate_positive_integer_choices(
            "epoch_choices",
            self.epoch_choices,
        )

    def _validate_batch_size_choices(self) -> None:
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

    def __init__(self, config: TunerConfig | None = None) -> None:
        self.config = config or TunerConfig()

    def tune(
            self,
            baseline: RunBaseline,
            random_state: int | None = None,
    ) -> TrainingConfig:
        sampler = optuna.samplers.TPESampler(seed=random_state)

        study = optuna.create_study(
            direction=self.config.direction,
            sampler=sampler,
        )
        study.optimize(
            lambda trial: self._objective(
                trial,
                baseline,
                random_state,
            ),
            n_trials=self.config.n_trials,
        )

        return self._build_training_config_from_params(study.best_params)

    def _objective(
            self,
            trial,
            baseline: RunBaseline,
            random_state: int | None,
    ) -> float:
        candidate_config = self._build_training_config_from_trial(trial)

        trainer = Trainer(candidate_config)
        trainer.set_fnn_model_builder(
            baseline.X.shape[1],
            baseline.Y.shape[1],
        )

        x_train = _to_tensor(baseline.X_train)
        y_train = _to_tensor(baseline.Y_train)

        x_test = baseline.X_test
        y_test = baseline.Y_test

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

        tuning_metric = MetricName(self.config.metric)
        scores = get_model_scores(
            predictions=predictions,
            y_test=y_test,
            metrics=frozenset({tuning_metric}),
        )

        return scores[tuning_metric.value]

    def _build_training_config_from_params(
        self,
        params: dict[str, object],
    ) -> TrainingConfig:
        return TrainingConfig(
            optimizer=params["optimizer"],
            learning_rate=params["learning_rate"],
            loss=params["loss"],
            epochs=params["epochs"],
            batch_size=params["batch_size"],
        )

    def _build_training_config_from_trial(self, trial) -> TrainingConfig:
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
