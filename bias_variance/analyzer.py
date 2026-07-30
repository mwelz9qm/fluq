from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from bias_variance.generators.base import Generator
from bias_variance.models.evaluation import (
    EvaluationMethod,
    Evaluator,
    MetricName,
    get_model_predictions,
    get_model_scores,
)
from bias_variance.models.fnn import FnnArchitecture
from bias_variance.models.training import Trainer, TrainingConfig


class StudyName(StrEnum):
    MODEL = 'model'
    SAMPLING = 'sampling'
    DATA = 'data'


@dataclass(frozen=True, slots=True)
class RunRecord:
    run_id: str
    created_at: datetime
    n_iter: int
    test_size: float
    test_metrics: tuple[str, ...]
    optimizer: str
    learning_rate: float
    loss: str
    epochs: int
    batch_size: int
    device: str
    base_architecture: tuple[int, ...]
    base_x_train: tuple[float]
    base_y_train: tuple[float]
    base_x_test: tuple[float]
    base_y_test: tuple[float]
    input_columns: tuple[str, ...]
    output_columns: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class StudyRecord:
    study_id: str
    run_id: str
    study_name: str
    evaluation_method: str


@dataclass(frozen=True, slots=True)
class GroupRecord:
    group_id: str
    study_id: str
    group_name: str
    averaging_strategy_bias: float | None
    averaging_strategy_variance: float | None
    pointwise_strategy_bias: float | None
    pointwise_strategy_variance: float | None


@dataclass(frozen=True, slots=True)
class ModelRecord:
    model_id: str
    group_id: str
    architecture: tuple[int, ...]
    test_scores: Mapping[str, float]
    x_train: tuple[float, ...]
    y_train: tuple[float, ...]
    x_test: tuple[float, ...]
    y_test: tuple[float, ...]
    predictions: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class DatasetSplit:
    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.DataFrame
    y_test: pd.DataFrame


@dataclass(frozen=True, slots=True)
class StudyBaseline:
    inputs: pd.DataFrame
    outputs: pd.DataFrame
    split: DatasetSplit
    architecture: FnnArchitecture


@dataclass(frozen=True, slots=True)
class StudyConfig:
    n_iter: int = 100
    test_size: float = 0.2
    test_metrics: frozenset[MetricName]
    random_state: int | None = None
    baseline: StudyBaseline
    evaluation_methods: frozenset[EvaluationMethod] = {EvaluationMethod.AVERAGING, EvaluationMethod.POINTWISE}
    studies: frozenset[StudyName] = {StudyName.MODEL, StudyName.SAMPLING, StudyName.DATA}


class BiasAnalyzer:
    def __init__(
        self,
        result_store,
    ):
        self.result_store = result_store

    def _build_generator(
        self,
        study_name: StudyName,
        evaluation_method: EvaluationMethod,
        test_size: float,
        baseline: StudyBaseline,
    ) -> Generator[FnnArchitecture] | Generator[pd.DataFrame]:
        pass

    def _run_bias_study(
        self,
        study_id: str,
        study_name: StudyName,
        evaluation_method: EvaluationMethod,
        n_iter: int,
        test_size: float,
        test_metrics: frozenset[MetricName],
        resolved_device: torch.device,
        random_state: int | None,
        baseline: StudyBaseline,
        trainer: Trainer,
    ) -> None:
        for i in np.arange(n_iter):
            generator = self._build_generator(study_name, evaluation_method, test_size, baseline)
            variations = generator.generate(random_state=random_state)
            group_ids = set()
            for j, variation in enumerate(variations):
                if i == 0:
                    group_id = ''
                    group_ids.add(group_id)
                    group_record = GroupRecord(
                        group_id=group_id,
                        study_id=study_id,
                        group_name=variation.label,
                        averaging_strategy_bias=None,
                        averaging_strategy_variance=None,
                        pointwise_strategy_bias=None,
                        pointwise_strategy_variance=None
                    )
                    self.result_store.add(group_record)

                if evaluation_method == EvaluationMethod.POINTWISE:
                    if study_name == StudyName.MODEL:
                        trained_model = trainer.train(
                            architecture=variation.generated,
                            x_train=baseline.split.x_train,
                            y_train=baseline.split.y_train,
                            random_state=random_state + j
                        )
                        predictions = trainer.predict(trained_model, baseline.split.x_test)
                        scores = {}
                        model_id = ''
                        model_record = ModelRecord(
                            model_id=model_id,
                            group_id=group_ids[j],
                            architecture=variation.generated.hidden_layers,
                            scores=scores,
                            loss='',
                            x_train=baseline.split.x_train,
                            y_train=baseline.split.y_train,
                            x_test=baseline.split.x_test,
                            y_test=baseline.split.y_test,
                            predictions=predictions
                        )
                        self.result_store.add(model_record)
                        
                    else:
                        trained_model = trainer.train(
                            architecture=baseline.architecture,
                            x_train=variation.generated[baseline.inputs.columns],
                            y_train=variation.generated[baseline.outputs.columns],
                            random_state=random_state + j
                        )
                        predictions = trainer.predict(trained_model, baseline.split.x_test)
                        scores = {}
                        model_id = ''
                        model_record = ModelRecord(
                            model_id=model_id,
                            group_id=group_ids[j],
                            architecture=baseline.architecture.hidden_layers,
                            test_scores=scores,
                            x_train=variation.generated[baseline.inputs.columns],
                            y_train=variation.generated[baseline.outputs.columns],
                            x_test=baseline.split.x_test,
                            y_test=baseline.split.y_test,
                            predictions=predictions
                        )
                        self.result_store.add(model_record)

                else:
                    if study_name == StudyName.MODEL:
                        x_train, x_test, y_train, y_test = train_test_split(
                            baseline.inputs,
                            baseline.outputs,
                            test_size=test_size,
                            random_state=random_state
                        )
                        trained_model = trainer.train(
                            architecture=variation.generated,
                            x_train=x_train,
                            y_train=y_train,
                            random_state=random_state + j
                        )
                        predictions = get_model_predictions(
                            model=trained_model,
                            x_test=np.array(x_test, dtype=np.float32),
                            resolved_device=resolved_device,
                        )
                        scores = get_model_scores(
                            predictions=predictions,
                            y_test=np.array(y_test, dtype=np.float32),
                            metrics=test_metrics,
                        )
                        model_id = ''
                        model_record = ModelRecord(
                            model_id=model_id,
                            group_id=group_ids[j],
                            architecture=variation.generated.hidden_layers,
                            test_scores=scores,
                            x_train=x_train,
                            y_train=y_train,
                            x_test=x_test,
                            y_test=y_test,
                            predictions=predictions
                        )
                        self.result_store.add(model_record)

                    else:
                        x_train, x_test, y_train, y_test = train_test_split(
                            variation.generated[baseline.inputs.columns],
                            variation.generated[baseline.outputs.columns],
                            test_size=test_size,
                            random_state=random_state
                        )
                        trained_model = trainer.train(
                            architecture=baseline.architecture,
                            x_train=variation.generated[baseline.inputs.columns],
                            y_train=variation.generated[baseline.outputs.columns],
                            random_state=random_state + j
                        )
                        predictions = trainer.predict(trained_model, x_test)
                        scores = {}
                        model_id = ''
                        model_record = ModelRecord(
                            model_id=model_id,
                            group_id=group_ids[j],
                            architecture=baseline.architecture.hidden_layers,
                            test_scores=scores,
                            x_train=x_train,
                            y_train=y_train,
                            x_test=x_test,
                            y_test=y_test,
                            predictions=predictions
                        )
                        self.result_store.add(model_record)

    def run_bias_studies(
        self,
        study_config: StudyConfig,
        training_config: TrainingConfig,
    ) -> None:
        run_id = ''
        
        trainer =  Trainer(training_config)
        trainer.set_model_builder(
            study_config.baseline.inputs.shape[1],
            study_config.baseline.outputs.shape[1]
        )
        
        for study_name in study_config.studies:
            for evaluation_method in study_config.evaluation_methods:
                study_id = ''
                self._run_bias_study(
                    study_id,
                    study_name,
                    evaluation_method,
                    study_config.n_iter,
                    study_config.test_size,
                    study_config.test_metrics,
                    training_config.resolved_device,
                    study_config.random_state,
                    study_config.baseline,
                    trainer
                )

                study_record = StudyRecord(
                    study_id=study_id,
                    run_id=run_id,
                    study_name=study_name.value,
                    evaluation_method=evaluation_method.value
                )
                self.result_store.add(study_record)

        evaluator = Evaluator(self.result_store)
        evaluator.evaluate(study_config.evaluation_methods, run_id)

        run_record = RunRecord(
            run_id=run_id,
            created_at=datetime.now(tz=''),
            n_iter=study_config.n_iter,
            test_size=study_config.test_size,
            test_metrics=study_config.test_metrics,
            optimizer=training_config.optimizer,
            learning_rate=training_config.learning_rate,
            train_loss=training_config.loss,
            epochs=training_config.epochs,
            batch_size=training_config.batch_size,
            device=training_config.device,
            base_architecture=study_config.baseline.architecture,
            base_x_train=study_config.baseline.split.x_train.to_numpy(),
            base_y_train=study_config.baseline.split.y_train.to_numpy(),
            base_x_test=study_config.baseline.split.x_test.to_numpy(),
            base_y_test=study_config.baseline.split.y_test.to_numpy(),
            input_columns=study_config.baseline.inputs.columns,
            output_columns=study_config.baseline.outputs.columns
        )
        self.result_store.add(run_record)
        self.result_store.commit()
