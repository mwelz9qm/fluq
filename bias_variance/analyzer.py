from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from bias_variance.generators.base import Generator, GeneratorConfig
from bias_variance.generators.fnn_architecture import (
    FnnArchitectureGenerator,
    FnnArchitectureGeneratorConfig,
)
from bias_variance.generators.noise import NoiseGenerator, NoiseGeneratorConfig
from bias_variance.generators.sampling import (
    SamplingGenerator,
    SamplingGeneratorConfig,
)
from bias_variance.models.evaluation import (
    EvaluationMethod,
    Evaluator,
    MetricName,
    get_model_predictions,
    get_model_scores,
)
from bias_variance.models.fnn import FnnArchitecture
from bias_variance.models.training import Trainer, TrainingConfig
from bias_variance.persistence.records import (
    GroupRecord,
    ModelRecord,
    RunRecord,
    ScoreRecord,
    StudyRecord,
    TestPointRecord,
    TrainPointRecord,
)
from bias_variance.persistence.store import ResultStore


class StudyName(StrEnum):
    MODEL = 'model'
    SAMPLING = 'sampling'
    DATA = 'data'


@dataclass(frozen=True, slots=True)
class Study:
    study_name: StudyName
    evaluation_method: EvaluationMethod
    generator_config: GeneratorConfig

    def __post_init__(self) -> None:
        match self.study_name:
            case StudyName.MODEL:
                if not isinstance(self.generator_config, FnnArchitectureGeneratorConfig):
                    raise TypeError(
                        f'Mismatched generator_config type with study_name: {self.generator_config!r}, {self.study_name!r}.'
                    )
                
            case StudyName.SAMPLING:
                if not isinstance(self.generator_config, SamplingGeneratorConfig):
                    raise TypeError(
                        f'Mismatched generator_config type with study_name: {self.generator_config!r}, {self.study_name!r}.'
                    )

            case StudyName.DATA:
                if not isinstance(self.generator_config, NoiseGeneratorConfig):
                    raise TypeError(
                        f'Mismatched generator_config type with study_name: {self.generator_config!r}, {self.study_name!r}.'
                    )

    @property
    def description(self) -> tuple[StudyName, EvaluationMethod]:
        return (self.study_name, self.evaluation_method)


@dataclass(frozen=True, slots=True)
class DatasetSplit:
    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.DataFrame
    y_test: pd.DataFrame

    @property
    def full(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        return (self.x_train, self.x_test, self.y_train, self.y_test)

    @property
    def train_set(
        self,
    ) -> pd.DataFrame:
        return pd.concat([self.x_train, self.y_train])


@dataclass(frozen=True, slots=True)
class RunBaseline:
    inputs: pd.DataFrame
    outputs: pd.DataFrame
    split: DatasetSplit
    architecture: FnnArchitecture

    @property
    def dataset(
        self,
    ) -> pd.DataFrame:
        return pd.concat([self.inputs, self.outputs])


@dataclass(frozen=True, slots=True)
class RunConfig:
    n_iter: int = 100
    test_size: float = 0.2
    test_metrics: frozenset[MetricName]
    random_state: int | None = None
    baseline: RunBaseline
    studies: tuple[Study]


class BiasAnalyzer:
    def __init__(
        self,
        result_store: ResultStore,
    ):
        self.result_store = result_store

    @staticmethod
    def _create_split_and_architecture(
        study_description: tuple[StudyName, EvaluationMethod],
        baseline: RunBaseline,
        generated_variation: pd.DataFrame | FnnArchitecture,
        test_size: float,
        random_state: int | None,
    ) -> tuple[DatasetSplit, FnnArchitecture]:
        match study_description:
            case (StudyName.MODEL, EvaluationMethod.POINTWISE):
                split = tuple(
                    frame.to_numpy(dtype=np.float32, copy=True)
                    for frame in baseline.split.full
                )
                architecture = generated_variation
        
            case (StudyName.MODEL, EvaluationMethod.AVERAGING):
                split = tuple(
                    train_test_split(
                        baseline.inputs,
                        baseline.outputs,
                        test_size=test_size,
                        random_state=random_state
                    )
                )
                architecture = generated_variation
        
            case (_, EvaluationMethod.POINTWISE):
                frames = (
                    generated_variation[baseline.inputs.columns],
                    baseline.split.x_test,
                    generated_variation[baseline.outputs.columns],
                    baseline.split.y_test,
                )
                split = tuple(
                    frame.to_numpy(dtype=np.float32, copy=True)
                    for frame in frames
                )
                architecture = baseline.architecture
        
            case (_, EvaluationMethod.AVERAGING):
                split = tuple(
                    train_test_split(
                        generated_variation[baseline.inputs.columns],
                        generated_variation[baseline.outputs.columns],
                        test_size=test_size,
                        random_state=random_state
                    )
                )
                architecture = baseline.architecture
        
            case _:
                raise ValueError(
                    f'Unknown study_description: {study_description!r}.'
                )

        return DatasetSplit(*split), architecture

    @staticmethod
    def _create_generator(
        study: Study,
    ) -> Generator[FnnArchitecture] | Generator[pd.DataFrame]:
        match study.description:
            case (StudyName.MODEL, _):
                return FnnArchitectureGenerator(
                    settings=study.generator_config,
                )

            case (StudyName.SAMPLING, EvaluationMethod.POINTWISE):
                return SamplingGenerator(
                    settings=study.generator_config,
                )

            case (StudyName.SAMPLING, EvaluationMethod.AVERAGING):
                return SamplingGenerator(
                    settings=study.generator_config,
                )

            case (StudyName.DATA, EvaluationMethod.POINTWISE):
                return NoiseGenerator(
                    settings=study.generator_config,
                )

            case (StudyName.DATA, EvaluationMethod.AVERAGING):
                return NoiseGenerator(
                    settings=study.generator_config,
                )

            case _:
                raise ValueError(
                    f'Unknown study: {study!r}.'
                )

    def _run_study(
        self,
        study_id: int,
        study: Study,
        n_iter: int,
        test_size: float,
        test_metrics: frozenset[MetricName],
        resolved_device: torch.device,
        random_state: int | None,
        baseline: RunBaseline,
        trainer: Trainer,
    ) -> None:
        # create generator for study
        generator = self._create_generator(study)

        # initialize set of group ids
        group_ids = set()
        for variation_label in study.generator_config.variation_labels:
            # build and store group record
            group_record = GroupRecord(
                study_id=study_id,
                group_name=variation_label
            )
            group_id = self.result_store.add(group_record)

            # add the group id
            group_ids.add(group_id)

        # run model training loop
        for _ in np.arange(n_iter):

            # generate the study variations
            variations = generator.generate(random_state=random_state)

            # run variation loop
            for j, variation in enumerate(variations):

                # create the model's train-test split and architecture
                split, architecture = self._create_split_and_architecture(
                    study.description,
                    baseline,
                    variation.generated,
                    test_size,
                    random_state
                )

                # build and store model record
                model_record = ModelRecord(
                    group_id=group_ids[j],
                    architecture=architecture.hidden_layers
                )
                model_id = self.result_store.add(model_record)

                # train point record building loop
                for inputs, outputs in zip(
                    split.x_train.itertuples(),
                    split.y_train.itertuples()
                ):
                    # build and store the train point record
                    train_point_record = TrainPointRecord(
                        model_id=model_id,
                        run_id=None,
                        inputs=inputs,
                        outputs=outputs,
                    )
                    self.result_store.add(train_point_record)

                # train model with model trainer
                trained_model = trainer.train(
                    architecture=architecture,
                    x_train=split.x_train,
                    y_train=split.y_train,
                    random_state=random_state
                )

                # get the model's predictions
                model_predictions = get_model_predictions(
                    model=trained_model,
                    x_test=split.x_test,
                    resolved_device=resolved_device,
                )

                # test point record building loop
                for inputs, outputs, row_predictions in zip(
                    split.x_test.itertuples(),
                    split.y_test.itertuples(),
                    model_predictions
                ):
                    # build and store the test point record
                    test_point_record = TestPointRecord(
                        model_id=model_id,
                        run_id=None,
                        set_position=inputs.index,
                        inputs=inputs,
                        outputs=outputs,
                        predictions=row_predictions
                    )
                    self.result_store.add(test_point_record)

                # get the model's scores
                scores = get_model_scores(
                    predictions=row_predictions,
                    y_test=split.y_test,
                    metrics=test_metrics,
                )

                # score record building loop
                for metric, score in scores.items():
                    # build and store score record
                    score_record = ScoreRecord(
                        model_id=model_id,
                        metric=metric,
                        score=score
                    )
                    self.result_store.add(score_record)

    def run_studies(
        self,
        run_config: RunConfig,
        training_config: TrainingConfig,
    ) -> 'BiasAnalyzer':
        # build and store run record
        run_record = RunRecord(
            run_id='--INSERT UUID--',
            created_at=datetime.now(tz=''),
            n_iter=run_config.n_iter,
            test_size=run_config.test_size,
            test_metrics=run_config.test_metrics,
            optimizer=training_config.optimizer,
            learning_rate=training_config.learning_rate,
            train_loss=training_config.loss,
            epochs=training_config.epochs,
            batch_size=training_config.batch_size,
            device=training_config.device,
            base_architecture=run_config.baseline.architecture,
            base_x_train=run_config.baseline.split.x_train.to_numpy(),
            base_y_train=run_config.baseline.split.y_train.to_numpy(),
            base_x_test=run_config.baseline.split.x_test.to_numpy(),
            base_y_test=run_config.baseline.split.y_test.to_numpy(),
            input_columns=run_config.baseline.inputs.columns,
            output_columns=run_config.baseline.outputs.columns
        )
        run_id = self.result_store.add(run_record)

        # build model trainer for every study
        trainer =  Trainer(training_config)
        trainer.set_model_builder(
            run_config.baseline.inputs.shape[1],
            run_config.baseline.outputs.shape[1]
        )

        # run study loop
        for study in run_config.studies:
            # create and store study record
            study_record = StudyRecord(
                run_id=run_id,
                study_name=study.study_name.value,
                evaluation_method=study.evaluation_method.value
            )
            study_id = self.result_store.add(study_record)

            # run the study
            self._run_study(
                study_id,
                study,
                run_config.n_iter,
                run_config.test_size,
                run_config.test_metrics,
                training_config.resolved_device,
                run_config.random_state,
                run_config.baseline,
                trainer
            )

        # commit all run results and close result store connection
        self.result_store.commit()
        self.result_store.close()

        return self

    def decompose_bias_and_variance(
        self,
        run_id: str | None = None
    ) -> pd.DataFrame:
        if run_id is None:
            run_id = self.result_store.get_recent_run()

        if not run_id:
            raise ValueError(
                'No runs performed. Call run_studies() to get run.'
            )
        
        evaluator = Evaluator(self.result_store)
        group_updates = evaluator.evaluate(run_id)
        for group_update in group_updates:
            self.result_store.update_group(
                group_update.group_id,
                group_update.bias,
                group_update.variance
            )

        results = pd.DataFrame() # TODO: Use result_store to build dataframe with study_name, group_name, evaluation_method, strategy_bias, and strategy_variance.

        return results
