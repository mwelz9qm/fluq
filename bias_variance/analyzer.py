from datetime import UTC, datetime
from os import PathLike
from pathlib import Path
from typing import Self
from uuid import uuid4

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from bias_variance.config import (
    DatasetSplit,
    RunBaseline,
    RunConfig,
    Study,
    StudyName,
    _build_run_config,
)
from bias_variance.generators.base import Generator
from bias_variance.generators.fnn_architecture import FnnArchitectureGenerator
from bias_variance.generators.noise import NoiseGenerator
from bias_variance.generators.sampling import SamplingGenerator
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


class BiasAnalyzer:
    def __init__(
        self,
        db_path: str | PathLike[str] = 'bias_variance.sqlite3',
        *,
        db_timeout: float = 5.0
    ) -> None:
        self.db_path = Path(db_path).expanduser().resolve()
        self.db_timeout = db_timeout

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
                    frame.astype(np.float32).copy()
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
                    frame.astype(np.float32).copy()
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
        store: ResultStore
    ) -> None:
        # create generator for study
        generator = self._create_generator(study)

        # Map variation labels to their persisted group IDs.
        group_ids: dict[str, int] = {}
        for variation_label in study.generator_config.variation_labels:
            # build and store group record
            group_record = GroupRecord(
                study_id=study_id,
                group_name=variation_label
            )
            group_id = store.add(group_record)

            group_ids[variation_label] = group_id

        # build the seed sequence for each model
        seed_sequence = np.random.SeedSequence(random_state)
        model_seeds = seed_sequence.generate_state(n_iter)

        # run model training loop
        for model_seed in model_seeds:
            resolved_seed = int(model_seed)

            # generate the study variations
            variations = generator.generate(random_state=resolved_seed)

            # run variation loop
            for variation in variations:

                # create the model's train-test split and architecture
                split, architecture = self._create_split_and_architecture(
                    study.description,
                    baseline,
                    variation.generated,
                    test_size,
                    resolved_seed
                )

                # build and store model record
                model_record = ModelRecord(
                    group_id=group_ids[variation.label],
                    architecture=architecture.hidden_layers
                )
                model_id = store.add(model_record)

                # train point record building loop
                for inputs, outputs in zip(
                    split.x_train.itertuples(index=False, name=None),
                    split.y_train.itertuples(index=False, name=None),
                ):
                    # build and store the train point record
                    train_point_record = TrainPointRecord(
                        model_id=model_id,
                        run_id=None,
                        inputs=inputs,
                        outputs=outputs,
                    )
                    store.add(train_point_record)

                # train model with model trainer
                trained_model = trainer.train(
                    architecture=architecture,
                    x_train=split.x_train,
                    y_train=split.y_train,
                    random_state=resolved_seed
                )

                # get the model's predictions
                model_predictions = get_model_predictions(
                    model=trained_model,
                    x_test=split.x_test,
                    resolved_device=resolved_device,
                )

                # test point record building loop
                for set_position, (inputs, outputs, row_predictions) in enumerate(
                    zip(
                        split.x_test.itertuples(index=False, name=None),
                        split.y_test.itertuples(index=False, name=None),
                        model_predictions,
                    )
                ):
                    # build and store the test point record
                    test_point_record = TestPointRecord(
                        model_id=model_id,
                        run_id=None,
                        set_position=int(set_position),
                        inputs=inputs,
                        outputs=outputs,
                        predictions=row_predictions
                    )
                    store.add(test_point_record)

                # get the model's scores
                scores = get_model_scores(
                    predictions=model_predictions,
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
                    store.add(score_record)

    def run_studies(
        self,
        inputs: pd.DataFrame,
        outputs: pd.DataFrame,
        *,
        run_config: RunConfig | None = None,
        training_config: TrainingConfig | None = None,
    ) -> Self:
        run_config = run_config or _build_run_config(inputs, outputs)
        training_config = training_config or TrainingConfig()
        
        # maintain result store lifecyle within method call
        with ResultStore(self.db_path, timeout=self.db_timeout) as store:
            # create the database tables
            store.create_tables()
        
            # build and store run record
            run_record = RunRecord(
                run_id=str(uuid4()),
                created_at=datetime.now(UTC),
                n_iter=run_config.n_iter,
                test_size=run_config.test_size,
                test_metrics=tuple(
                    metric.value
                    for metric
                    in run_config.test_metrics
                ),
                optimizer=training_config.optimizer,
                learning_rate=training_config.learning_rate,
                loss=training_config.loss,
                epochs=training_config.epochs,
                batch_size=training_config.batch_size,
                device=training_config.device,
                base_architecture=(
                    run_config.baseline.architecture.hidden_layers
                ),
                input_columns=tuple(
                    str(column)
                    for column
                    in run_config.baseline.inputs.columns
                ),
                output_columns=tuple(
                    str(column)
                    for column
                    in run_config.baseline.outputs.columns
                )
            )
            run_id = store.add(run_record)

            # build model trainer for every study
            trainer =  Trainer(training_config)
            trainer.set_fnn_model_builder(
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
                study_id = store.add(study_record)

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
                    trainer,
                    store
                )

        return self

    def decompose_bias_and_variance(
        self,
        run_id: str | None = None
    ) -> pd.DataFrame:
        with ResultStore(self.db_path, timeout=self.db_timeout) as store:
            store.create_tables()

            if run_id is None:
                run_id = store.get_recent_run()
            
            elif not store.does_run_exist(run_id):
                raise ValueError(
                    f'Run does not exist: {run_id}.'
                )

            if run_id is None:
                raise ValueError(
                    'No runs performed. Call run_studies() to get run.'
                )

            evaluator = Evaluator(store)
            group_updates = evaluator.evaluate(run_id)

            for group_update in group_updates:
                store.update_group(
                    group_update.group_id,
                    group_update.bias,
                    group_update.variance
                )

            result_rows = store.get_bias_variance_results(run_id)
            results = pd.DataFrame(
                result_rows,
                columns=(
                    'study_name',
                    'group_name',
                    'evaluation_method',
                    'bias',
                    'variance',
                ),
            )

        return results
