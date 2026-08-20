from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
from os import PathLike
from pathlib import Path
from typing import Literal, Self
from uuid import uuid4

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.model_selection import train_test_split

from bias_variance.config import (
    RunBaseline,
    RunConfigBuilder,
    Study,
    StudyBias,
)
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
from bias_variance.models.tuner import Tuner
from bias_variance.persistence.records import (
    GroupRecord,
    ModelRecord,
    RunRecord,
    ScoreRecord,
    StudyRecord,
    TestPointRecord,
    TrainPointRecord,
)
from bias_variance.persistence.store import ResultStore, StoredRun
from bias_variance.plotting import (
    plot_bias_and_variance as plot_prediction_distribution,
)
from bias_variance.plotting import (
    plot_error_components,
)
from bias_variance.plotting import (
    plot_summary as plot_summary_bars,
)

type OutputSelector = int | str
type PlotKind = Literal['components', 'error_relationship']


@dataclass(frozen=True, slots=True)
class GroupPlotResult:
    """Matplotlib objects and metadata for one plotted result group."""

    run_id: str
    study_id: int
    group_id: int
    group_name: str
    evaluation_method: str
    output_index: int
    output_name: str
    figure: Figure
    prediction_axes: Axes
    metric_axes: Axes | None


class BiasAnalyzer:
    '''
    The BiasAnalyzer constructs a series of studies for analyzing the bias
    and variance across different model variations. The analyzer starts by
    gathering results on the ran studies, evaluating the bias and variance,
    and reviewing the evaluations with plots and tables.

    Attributes
    ----------------
    db_path: str | PathLike[str]
        Path to SQLite database for establishing a connection.
    db_timeout: float, default = 5.0
        Timeout in seconds

    '''
    def __init__(
        self,
        db_path: str | PathLike[str] = 'bias_variance.sqlite3',
        *,
        db_timeout: float = 5.0
    ) -> None:
        self.db_path = Path(db_path).expanduser().resolve()
        self.db_timeout = db_timeout
        self._selected_run_id: str | None = None

    @property
    def selected_run_id(self) -> str | None:
        return self._selected_run_id

    def _require_selected_run(self, store: ResultStore) -> str:
        if self._selected_run_id is None:
            raise RuntimeError(
                'No run is selected. Call run_studies() or select_run(run_id).'
            )

        if not store.does_run_exist(self._selected_run_id):
            raise RuntimeError(
                f'Selected run no longer exists: {self._selected_run_id}.'
            )

        return self._selected_run_id

    def select_run(self, run_id: str) -> Self:
        if not isinstance(run_id, str):
            raise TypeError(
                'run_id must be a string.'
            )
        if not run_id:
            raise ValueError(
                'run_id must not be empty.'
            )

        with ResultStore(self.db_path, timeout=self.db_timeout) as store:
            store.create_tables()
            if not store.does_run_exist(run_id):
                raise ValueError(
                    f'Run does not exist: {run_id}. '
                    'Call get_run_history() to view available runs.'
                )

        self._selected_run_id = run_id
        return self

    def get_run_history(self) -> pd.DataFrame:
        """Return all persisted runs as a newest-first DataFrame."""
        with ResultStore(self.db_path, timeout=self.db_timeout) as store:
            store.create_tables()
            runs = store.get_runs()

        return pd.DataFrame.from_records(
            (asdict(run) for run in runs),
            columns=tuple(field.name for field in fields(StoredRun)),
        )

    @staticmethod
    def _create_split_and_architecture(
        study_description: tuple[StudyBias, EvaluationMethod],
        baseline: RunBaseline,
        generated_variation: pd.DataFrame | FnnArchitecture,
        test_size: float,
        random_state: int | None,
    ) -> tuple[
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame],
        FnnArchitecture,
    ]:
        match study_description:
            case (StudyBias.MODEL, EvaluationMethod.POINTWISE):
                split = tuple(
                    frame.astype(np.float32).copy()
                    for frame in baseline.split
                )
                architecture = generated_variation
        
            case (StudyBias.MODEL, EvaluationMethod.AVERAGING):
                split = tuple(
                    train_test_split(
                        baseline.X,
                        baseline.Y,
                        test_size=test_size,
                        random_state=random_state
                    )
                )
                architecture = generated_variation
        
            case (_, EvaluationMethod.POINTWISE):
                frames = (
                    generated_variation[baseline.X.columns],
                    baseline.X_test,
                    generated_variation[baseline.Y.columns],
                    baseline.Y_test,
                )
                split = tuple(
                    frame.astype(np.float32).copy()
                    for frame in frames
                )
                architecture = baseline.architecture
        
            case (_, EvaluationMethod.AVERAGING):
                split = tuple(
                    train_test_split(
                        generated_variation[baseline.X.columns],
                        generated_variation[baseline.Y.columns],
                        test_size=test_size,
                        random_state=random_state
                    )
                )
                architecture = baseline.architecture
        
            case _:
                raise ValueError(
                    f'Unknown study_description: {study_description!r}.'
                )

        return split, architecture

    def _run_study(
        self,
        study_id: int,
        study: Study,
        method: EvaluationMethod,
        n_iter: int,
        test_size: float,
        test_metrics: frozenset[MetricName],
        resolved_device: torch.device,
        random_state: int | None,
        baseline: RunBaseline,
        trainer: Trainer,
        store: ResultStore
    ) -> None:
        # Map variation labels to their persisted group IDs.
        group_ids: dict[str, int] = {}
        for variation_label in study.variation_generator.variation_labels:
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
            variations = study.variation_generator.generate(random_state=resolved_seed)

            # run variation loop
            for variation in variations:
                study_description = (study.study_bias, method)

                # create the model's train-test split and architecture
                (X_train, X_test, Y_train, Y_test), architecture = self._create_split_and_architecture(
                    study_description,
                    baseline,
                    variation.generated,
                    test_size,
                    resolved_seed
                )

                #------START MODEL BUILD, TRAIN, PREDICT, TEST------

                # build and train model with model trainer
                trained_model = trainer.train(
                    architecture=architecture,
                    x_train=X_train,
                    y_train=Y_train,
                    random_state=resolved_seed
                )

                # get the model's predictions
                model_predictions = get_model_predictions(
                    model=trained_model,
                    x_test=X_test,
                    resolved_device=resolved_device,
                )

                # get model's mean and variance of predictions
                model_mean_prediction = np.mean(
                    np.asarray(model_predictions, dtype=float),
                    axis=0
                )
                model_variance_prediction = np.var(
                    np.asarray(model_predictions, dtype=float),
                    axis=0
                )

                # get the model's scores
                scores = get_model_scores(
                    predictions=model_predictions,
                    y_test=Y_test,
                    metrics=test_metrics,
                    is_uniform=False
                )

                #------END MODEL BUILD, TRAIN, PREDICT, TEST------

                #------START RECORD BUILDING------

                # build and store model record
                model_record = ModelRecord(
                    group_id=group_ids[variation.label],
                    architecture=architecture.hidden_layers,
                    model_mean_prediction=model_mean_prediction,
                    model_variance_prediction=model_variance_prediction
                )
                model_id = store.add(model_record)

                # train point record building loop
                for inputs, outputs in zip(
                    X_train.itertuples(index=False, name=None),
                    Y_train.itertuples(index=False, name=None),
                    strict=True
                ):
                    # build and store the train point record
                    train_point_record = TrainPointRecord(
                        model_id=model_id,
                        run_id=None,
                        input=inputs,
                        output=outputs,
                    )
                    store.add(train_point_record)

                # test point record building loop
                for set_position, (inputs, outputs, row_predictions) in enumerate(
                    zip(
                        X_test.itertuples(index=False, name=None),
                        Y_test.itertuples(index=False, name=None),
                        model_predictions,
                        strict=True
                    )
                ):
                    # build and store the test point record
                    test_point_record = TestPointRecord(
                        model_id=model_id,
                        run_id=None,
                        set_position=int(set_position),
                        input=inputs,
                        output=outputs,
                        prediction=row_predictions
                    )
                    store.add(test_point_record)

                # score record building loop
                for metric, score in scores.items():
                    # build and store score record
                    score_record = ScoreRecord(
                        model_id=model_id,
                        metric=metric,
                        score=score
                    )
                    store.add(score_record)

                #------END RECORD BUILDING------

    def run_studies(
        self,
        X: pd.DataFrame,
        Y: pd.DataFrame,
        run_settings: Mapping[str, object] | None = None,
        *,
        training_config: TrainingConfig | None = None,
    ) -> Self:
        # Build the run config from the run_settings, and
        # inputs (X) and outputs(Y)
        run_config = (
            RunConfigBuilder()
            .set_X(X)
            .set_Y(Y)
            .apply_run_settings(run_settings)
            .build()
        )

        # Tune training hyperparameters when the user does not provide them.
        if training_config is None:
            tuner = Tuner()
            training_config = tuner.tune(
                baseline=run_config.baseline,
                random_state=run_config.random_state,
            )

        # build model trainer for every study
        trainer = Trainer(training_config)
        trainer.set_fnn_model_builder(
            run_config.baseline.X.shape[1],
            run_config.baseline.Y.shape[1]
        )

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
                device=str(training_config.resolved_device),
                base_architecture=(
                    run_config.baseline.architecture.hidden_layers
                ),
                input_columns=tuple(
                    str(column)
                    for column
                    in run_config.baseline.X.columns
                ),
                output_columns=tuple(
                    str(column)
                    for column
                    in run_config.baseline.Y.columns
                )
            )
            run_id = store.add(run_record)

            # Iterate through the methods and studies
            for method in run_config.evaluation_methods:
                for study in run_config.studies:
                    study_record = StudyRecord(
                        run_id=run_id,
                        study_name=study.study_bias.value,
                        evaluation_method=method.value
                    )

                    study_id = store.add(study_record)

                    # assign the base dataset for generating dataset variations
                    match (method, study.variation_generator):
                        case (EvaluationMethod.AVERAGING, NoiseGenerator() | SamplingGenerator()):
                            study.variation_generator.base_dataset = run_config.baseline.dataset

                        case (EvaluationMethod.POINTWISE, NoiseGenerator() | SamplingGenerator()):
                            study.variation_generator.base_dataset = run_config.baseline.train_set

                    self._run_study(
                        study_id,
                        study,
                        method,
                        run_config.n_iter,
                        run_config.test_size,
                        run_config.test_metrics,
                        training_config.resolved_device,
                        run_config.random_state,
                        run_config.baseline,
                        trainer,
                        store
                    )

        self._selected_run_id = run_id

        return self

    def decompose_bias_and_variance(self) -> pd.DataFrame:
        with ResultStore(self.db_path, timeout=self.db_timeout) as store:
            store.create_tables()
            run_id = self._require_selected_run(store)

            result_rows = store.get_bias_variance_results(run_id)
            if not result_rows:
                raise ValueError(
                    f'Run contains no evaluation groups: {run_id}.'
                )

            completion = tuple(
                (row[3] is not None, row[4] is not None)
                for row in result_rows
            )
            if all(
                has_bias and has_variance
                for has_bias, has_variance in completion
            ):
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

            if any(
                has_bias or has_variance
                for has_bias, has_variance in completion
            ):
                raise RuntimeError(
                    f'Run contains partially evaluated results: {run_id}.'
                )

            evaluator = Evaluator(store)
            result = evaluator.evaluate(run_id)
            for evaluation in result.evaluations:
                store.add(evaluation)

            for group_update in result.update_groups:
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

    _PLOT_DATA_COLUMNS = (
        'run_id',
        'study_id',
        'study_name',
        'evaluation_method',
        'group_id',
        'group_name',
        'result_type',
        'record_index',
        'test_set_position',
        'model_id',
        'output_index',
        'output_name',
        'actual_value',
        'mean_prediction',
        'prediction_variance',
        'prediction_std',
        'squared_bias',
        'mse',
    )

    @staticmethod
    def _validated_result_vector(
        values: object,
        *,
        expected_size: int,
        field_name: str,
        group_id: int,
        record_id: int,
        non_negative: bool = False,
    ) -> np.ndarray:
        """Validate one persisted per-output result vector."""
        try:
            array = np.asarray(values, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f'{field_name} for group {group_id}, record {record_id} '
                'must contain numeric values.'
            ) from exc

        if array.ndim != 1 or len(array) != expected_size:
            raise ValueError(
                f'{field_name} for group {group_id}, record {record_id} must '
                f'contain {expected_size} output values; got shape '
                f'{array.shape}.'
            )
        if not np.isfinite(array).all():
            raise ValueError(
                f'{field_name} for group {group_id}, record {record_id} '
                'contains non-finite values.'
            )
        if non_negative:
            scale = max(1.0, float(np.max(np.abs(array), initial=0.0)))
            tolerance = np.finfo(float).eps * scale * 100
            if (array < -tolerance).any():
                raise ValueError(
                    f'{field_name} for group {group_id}, record {record_id} '
                    'contains negative values.'
                )
            array = np.maximum(array, 0.0)
        return array

    def get_bias_variance_plot_data(self) -> pd.DataFrame:
        """Return tidy, multi-output plot data for the selected run.

        The selected run must already have been successfully decomposed. Each
        row represents one pointwise evaluation or model result for one output.
        """
        with ResultStore(self.db_path, timeout=self.db_timeout) as store:
            store.create_tables()
            run_id = self._require_selected_run(store)
            run = store.get_run(run_id)
            if run is None:
                raise RuntimeError(f'Selected run no longer exists: {run_id}.')

            groups = store.get_run_groups(run_id)
            if not groups:
                raise ValueError(
                    f'Run contains no studies or result groups: {run_id}.'
                )

            completion_rows = store.get_bias_variance_results(run_id)
            if len(completion_rows) != len(groups):
                raise RuntimeError(
                    f'Run has incomplete result metadata: {run_id}.'
                )
            if not all(
                row[3] is not None and row[4] is not None
                for row in completion_rows
            ):
                raise RuntimeError(
                    f'Run has not been fully decomposed: {run_id}. Call '
                    'decompose_bias_and_variance() first.'
                )

            output_names = run.output_columns
            if not output_names:
                raise ValueError(f'Run contains no output metadata: {run_id}.')

            prepared_rows: list[dict[str, object]] = []
            for group in groups:
                try:
                    method = EvaluationMethod(group.evaluation_method)
                except ValueError as exc:
                    raise ValueError(
                        'Unsupported evaluation method '
                        f'{group.evaluation_method!r} for group '
                        f'{group.group_id}.'
                    ) from exc

                if method is EvaluationMethod.POINTWISE:
                    records = store.get_test_point_results(group.group_id)
                    result_type = 'test_point'
                else:
                    records = store.get_model_results(group.group_id)
                    result_type = 'model'

                if not records:
                    raise ValueError(
                        f'Group {group.group_id} ({method.value}) contains no '
                        'detailed results.'
                    )

                for record_index, record in enumerate(records):
                    if method is EvaluationMethod.POINTWISE:
                        record_id = record.test_point_position
                        mean = self._validated_result_vector(
                            record.mean,
                            expected_size=len(output_names),
                            field_name='mean',
                            group_id=group.group_id,
                            record_id=record_id,
                        )
                        variance = self._validated_result_vector(
                            record.variance,
                            expected_size=len(output_names),
                            field_name='variance',
                            group_id=group.group_id,
                            record_id=record_id,
                            non_negative=True,
                        )
                        actual = self._validated_result_vector(
                            record.actual,
                            expected_size=len(output_names),
                            field_name='actual',
                            group_id=group.group_id,
                            record_id=record_id,
                        )
                        squared_bias = self._validated_result_vector(
                            record.squared_bias,
                            expected_size=len(output_names),
                            field_name='squared_bias',
                            group_id=group.group_id,
                            record_id=record_id,
                            non_negative=True,
                        )
                        mse = np.full(len(output_names), np.nan)
                        test_set_position: int | None = record_id
                        model_id: int | None = None
                    else:
                        record_id = record.model_id
                        mean = self._validated_result_vector(
                            record.mean,
                            expected_size=len(output_names),
                            field_name='mean',
                            group_id=group.group_id,
                            record_id=record_id,
                        )
                        variance = self._validated_result_vector(
                            record.variance,
                            expected_size=len(output_names),
                            field_name='variance',
                            group_id=group.group_id,
                            record_id=record_id,
                            non_negative=True,
                        )
                        mse = self._validated_result_vector(
                            record.mse,
                            expected_size=len(output_names),
                            field_name='mse',
                            group_id=group.group_id,
                            record_id=record_id,
                            non_negative=True,
                        )
                        actual = np.full(len(output_names), np.nan)
                        squared_bias = np.full(len(output_names), np.nan)
                        test_set_position = None
                        model_id = record_id

                    for output_index, output_name in enumerate(output_names):
                        prepared_rows.append(
                            {
                                'run_id': run_id,
                                'study_id': group.study_id,
                                'study_name': group.study_name,
                                'evaluation_method': method.value,
                                'group_id': group.group_id,
                                'group_name': group.group_name,
                                'result_type': result_type,
                                'record_index': record_index,
                                'test_set_position': test_set_position,
                                'model_id': model_id,
                                'output_index': output_index,
                                'output_name': str(output_name),
                                'actual_value': float(actual[output_index]),
                                'mean_prediction': float(mean[output_index]),
                                'prediction_variance': float(
                                    variance[output_index]
                                ),
                                'prediction_std': float(
                                    np.sqrt(variance[output_index])
                                ),
                                'squared_bias': float(
                                    squared_bias[output_index]
                                ),
                                'mse': float(mse[output_index]),
                            }
                        )

        return pd.DataFrame.from_records(
            prepared_rows,
            columns=self._PLOT_DATA_COLUMNS,
        )

    @classmethod
    def _select_plot_output(
        cls,
        results: pd.DataFrame,
        output: OutputSelector,
    ) -> pd.DataFrame:
        if not isinstance(results, pd.DataFrame):
            raise TypeError('results must be a pandas DataFrame.')
        missing = set(cls._PLOT_DATA_COLUMNS) - set(results.columns)
        if missing:
            raise ValueError(
                f'Results are missing required columns: {sorted(missing)}.'
            )
        if results.empty:
            raise ValueError('Results must not be empty.')

        outputs = results[['output_index', 'output_name']].drop_duplicates()
        names_per_index = outputs.groupby('output_index')['output_name'].nunique()
        if (names_per_index != 1).any():
            raise ValueError(
                'Each output_index must identify exactly one output_name.'
            )
        if isinstance(output, int) and not isinstance(output, bool):
            matching = outputs.loc[outputs['output_index'] == output]
            if matching.empty:
                valid = sorted(outputs['output_index'].astype(int).tolist())
                raise IndexError(
                    f'Unknown output index {output}; valid indices are {valid}.'
                )
        elif isinstance(output, str):
            matching = outputs.loc[outputs['output_name'] == output]
            if matching.empty:
                valid = sorted(outputs['output_name'].astype(str).tolist())
                raise KeyError(
                    f'Unknown output name {output!r}; valid names are {valid}.'
                )
            if len(matching) > 1:
                raise ValueError(
                    f'Output name {output!r} is ambiguous; select by index.'
                )
        else:
            raise TypeError('output must be an integer or string.')

        output_index = int(matching.iloc[0]['output_index'])
        return results.loc[results['output_index'] == output_index].copy()

    @staticmethod
    def _merged_plot_settings(
        settings: Mapping[str, object],
        overrides: Mapping[str, object],
    ) -> dict[str, object]:
        merged = dict(settings)
        for section in ('prediction', 'metrics'):
            base_section = merged.get(section, {})
            override_section = overrides.get(section, {})
            if not isinstance(base_section, Mapping) or not isinstance(
                override_section, Mapping
            ):
                raise TypeError(f'{section} settings must be mappings.')
            merged[section] = {**base_section, **override_section}
        merged.update(
            {
                key: value
                for key, value in overrides.items()
                if key not in {'prediction', 'metrics'}
            }
        )
        return merged

    def plot_bias_and_variance(
        self,
        results: pd.DataFrame,
        *,
        output: OutputSelector,
        plot_kind: PlotKind = 'components',
        plot_settings: Mapping[str, object] | None = None,
        group_settings: Mapping[int, Mapping[str, object]] | None = None,
        max_plots: int | None = None,
    ) -> tuple[GroupPlotResult, ...]:
        """Plot one explicitly selected output for every result group.

        ``components`` preserves test-point or model order in two aligned
        panels: prediction means with standard-deviation bars, followed by
        squared bias/MSE and variance. ``error_relationship`` instead places
        squared bias or MSE on the x-axis in one diagnostic panel. Pointwise
        standard deviations describe variation across models at one point;
        averaging standard deviations describe one model's predictions across
        its test observations.

        The supplied DataFrame is not cached or mutated. One independent
        figure is returned per group, and Matplotlib display is left to the
        caller.
        """
        if plot_kind not in {'components', 'error_relationship'}:
            raise ValueError(
                "plot_kind must be 'components' or 'error_relationship'."
            )
        if plot_settings is not None and not isinstance(plot_settings, Mapping):
            raise TypeError('plot_settings must be a mapping or None.')
        if group_settings is not None and not isinstance(group_settings, Mapping):
            raise TypeError('group_settings must be a mapping or None.')
        if max_plots is not None:
            if not isinstance(max_plots, int) or isinstance(max_plots, bool):
                raise TypeError('max_plots must be an integer or None.')
            if max_plots <= 0:
                raise ValueError('max_plots must be positive.')

        selected = self._select_plot_output(results, output)
        run_ids = selected['run_id'].drop_duplicates()
        if len(run_ids) != 1:
            raise ValueError('Results must contain exactly one run_id.')

        settings = dict(plot_settings or {})
        overrides_by_group = group_settings or {}
        plots: list[GroupPlotResult] = []
        grouped = selected.groupby(
            ['study_id', 'group_id'],
            sort=False,
            dropna=False,
        )
        for (_, group_id), group_data in grouped:
            if max_plots is not None and len(plots) >= max_plots:
                break
            group_id = int(group_id)
            overrides = overrides_by_group.get(group_id, {})
            if not isinstance(overrides, Mapping):
                raise TypeError(
                    f'group_settings[{group_id}] must be a mapping.'
                )
            resolved = self._merged_plot_settings(settings, overrides)
            group_data = group_data.copy()
            group_data['record_index'] = pd.to_numeric(
                group_data['record_index'], errors='raise'
            )
            if (
                not np.isfinite(group_data['record_index']).all()
                or (group_data['record_index'] < 0).any()
                or (group_data['record_index'] % 1 != 0).any()
            ):
                raise ValueError(
                    f'Group {group_id} has invalid record_index values.'
                )
            group_data = group_data.sort_values('record_index')

            metadata_columns = (
                'study_id',
                'group_name',
                'evaluation_method',
                'output_index',
                'output_name',
            )
            for column in metadata_columns:
                if group_data[column].nunique(dropna=False) != 1:
                    raise ValueError(
                        f'Group {group_id} has inconsistent {column} values.'
                    )
            if group_data['record_index'].duplicated().any():
                raise ValueError(
                    f'Group {group_id} has duplicate record_index values.'
                )

            try:
                method = EvaluationMethod(
                    str(group_data.iloc[0]['evaluation_method'])
                )
            except ValueError as exc:
                raise ValueError(
                    'Unsupported evaluation method for group '
                    f'{group_id}: '
                    f'{group_data.iloc[0]["evaluation_method"]!r}.'
                ) from exc

            numeric_columns = (
                'mean_prediction',
                'prediction_variance',
                'prediction_std',
            )
            for column in numeric_columns:
                group_data[column] = pd.to_numeric(
                    group_data[column], errors='raise'
                )
            if not np.isfinite(group_data[list(numeric_columns)]).all().all():
                raise ValueError(
                    f'Group {group_id} contains non-finite plot values.'
                )
            if (
                (group_data['prediction_variance'] < 0).any()
                or (group_data['prediction_std'] < 0).any()
            ):
                raise ValueError(
                    f'Group {group_id} contains negative prediction spread.'
                )

            if method is EvaluationMethod.POINTWISE:
                x_positions = pd.to_numeric(
                    group_data['test_set_position'], errors='raise'
                ).to_numpy(dtype=float)
                actual_values = pd.to_numeric(
                    group_data['actual_value'], errors='raise'
                ).to_numpy(dtype=float)
                primary_column = 'squared_bias'
                primary_label = 'Squared bias'
                variance_label = 'Pointwise model variance'
                x_label = 'Test-set position'
            else:
                x_positions = (
                    group_data['record_index'].to_numpy(dtype=float) + 1
                )
                actual_values = None
                primary_column = 'mse'
                primary_label = 'Model MSE'
                variance_label = 'Within-model prediction variance'
                x_label = 'Model number'

            primary_values = pd.to_numeric(
                group_data[primary_column], errors='raise'
            ).to_numpy(dtype=float)
            if (
                not np.isfinite(primary_values).all()
                or (primary_values < 0).any()
            ):
                raise ValueError(
                    f'Group {group_id} contains invalid {primary_column} values.'
                )

            method_name = method.value.upper()
            group_name = str(group_data.iloc[0]['group_name'])
            output_index = int(group_data.iloc[0]['output_index'])
            output_name = str(group_data.iloc[0]['output_name'])
            title = str(
                resolved.get('title', f'{method_name}: {group_name}')
            )
            output_label = f'{output_name} [{output_index}]'
            prediction_settings = dict(resolved['prediction'])
            prediction_settings.setdefault('ylabel', output_label)

            if plot_kind == 'components':
                figure, axes = plt.subplots(
                    2,
                    1,
                    figsize=resolved.get('figsize', (10, 8)),
                    sharex=True,
                    gridspec_kw={
                        'height_ratios': resolved.get(
                            'height_ratios', (2, 1)
                        )
                    },
                )
                prediction_axes, metric_axes = axes
                prediction_settings.setdefault('title', 'Prediction summary')
                prediction_settings.setdefault('xlabel', '')
                plot_prediction_distribution(
                    x_positions,
                    group_data['mean_prediction'],
                    group_data['prediction_std'],
                    prediction_settings,
                    actual_values=actual_values,
                    ax=prediction_axes,
                )

                metric_settings = dict(resolved['metrics'])
                metric_settings.setdefault('title', 'Error components')
                metric_settings.setdefault('xlabel', x_label)
                metric_settings.setdefault(
                    'ylabel', f'Squared {output_label}'
                )
                metric_settings.setdefault('primary_label', primary_label)
                metric_settings.setdefault('variance_label', variance_label)
                plot_error_components(
                    x_positions,
                    primary_values,
                    group_data['prediction_variance'],
                    metric_settings,
                    ax=metric_axes,
                )
            else:
                figure, prediction_axes = plt.subplots(
                    figsize=resolved.get('figsize', (10, 6))
                )
                metric_axes = None
                prediction_settings.setdefault('title', '')
                prediction_settings.setdefault(
                    'xlabel',
                    'Squared bias' if method is EvaluationMethod.POINTWISE
                    else 'Model MSE',
                )
                plot_prediction_distribution(
                    primary_values,
                    group_data['mean_prediction'],
                    group_data['prediction_std'],
                    prediction_settings,
                    actual_values=actual_values,
                    ax=prediction_axes,
                )

            figure.suptitle(title)
            figure.tight_layout(rect=(0, 0, 1, 0.96))
            plots.append(
                GroupPlotResult(
                    run_id=str(run_ids.iloc[0]),
                    study_id=int(group_data.iloc[0]['study_id']),
                    group_id=group_id,
                    group_name=group_name,
                    evaluation_method=method.value,
                    output_index=output_index,
                    output_name=output_name,
                    figure=figure,
                    prediction_axes=prediction_axes,
                    metric_axes=metric_axes,
                )
            )

        if not plots:
            raise ValueError('Results contain no groups for the selected output.')
        return tuple(plots)

    def plot_summary(
        self,
        results: pd.DataFrame,
        *,
        output: OutputSelector | None = None,
        plot_settings: Mapping[str, Mapping[str, object]] | None = None,
    ) -> dict[EvaluationMethod, Axes]:
        """Plot equal-weighted study summaries from tidy prepared results.

        With ``output=None``, every group/output combination receives equal
        weight. Selecting an output by index or name instead aggregates that
        output across the groups in each study. Pointwise and averaging
        results use separate axes because MSE is a total-error proxy rather
        than a direct squared-bias estimate.
        """
        if plot_settings is not None and not isinstance(plot_settings, Mapping):
            raise TypeError('plot_settings must be a mapping or None.')

        if output is None:
            if not isinstance(results, pd.DataFrame):
                raise TypeError('results must be a pandas DataFrame.')
            missing = set(self._PLOT_DATA_COLUMNS) - set(results.columns)
            if missing:
                raise ValueError(
                    f'Results are missing required columns: {sorted(missing)}.'
                )
            if results.empty:
                raise ValueError('Results must not be empty.')
            outputs = results[
                ['output_index', 'output_name']
            ].drop_duplicates()
            names_per_index = outputs.groupby(
                'output_index'
            )['output_name'].nunique()
            if (names_per_index != 1).any():
                raise ValueError(
                    'Each output_index must identify exactly one output_name.'
                )
            selected = results.copy()
            output_description = 'All outputs'
        else:
            selected = self._select_plot_output(results, output)
            output_index = int(selected.iloc[0]['output_index'])
            output_name = str(selected.iloc[0]['output_name'])
            output_description = f'{output_name} [{output_index}]'

        run_ids = selected['run_id'].drop_duplicates()
        if len(run_ids) != 1:
            raise ValueError('Results must contain exactly one run_id.')

        unknown_methods = set(selected['evaluation_method']) - {
            method.value for method in EvaluationMethod
        }
        if unknown_methods:
            raise ValueError(
                'Results contain unsupported evaluation methods: '
                f'{sorted(unknown_methods)}.'
            )

        resolved_settings = plot_settings or {}
        axes: dict[EvaluationMethod, Axes] = {}
        for method in EvaluationMethod:
            method_rows = selected.loc[
                selected['evaluation_method'] == method.value
            ].copy()
            if method_rows.empty:
                continue

            primary_column = (
                'squared_bias'
                if method is EvaluationMethod.POINTWISE
                else 'mse'
            )
            for column in (primary_column, 'prediction_variance'):
                method_rows[column] = pd.to_numeric(
                    method_rows[column], errors='raise'
                )
            metric_values = method_rows[
                [primary_column, 'prediction_variance']
            ]
            if (
                not np.isfinite(metric_values).all().all()
                or (metric_values < 0).any().any()
            ):
                raise ValueError(
                    f'{method.value} summary values must be finite and '
                    'non-negative.'
                )

            expected_outputs = method_rows['output_index'].nunique()
            outputs_per_group = method_rows.groupby(
                ['study_id', 'group_id'], sort=False
            )['output_index'].nunique()
            if (outputs_per_group != expected_outputs).any():
                raise ValueError(
                    f'{method.value} groups contain inconsistent outputs.'
                )

            group_output_means = (
                method_rows.groupby(
                    [
                        'study_id',
                        'study_name',
                        'group_id',
                        'output_index',
                    ],
                    sort=False,
                )[[primary_column, 'prediction_variance']]
                .mean()
                .reset_index()
            )
            study_means = (
                group_output_means.groupby(
                    ['study_id', 'study_name'], sort=False
                )[[primary_column, 'prediction_variance']]
                .mean()
            )

            method_settings = resolved_settings.get(method.value, {})
            if not isinstance(method_settings, Mapping):
                raise TypeError(
                    f'plot_settings[{method.value!r}] must be a mapping.'
                )
            method_settings = dict(method_settings)
            method_settings.setdefault(
                'title',
                f'{method.value.upper()} Summary — {output_description}',
            )
            labels = tuple(
                str(study_name).title()
                for _, study_name in study_means.index
            )
            axes[method] = plot_summary_bars(
                labels,
                study_means[primary_column],
                study_means['prediction_variance'],
                method_settings,
                primary_label=(
                    'Mean squared bias'
                    if method is EvaluationMethod.POINTWISE
                    else 'Mean model MSE (total-error proxy)'
                ),
                variance_label=(
                    'Mean pointwise model variance'
                    if method is EvaluationMethod.POINTWISE
                    else 'Mean within-model prediction variance'
                ),
            )

        if not axes:
            raise ValueError('Results contain no supported evaluation methods.')
        return axes
