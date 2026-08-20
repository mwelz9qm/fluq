from collections.abc import Mapping
from dataclasses import asdict, fields
from datetime import UTC, datetime
from os import PathLike
from pathlib import Path
from typing import Self
from uuid import uuid4
from warnings import warn

import numpy as np
import pandas as pd
import torch
from matplotlib.axes import Axes
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
    plot_bias_variance,
    plot_prediction_comparison,
)
from bias_variance.plotting import (
    plot_summary as plot_summary_bars,
)


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

    def get_bias_variance_summary(
        self,
        run_id: str | None = None,
    ) -> pd.DataFrame:
        """Return group-level bias/variance results in normalized long form."""
        history = self.get_run_history()
        if run_id is None:
            if history.empty:
                raise ValueError(
                    'No runs performed. Call run_studies() to get run.'
                )
            run_id = str(history.iloc[0]['run_id'])

        run_rows = history.loc[history['run_id'] == run_id]
        if run_rows.empty:
            raise ValueError(f'Run does not exist: {run_id}.')

        output_names = tuple(run_rows.iloc[0]['output_columns'])
        results = self.decompose_bias_and_variance(run_id)
        rows: list[dict[str, object]] = []

        for result in results.itertuples(index=False):
            try:
                method = EvaluationMethod(result.evaluation_method)
                StudyBias(result.study_name)
            except ValueError as exc:
                raise ValueError(
                    'Stored summary contains an unknown study or evaluation '
                    'method.'
                ) from exc

            bias = np.asarray(result.bias, dtype=float)
            variance = np.asarray(result.variance, dtype=float)
            if (
                bias.ndim != 1
                or variance.shape != bias.shape
                or len(bias) != len(output_names)
                or not np.isfinite(bias).all()
                or not np.isfinite(variance).all()
                or (bias < 0).any()
                or (variance < 0).any()
            ):
                raise ValueError(
                    'Stored bias and variance must be finite, non-negative '
                    'vectors matching the run outputs.'
                )

            primary_metric = (
                'squared_bias'
                if method is EvaluationMethod.POINTWISE
                else 'mse'
            )
            for output_index, output_name in enumerate(output_names):
                common = {
                    'run_id': run_id,
                    'study_name': result.study_name,
                    'evaluation_method': method.value,
                    'group_name': result.group_name,
                    'output_index': output_index,
                    'output_name': output_name,
                }
                rows.extend(
                    (
                        {
                            **common,
                            'metric_name': primary_metric,
                            'metric_value': float(bias[output_index]),
                        },
                        {
                            **common,
                            'metric_name': 'variance',
                            'metric_value': float(variance[output_index]),
                        },
                    )
                )

        return pd.DataFrame.from_records(rows)

    @staticmethod
    def _select_summary_output(
        summary: pd.DataFrame,
        output: int | str | None,
    ) -> pd.DataFrame:
        required = {
            'study_name',
            'evaluation_method',
            'group_name',
            'output_index',
            'output_name',
            'metric_name',
            'metric_value',
        }
        missing = required - set(summary.columns)
        if missing:
            raise ValueError(
                f'Summary is missing required columns: {sorted(missing)}.'
            )
        if summary.empty:
            raise ValueError('Summary must not be empty.')

        outputs = summary[['output_index', 'output_name']].drop_duplicates()
        if output is None:
            if len(outputs) != 1:
                raise ValueError(
                    'This summary contains multiple outputs; select one with '
                    'output=<index> or output=<name>.'
                )
            selected = summary['output_index'] == outputs.iloc[0]['output_index']
        elif isinstance(output, int) and not isinstance(output, bool):
            selected = summary['output_index'] == output
            if not selected.any():
                valid = sorted(outputs['output_index'].astype(int).tolist())
                raise ValueError(
                    f'Unknown output index {output}; valid indices are {valid}.'
                )
        elif isinstance(output, str):
            matching = outputs.loc[outputs['output_name'] == output]
            if matching.empty:
                valid = sorted(outputs['output_name'].astype(str).tolist())
                raise ValueError(
                    f'Unknown output name {output!r}; valid names are {valid}.'
                )
            if len(matching) > 1:
                raise ValueError(
                    f'Output name {output!r} is ambiguous; select by index.'
                )
            selected = summary['output_index'] == matching.iloc[0]['output_index']
        else:
            raise TypeError('output must be an integer, string, or None.')

        return summary.loc[selected].copy()

    def plot_summary(
        self,
        summary: pd.DataFrame,
        *,
        output: int | str | None = None,
        settings: Mapping[str, Mapping[str, object]] | None = None,
    ) -> dict[EvaluationMethod, Axes]:
        """Plot equal-weighted group means for one selected output."""
        if settings is not None and not isinstance(settings, Mapping):
            raise TypeError('settings must be a mapping or None.')

        selected = self._select_summary_output(summary, output)
        unknown_methods = set(selected['evaluation_method']) - {
            method.value for method in EvaluationMethod
        }
        if unknown_methods:
            raise ValueError(
                f'Summary contains unknown evaluation methods: '
                f'{sorted(unknown_methods)}.'
            )
        selected['metric_value'] = pd.to_numeric(
            selected['metric_value'], errors='raise'
        )
        if (
            not np.isfinite(selected['metric_value']).all()
            or (selected['metric_value'] < 0).any()
        ):
            raise ValueError('metric_value must be finite and non-negative.')

        resolved_settings = settings or {}
        axes: dict[EvaluationMethod, Axes] = {}
        output_name = str(selected.iloc[0]['output_name'])
        for method in EvaluationMethod:
            method_rows = selected.loc[
                selected['evaluation_method'] == method.value
            ]
            if method_rows.empty:
                continue

            primary_metric = (
                'squared_bias'
                if method is EvaluationMethod.POINTWISE
                else 'mse'
            )
            unknown_metrics = set(method_rows['metric_name']) - {
                primary_metric,
                'variance',
            }
            if unknown_metrics:
                raise ValueError(
                    f'{method.value} summary contains unknown metrics: '
                    f'{sorted(unknown_metrics)}.'
                )
            grouped = (
                method_rows.groupby(
                    ['study_name', 'metric_name'], sort=False
                )['metric_value']
                .mean()
                .unstack()
            )
            missing_metrics = {primary_metric, 'variance'} - set(grouped.columns)
            if missing_metrics:
                raise ValueError(
                    f'{method.value} summary is missing metrics: '
                    f'{sorted(missing_metrics)}.'
                )

            labels = tuple(
                f'{StudyBias(study).value.title()} — {method.name}'
                for study in grouped.index
            )
            method_settings = dict(resolved_settings.get(method.value, {}))
            method_settings.setdefault(
                'title', f'{method.name} Summary — {output_name}'
            )
            axes[method] = plot_summary_bars(
                labels,
                grouped[primary_metric],
                grouped['variance'],
                primary_label=(
                    'Squared Bias'
                    if method is EvaluationMethod.POINTWISE
                    else 'MSE'
                ),
                settings=method_settings,
            )

        if not axes:
            raise ValueError('Summary contains no supported evaluation methods.')
        return axes

    @staticmethod
    def _select_plot_value(value, index: int, *, name: str) -> float:
        """Select one output or input value from a scalar or vector cell."""
        try:
            values = np.asarray(value, dtype=float)
        except (TypeError, ValueError) as exc:
            raise TypeError(f'{name} must contain numeric values.') from exc

        if values.ndim == 0:
            if index != 0:
                raise IndexError(f'{name} has no value at index {index}.')
            selected = float(values)
        else:
            flattened = values.reshape(-1)
            if index >= len(flattened):
                raise IndexError(f'{name} has no value at index {index}.')
            selected = float(flattened[index])

        if not np.isfinite(selected):
            raise ValueError(f'{name} must contain only finite values.')
        return selected

    @classmethod
    def _prepare_group_plot_data(
        col,
        rows,
        *,
        input_index: int,
        output_index: int,
    ) -> pd.DataFrame:
        """Normalize result-store rows into scalar plotting columns.

        The result-store query is expected to return either a DataFrame, a
        column mapping, or records containing 'x', 'actual',
        'prediction_mean', 'bias', and 'variance'.  For convenience,
        pointwise rows may use 'input'/'actual_output' and averaging rows
        may use 'r2'/'sample_mean' instead of 'x'/'actual'.  A
        'test_points' or 'model_ids' cell may alternatively contain a
        mapping with 'x' and 'actual' keys or an '(x, actual)' pair.
        """
        if isinstance(rows, pd.DataFrame):
            frame = rows.copy()
        elif isinstance(rows, Mapping):
            try:
                frame = pd.DataFrame(rows)
            except ValueError:
                frame = pd.DataFrame([rows])
        else:
            frame = pd.DataFrame.from_records(rows)

        if frame.empty:
            return frame

        if 'x' not in frame:
            if 'r2' in frame:
                frame['x'] = frame['r2']
            elif 'input' in frame:
                frame['x'] = frame['input']
        if 'actual' not in frame:
            if 'sample_mean' in frame:
                frame['actual'] = frame['sample_mean']
            elif 'actual_output' in frame:
                frame['actual'] = frame['actual_output']

        if 'x' not in frame or 'actual' not in frame:
            paired_column = next(
                (
                    column
                    for column in ('test_points', 'model_ids')
                    if column in frame
                ),
                None,
            )
            if paired_column is not None:
                paired_values: list[tuple[object, object]] = []
                for value in frame[paired_column]:
                    if isinstance(value, Mapping):
                        try:
                            pair = (value['x'], value['actual'])
                        except KeyError as exc:
                            raise ValueError(
                                f'{paired_column} mappings must contain x '
                                'and actual values.'
                            ) from exc
                    else:
                        try:
                            pair = tuple(value)
                        except TypeError as exc:
                            raise ValueError(
                                f'{paired_column} values must be (x, actual) '
                                'pairs, not identifiers alone.'
                            ) from exc
                        if len(pair) != 2:
                            raise ValueError(
                                f'{paired_column} values must be (x, actual) '
                                'pairs.'
                            )
                    paired_values.append(pair)

                if 'x' not in frame:
                    frame['x'] = [pair[0] for pair in paired_values]
                if 'actual' not in frame:
                    frame['actual'] = [pair[1] for pair in paired_values]

        required = {'x', 'actual', 'prediction_mean', 'bias', 'variance'}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(
                'Group plot results are missing required columns: '
                f'{sorted(missing)}.'
            )

        normalized = pd.DataFrame(index=frame.index)
        normalized['x'] = [
            col._select_plot_value(value, input_index, name='x')
            for value in frame['x']
        ]
        for column in ('actual', 'prediction_mean', 'bias', 'variance'):
            normalized[column] = [
                col._select_plot_value(value, output_index, name=column)
                for value in frame[column]
            ]

        for optional_column in ('group_name', 'evaluation_method'):
            if optional_column in frame:
                normalized[optional_column] = frame[optional_column].to_numpy()

        if 'evaluation_method' not in normalized:
            if 'test_points' in frame:
                normalized['evaluation_method'] = (
                    EvaluationMethod.POINTWISE.value
                )
            elif 'model_ids' in frame:
                normalized['evaluation_method'] = (
                    EvaluationMethod.AVERAGING.value
                )

        if (normalized['bias'] < 0).any():
            raise ValueError('bias must contain non-negative squared errors.')
        if (normalized['variance'] < 0).any():
            raise ValueError('variance must contain non-negative values.')

        return normalized.reset_index(drop=True)

    @staticmethod
    def _has_extreme_plot_range(
        data: pd.DataFrame,
        max_axis_range: float | None,
    ) -> bool:
        if max_axis_range is None:
            return False

        errors = np.sqrt(data['variance'].to_numpy(dtype=float))
        x_values = data['x'].to_numpy(dtype=float)
        lower = np.minimum(
            data['actual'].to_numpy(dtype=float),
            data['prediction_mean'].to_numpy(dtype=float) - errors,
        )
        upper = np.maximum(
            data['actual'].to_numpy(dtype=float),
            data['prediction_mean'].to_numpy(dtype=float) + errors,
        )

        values = (
            x_values,
            lower,
            upper,
            data['bias'].to_numpy(dtype=float),
            data['variance'].to_numpy(dtype=float),
        )
        return any(
            np.ptp(axis_values) > max_axis_range
            or np.max(np.abs(axis_values)) > max_axis_range
            for axis_values in values
        )

    def plot_results(
        self,
        run_id: str | None = None,
        *,
        max_plots: int = 12,
        max_axis_range: float | None = 1000000.0,
        input_index: int = 0,
        output_index: int = 0,
        settings: Mapping[str, Mapping[str, object]] | None = None,
    ) -> tuple[Axes, ...]:
        """Plot prediction comparisons and bias/variance for a run's groups.

        Deprecated: use get_bias_variance_summary() and plot_summary() for the
        supported run-summary workflow.

        The forthcoming result table must be exposed through
        'ResultStore.get_group_plot_results(group_id)'. That query should
        resolve method-specific database values into plot-ready rows. For a
        pointwise row, 'x' is a test input and 'actual' is its true output.
        For an averaging row, 'x' is a model R2 score and 'actual' is that
        model's test-set sample mean. Every row also supplies
        'prediction_mean', squared 'bias', and prediction 'variance'.

        'max_plots' limits plotted groups, with two Axes returned per group.
        Groups whose selected values exceed 'max_axis_range' are skipped.
        """
        warn(
            'plot_results() is deprecated; use get_bias_variance_summary() '
            'and plot_summary().',
            DeprecationWarning,
            stacklevel=2,
        )
        if not isinstance(max_plots, int) or isinstance(max_plots, bool):
            raise TypeError('max_plots must be an integer.')
        if max_plots <= 0:
            raise ValueError('max_plots must be positive.')
        if max_axis_range is not None and max_axis_range <= 0:
            raise ValueError('max_axis_range must be positive or None.')
        for name, index in (
            ('input_index', input_index),
            ('output_index', output_index),
        ):
            if not isinstance(index, int) or isinstance(index, bool):
                raise TypeError(f'{name} must be an integer.')
            if index < 0:
                raise ValueError(f'{name} must be non-negative.')
        if settings is not None and not isinstance(settings, Mapping):
            raise TypeError('settings must be a mapping or None.')

        resolved_settings = settings or {}
        axes: list[Axes] = []
        plotted_groups = 0

        # Maintain result store lifecyle within method call
        with ResultStore(self.db_path, timeout=self.db_timeout) as store:
            store.create_tables()

            if run_id is None:
                run_id = store.get_recent_run()

            elif not store.does_run_exist(run_id):
                raise ValueError(f'Run does not exist: {run_id}.')

            if run_id is None:
                raise ValueError(
                    'No runs performed. Call run_studies() to get run.'
                )

            get_plot_results = getattr(
                store,
                'get_group_plot_results',
                None,
            )
            if get_plot_results is None:
                raise NotImplementedError(
                    'ResultStore.get_group_plot_results(group_id) must be '
                    'implemented before run plots can be loaded.'
                )

            # Iterate through the studies and groups, plotting each group's results
            for study_id in store.get_studies(run_id):
                for group_id in store.get_groups(study_id):
                    if plotted_groups >= max_plots:
                        break

                    # Prepare the group plot data and skip if empty or extreme
                    group_data = self._prepare_group_plot_data(
                        get_plot_results(group_id),
                        input_index=input_index,
                        output_index=output_index,
                    )
                    if group_data.empty or self._has_extreme_plot_range(
                        group_data,
                        max_axis_range,
                    ):
                        continue

                    # Resolve the group name and evaluation method for plot titles
                    group_name = (
                        str(group_data['group_name'].iloc[0])
                        if 'group_name' in group_data
                        else f'Group {group_id}'
                    )
                    evaluation_method = (
                        str(group_data['evaluation_method'].iloc[0])
                        if 'evaluation_method' in group_data
                        else store.get_method(group_id)
                    )

                    # Set up scatter plot settings and plot the prediction comparison
                    comparison_settings = dict(
                        resolved_settings.get('comparison', {})
                    )
                    comparison_settings.setdefault(
                        'title',
                        f'{group_name} ({evaluation_method})',
                    )
                    comparison_settings.setdefault(
                        'xlabel',
                        'R2 score'
                        if evaluation_method == EvaluationMethod.AVERAGING.value
                        else 'Test input',
                    )
                    comparison_ax = plot_prediction_comparison(
                        group_data['x'],
                        group_data['actual'],
                        group_data['prediction_mean'],
                        np.sqrt(group_data['variance']),
                        settings=comparison_settings,
                    )

                    # Set up bar plot settings and plot the bias/variance bar chart
                    bar_settings = dict(
                        resolved_settings.get('bias_variance', {})
                    )
                    bar_settings.setdefault(
                        'title',
                        f'{group_name}: Mean Bias and Variance',
                    )
                    bar_ax = plot_bias_variance(
                        (group_name,),
                        (float(group_data['bias'].mean()),),
                        (float(group_data['variance'].mean()),),
                        settings=bar_settings,
                    )
                    axes.extend((comparison_ax, bar_ax))
                    plotted_groups += 1

                if plotted_groups >= max_plots:
                    break

        return tuple(axes)
