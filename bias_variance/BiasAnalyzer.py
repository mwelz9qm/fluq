from __future__ import annotations

import json
import os
import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from bias_variance._constants import (
    ACTUALS_DATASET_NAME,
    ARCHITECTURE_FIELD_NAME,
    BASELINE_ARCHITECTURE_FIELD_NAME,
    BATCH_SIZE_FIELD_NAME,
    CONF_INTERVAL_LOWER_FIELD_NAME,
    CONF_INTERVAL_UPPER_FIELD_NAME,
    CREATED_AT_FIELD_NAME,
    DEVICE_FIELD_NAME,
    EPOCHS_FIELD_NAME,
    FIT_ITERATIONS_DIR_NAME,
    ITERATION_FIELD_NAME,
    LEARNING_RATE_FIELD_NAME,
    LOSS_FIELD_NAME,
    MEAN_FIELD_NAME,
    METRICS_FIELD_NAME,
    MODEL_NAME,
    MODEL_SEED_FIELD_NAME,
    N_ITER_FIELD_NAME,
    OPTIMIZER_FIELD_NAME,
    PREDICTIONS_DATASET_NAME,
    PREDICTIONS_LAYER_NAME,
    RANDOM_STATE_FIELD_NAME,
    RESULTS_FILENAME,
    RUN_ID_FIELD_NAME,
    RUN_METADATA_FILENAME,
    STUDY_FIELD_NAME,
    TEST_SIZE_FIELD_NAME,
    TIMESTAMP_FIELD_NAME,
    VARIABLE_FIELD_NAME,
    VARIANCE_FIELD_NAME,
    EvaluationMethod,
    MetricName,
    PlotType,
    SamplingStrategyName,
    StudyName,
)
from bias_variance._plotting import (
    plot_mean_distribution,
    plot_prediction_means_by_r2_scores,
    plot_variance_contribution,
    plot_variance_distribution,
)
from bias_variance.generators.base import Generator, Variation
from bias_variance.generators.fnn_architecture import FnnArchitectureGenerator
from bias_variance.generators.noise import (
    NoiseGenerator,
    NoiseVariation,
)
from bias_variance.generators.sampling import (
    SamplingGenerator,
    SamplingStrategy,
)
from bias_variance.models.fnn.FnnArchitecture import FnnArchitecture
from bias_variance.models.fnn.FnnBuilder import FnnBuilder
from bias_variance.models.TrainingConfig import TrainingConfig
from common.sampling._sampling import (
    generate_latin_hypercube_samples,
    get_quantile_stratified_random_samples,
    get_random_samples,
)

type EvaluationMetrics = dict[str, object]
type MethodEvaluations = dict[str, EvaluationMetrics]
type VariationEvaluations = dict[str, MethodEvaluations]
type StudyEvaluations = dict[str, VariationEvaluations]
type EvaluationStore = dict[str, StudyEvaluations]

class BiasAnalyzer:
    '''
    Analyzes each bias by comparing to the base model and dataset to
    the generated predictions' 95% confidence interval.

    Parameters
    ------------
    inputs_df: pandas.DataFrame
        The input dataframe

    outputs_df: pandas.DataFrame
        The output dataframe
    
    model_settings: dict = { 'hidden_layers': [32, 32], 'activation' : 'relu', 'optimizer' : 'adam', 'loss' : 'mse', 'metrics' : ['rmse','r2','mse','mae'], 'epochs' : 100, 'batch_size' : 10, 'verbose' : 0 }
        All configurable settings for the base model.
        'hidden_layers': list[int] = Number of hidden layers and neurons per layer.
        'activation': str = Activation function.
        'optimizer': str = Optimizer algorithm.
        'loss': str = Loss function.
        'metrics': list[str] = Metrics for results.
        'epochs': int = Number of passes.
        'batch_size': int = Number of training samples.
        'verbose': int = logging infomation display modes.
    
    _results_df: pandas.DataFrame | None = None
        Caches the results after a study has ran.

    _runs_metadata_df: pandas.DataFrame | None = None
        Caches one metadata row per completed study run.
    
    _run_id: str | None = None
        The associated run id. Updates after run_bias_studies() is called.

    Questions
    ------------
    - Should we select all plots by default or only select the best representation w/ an auto selection feature?
    '''

    METRIC_OPTIONS = frozenset(MetricName)
    RESULTS_FILENAME = RESULTS_FILENAME
    RUN_METADATA_FILENAME = RUN_METADATA_FILENAME
    FIT_ITERATIONS_DIR_NAME = FIT_ITERATIONS_DIR_NAME
    STUDY_FIELD_NAME = STUDY_FIELD_NAME
    VARIABLE_FIELD_NAME = VARIABLE_FIELD_NAME

    RESULT_COLUMNS = (
        RUN_ID_FIELD_NAME,
        ITERATION_FIELD_NAME,
        STUDY_FIELD_NAME,
        VARIABLE_FIELD_NAME,
        MODEL_SEED_FIELD_NAME,
        ARCHITECTURE_FIELD_NAME,
        LOSS_FIELD_NAME,
        MetricName.RMSE,
        MetricName.R2,
        MetricName.MSE,
        MetricName.MAE,
        VARIANCE_FIELD_NAME,
        MEAN_FIELD_NAME,
        CONF_INTERVAL_LOWER_FIELD_NAME,
        CONF_INTERVAL_UPPER_FIELD_NAME,
    )

    RUN_METADATA_COLUMNS = (
        RUN_ID_FIELD_NAME,
        CREATED_AT_FIELD_NAME,
        RANDOM_STATE_FIELD_NAME,
        TEST_SIZE_FIELD_NAME,
        N_ITER_FIELD_NAME,
        OPTIMIZER_FIELD_NAME,
        LEARNING_RATE_FIELD_NAME,
        LOSS_FIELD_NAME,
        METRICS_FIELD_NAME,
        EPOCHS_FIELD_NAME,
        BATCH_SIZE_FIELD_NAME,
        DEVICE_FIELD_NAME,
        BASELINE_ARCHITECTURE_FIELD_NAME,
    )

    @staticmethod
    def _validate_random_state(random_state: int | None) -> None:
        if (
            random_state is not None
            and (
                not isinstance(random_state, int)
                or isinstance(random_state, bool)
            )
        ):
            raise TypeError('random_state must be an integer or None.')

    @staticmethod
    def _validate_test_size(test_size: float) -> None:
        if (
            not isinstance(test_size, (int, float))
            or isinstance(test_size, bool)
        ):
            raise TypeError('test_size must be numeric.')
        if not 0 < test_size < 1:
            raise ValueError('test_size must be between 0 and 1.')

    @staticmethod
    def _validate_n_iter(n_iter: int) -> None:
        if not isinstance(n_iter, int) or isinstance(n_iter, bool):
            raise TypeError('n_iter must be an integer.')
        if n_iter <= 0:
            raise ValueError('n_iter must be greater than 0.')

    @staticmethod
    def _validate_dataframe(
        dataframe: pd.DataFrame,
        *,
        name: str,
        require_numeric: bool = True,
    ) -> None:
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError(f'{name} must be a pandas DataFrame.')
        if dataframe.empty:
            raise ValueError(f'{name} must not be empty.')
        if dataframe.columns.has_duplicates:
            raise ValueError(f'{name} must not contain duplicate columns.')
        if require_numeric:
            non_numeric = list(
                dataframe.select_dtypes(exclude=[np.number]).columns
            )
            if non_numeric:
                raise TypeError(
                    f'{name} must contain only numeric columns; '
                    f'found {non_numeric}.'
                )
            values = dataframe.to_numpy(dtype=float, copy=False)
            if not np.isfinite(values).all():
                raise ValueError(f'{name} must contain only finite values.')

    @staticmethod
    def _validate_evaluation_methods(
        methods: Sequence[str],
    ) -> tuple[EvaluationMethod, ...]:
        if isinstance(methods, (str, bytes)) or not isinstance(
            methods,
            Sequence,
        ):
            raise TypeError(
                "evaluation_methods must be a sequence of method names."
            )

        if not methods:
            raise ValueError(
                "At least one evaluation method must be configured."
            )

        normalized: list[EvaluationMethod] = []

        for method in methods:
            if not isinstance(method, str):
                raise TypeError(
                    "Every evaluation method must be a string."
                )

            try:
                normalized_method = EvaluationMethod(method)
            except ValueError:
                raise ValueError(
                    f"Unsupported evaluation method: {method!r}."
                ) from None

            if normalized_method in normalized:
                raise ValueError(
                    "evaluation_methods must not contain duplicates."
                )

            normalized.append(normalized_method)

        return tuple(normalized)

    @classmethod
    def _validate_split(
        cls,
        split: object,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if not isinstance(split, (tuple, list)) or len(split) != 4:
            raise TypeError(
                'split must contain X_train, X_test, y_train, and y_test.'
            )

        X_train, X_test, y_train, y_test = split
        for name, dataframe in zip(
            ('X_train', 'X_test', 'y_train', 'y_test'),
            (X_train, X_test, y_train, y_test),
        ):
            cls._validate_dataframe(dataframe, name=name)

        if len(X_train) != len(y_train):
            raise ValueError('X_train and y_train must have the same row count.')
        if len(X_test) != len(y_test):
            raise ValueError('X_test and y_test must have the same row count.')
        if not X_train.index.equals(y_train.index):
            raise ValueError('X_train and y_train indexes must match.')
        if not X_test.index.equals(y_test.index):
            raise ValueError('X_test and y_test indexes must match.')

        return X_train, X_test, y_train, y_test

    def __init__(
        self,
        inputs_df: pd.DataFrame,
        outputs_df: pd.DataFrame,
        *,
        fnn_builder: FnnBuilder,
        baseline_architecture: FnnArchitecture,
        training_config: TrainingConfig,
        _results_df: pd.DataFrame | None = None,
        _runs_metadata_df: pd.DataFrame | None = None,
        _run_id: str | None = None,
    ) -> None:
        self._validate_dataframe(inputs_df, name='inputs_df')
        self._validate_dataframe(outputs_df, name='outputs_df')
        self._evaluations: dict = {}
        if len(inputs_df) != len(outputs_df):
            raise ValueError(
                'inputs_df and outputs_df must have the same number of rows.'
            )
        if not inputs_df.index.equals(outputs_df.index):
            raise ValueError('inputs_df and outputs_df indexes must match.')
        overlapping_columns = set(inputs_df.columns) & set(outputs_df.columns)
        if overlapping_columns:
            raise ValueError(
                'inputs_df and outputs_df must have distinct column names; '
                f'overlap: {sorted(overlapping_columns)}.'
            )
        if not isinstance(fnn_builder, FnnBuilder):
            raise TypeError('fnn_builder must be an FnnBuilder.')
        if not isinstance(baseline_architecture, FnnArchitecture):
            raise TypeError(
                'baseline_architecture must be an FnnArchitecture.'
            )
        if not isinstance(training_config, TrainingConfig):
            raise TypeError('training_config must be a TrainingConfig.')
        if training_config.optimizer not in {'adam', 'sgd'}:
            raise ValueError(
                f'Unsupported optimizer: {training_config.optimizer!r}.'
            )
        if training_config.loss not in {'mse', 'mae'}:
            raise ValueError(f'Unsupported loss: {training_config.loss!r}.')
        if not training_config.metrics:
            raise ValueError('training_config.metrics must not be empty.')
        unknown_metrics = set(training_config.metrics) - set(MetricName)
        if unknown_metrics:
            raise ValueError(f'Unsupported metrics: {sorted(unknown_metrics)}')
        if len(set(training_config.metrics)) != len(training_config.metrics):
            raise ValueError('training_config.metrics must not contain duplicates.')
        try:
            training_config.resolved_device
        except (RuntimeError, TypeError) as error:
            raise ValueError(
                f'Unsupported device: {training_config.device!r}.'
            ) from error
        if _results_df is not None and not isinstance(_results_df, pd.DataFrame):
            raise TypeError('_results_df must be a pandas DataFrame or None.')
        if (
            _runs_metadata_df is not None
            and not isinstance(_runs_metadata_df, pd.DataFrame)
        ):
            raise TypeError(
                '_runs_metadata_df must be a pandas DataFrame or None.'
            )
        if _results_df is not None:
            missing_columns = set(self.RESULT_COLUMNS) - set(_results_df.columns)
            if missing_columns:
                raise ValueError(
                    '_results_df is missing columns: '
                    f'{sorted(missing_columns)}.'
                )
        if _runs_metadata_df is not None:
            missing_columns = set(self.RUN_METADATA_COLUMNS) - set(
                _runs_metadata_df.columns
            )
            if missing_columns:
                raise ValueError(
                    '_runs_metadata_df is missing columns: '
                    f'{sorted(missing_columns)}.'
                )
        if _run_id is not None and (
            not isinstance(_run_id, str) or not _run_id.strip()
        ):
            raise TypeError('_run_id must be a non-empty string or None.')
        if fnn_builder.config.input_size != inputs_df.shape[1]:
            raise ValueError(
                'FnnBuilder input_size must match the number of input columns.'
            )

        if fnn_builder.config.output_size != outputs_df.shape[1]:
            raise ValueError(
                'FnnBuilder output_size must match the number of output columns.'
            )
        
        self.inputs_df = inputs_df
        self.outputs_df = outputs_df
        self.fnn_builder = fnn_builder
        self.baseline_architecture = baseline_architecture
        self.training_config = training_config
        self._results_df = _results_df
        self._runs_metadata_df = _runs_metadata_df
        self._run_id = _run_id
        self._evaluations: EvaluationStore = {}

    def _build_model(
        self,
        architecture: FnnArchitecture,
    ) -> nn.Sequential:
        if not isinstance(architecture, FnnArchitecture):
            raise TypeError('architecture must be an FnnArchitecture.')
        model = self.fnn_builder.build(architecture)
        if not isinstance(model, nn.Sequential):
            raise TypeError('FnnBuilder.build() must return nn.Sequential.')
        return model.to(self.training_config.resolved_device)

    def _build_run_metadata(
        self,
        *,
        n_iter: int,
        random_state: int | None,
        test_size: float,
    ) -> dict[str, object]:
        self._validate_n_iter(n_iter)
        self._validate_random_state(random_state)
        self._validate_test_size(test_size)
        if self._run_id is None:
            raise ValueError('_run_id is None.')

        return {
            RUN_ID_FIELD_NAME: self._run_id,
            CREATED_AT_FIELD_NAME: datetime.now(
                timezone.utc
            ).isoformat(),
            RANDOM_STATE_FIELD_NAME: random_state,
            TEST_SIZE_FIELD_NAME: test_size,
            N_ITER_FIELD_NAME: n_iter,
            OPTIMIZER_FIELD_NAME: self.training_config.optimizer,
            LEARNING_RATE_FIELD_NAME: (
                self.training_config.learning_rate
            ),
            LOSS_FIELD_NAME: self.training_config.loss,
            METRICS_FIELD_NAME: json.dumps(
                list(self.training_config.metrics)
            ),
            EPOCHS_FIELD_NAME: self.training_config.epochs,
            BATCH_SIZE_FIELD_NAME: (
                self.training_config.batch_size
            ),
            DEVICE_FIELD_NAME: str(
                self.training_config.resolved_device
            ),
            BASELINE_ARCHITECTURE_FIELD_NAME: json.dumps(
                list(self.baseline_architecture.hidden_layers)
            ),
        }

    def _record_run_metadata(
        self,
        *,
        n_iter: int,
        random_state: int | None,
        test_size: float,
    ) -> None:
        self._validate_n_iter(n_iter)
        self._validate_random_state(random_state)
        self._validate_test_size(test_size)
        if self._runs_metadata_df is None:
            raise ValueError('_runs_metadata_df is not initialized.')
        if self._run_id in set(
            self._runs_metadata_df[RUN_ID_FIELD_NAME].dropna()
        ):
            raise ValueError(f'Duplicate run ID: {self._run_id}')

        metadata = self._build_run_metadata(
            n_iter=n_iter,
            random_state=random_state,
            test_size=test_size,
        )

        self._runs_metadata_df = pd.concat(
            [
                self._runs_metadata_df,
                pd.DataFrame([metadata])
            ],
            ignore_index=True
        )

    @staticmethod
    def _load_or_initialize_table(
        filename: str,
        columns: tuple[str, ...],
    ) -> pd.DataFrame:
        if not isinstance(filename, str) or not filename.strip():
            raise TypeError('filename must be a non-empty string.')
        if not isinstance(columns, tuple) or not columns:
            raise TypeError('columns must be a non-empty tuple.')
        if any(not isinstance(column, str) or not column for column in columns):
            raise TypeError('Every column name must be a non-empty string.')
        if len(set(columns)) != len(columns):
            raise ValueError('columns must not contain duplicates.')
        if os.path.exists(filename):
            try:
                table = pd.read_csv(filename)
            except (OSError, pd.errors.ParserError) as error:
                raise ValueError(
                    f'Unable to load table from {filename!r}.'
                ) from error
            if table.columns.has_duplicates:
                raise ValueError(
                    f'Table {filename!r} contains duplicate columns.'
                )
            return table.reindex(columns=columns)
        
        return pd.DataFrame(columns=columns)

    def get_runs_metadata(self) -> pd.DataFrame:
        '''Return a copy of the persisted and in-memory run metadata.'''
        if self._runs_metadata_df is None:
            self._runs_metadata_df = self._load_or_initialize_table(
                RUN_METADATA_FILENAME,
                self.RUN_METADATA_COLUMNS,
            )

        return self._runs_metadata_df.copy()

    def get_run_metadata(self, run_id: str) -> pd.Series:
        '''Return the metadata row for one run ID.'''
        if not isinstance(run_id, str) or not run_id.strip():
            raise TypeError('run_id must be a non-empty string.')
        runs_metadata = self.get_runs_metadata()
        matches = runs_metadata[
            runs_metadata[RUN_ID_FIELD_NAME] == run_id
        ]

        if matches.empty:
            raise KeyError(f'Unknown run ID: {run_id}')

        return matches.iloc[0].copy()

    def get_results_with_metadata(self) -> pd.DataFrame:
        '''Join result rows to their run metadata using ``run_id``.'''
        if self._results_df is None:
            self._results_df = self._load_or_initialize_table(
                RESULTS_FILENAME,
                self.RESULT_COLUMNS,
            )

        joined = self._results_df.merge(
            self.get_runs_metadata(),
            on=RUN_ID_FIELD_NAME,
            how='left',
            validate='many_to_one',
            suffixes=('_result', '_run'),
        )
        metadata_run_ids = set(
            self.get_runs_metadata()[RUN_ID_FIELD_NAME].dropna()
        )
        orphaned_run_ids = set(
            self._results_df[RUN_ID_FIELD_NAME].dropna()
        ) - metadata_run_ids
        if orphaned_run_ids:
            raise ValueError(
                'Results reference unknown run IDs: '
                f'{sorted(orphaned_run_ids)}.'
            )
        return joined

    def _save_predictions_and_actuals(
        self,
        predictions: np.ndarray,
        actuals: pd.DataFrame,
        *,
        study: str,
        label: str,
        iteration: int,
    ) -> None:
        '''
        Save one iteration's predictions and actual values to the run's HDF5 file.

        Parameters
        ----------
        predictions : numpy.ndarray
            Values predicted by the trained model.
        actuals : pandas.DataFrame
            Expected output values corresponding by row to ``predictions``.
        study : str
            Name of the study that produced the predictions.
        label : str
            Label of the generated variation within the study.
        iteration : int
            Zero-based study iteration number.

        Returns
        -------
        None
        '''
        if self._run_id is None:
            raise ValueError('_run_id is None.')
        if not isinstance(predictions, np.ndarray):
            raise TypeError('predictions must be a NumPy array.')
        self._validate_dataframe(actuals, name='actuals')
        if predictions.ndim != 2:
            raise ValueError('predictions must be a two-dimensional array.')
        if predictions.shape != actuals.shape:
            raise ValueError(
                'predictions and actuals must have matching shapes.'
            )
        if not np.issubdtype(predictions.dtype, np.number):
            raise TypeError('predictions must contain numeric values.')
        if not np.isfinite(predictions).all():
            raise ValueError('predictions must contain only finite values.')
        if not isinstance(study, str) or not study.strip() or '/' in study:
            raise ValueError('study must be a non-empty HDF5-safe label.')
        if not isinstance(label, str) or not label.strip() or '/' in label:
            raise ValueError('label must be a non-empty HDF5-safe label.')
        if not isinstance(iteration, int) or isinstance(iteration, bool):
            raise TypeError('iteration must be an integer.')
        if iteration < 0:
            raise ValueError('iteration must be non-negative.')
        pred_file_path = os.path.join(
            FIT_ITERATIONS_DIR_NAME,
            f'{self._run_id}.h5'
        )
        os.makedirs(FIT_ITERATIONS_DIR_NAME, exist_ok=True)
        group_path = f'{study}/{label}/iteration_{iteration}'
        with h5py.File(pred_file_path, 'a') as hf:
            group = hf.create_group(group_path)
            group.create_dataset(PREDICTIONS_DATASET_NAME, data=predictions)
            group.create_dataset(ACTUALS_DATASET_NAME, data=actuals)

    def _to_tensor(
        self,
        dataframe: pd.DataFrame,
    ) -> torch.Tensor:
        self._validate_dataframe(dataframe, name='dataframe')
        array = dataframe.to_numpy(
            dtype=np.float32,
            copy=True
        )

        return torch.from_numpy(array)
    
    def _build_loss(self) -> nn.Module:
        losses: dict[str, type[nn.Module]] = {
            'mse': nn.MSELoss,
            'mae': nn.L1Loss
        }

        try:
            loss_type = losses[self.training_config.loss]
        except KeyError:
            raise ValueError(
                f'Unsupported loss: {self.training_config.loss!r}.'
                f'Expected one of {sorted(losses)}.'
            ) from None
        
        return loss_type()
    
    def _build_optimizer(
        self,
        model: nn.Module,
    ) -> optim.Optimizer:
        if not isinstance(model, nn.Module):
            raise TypeError('model must be a torch.nn.Module.')
        if not any(True for _ in model.parameters()):
            raise ValueError('model must contain trainable parameters.')
        optimizers: dict[
            str,
            type[optim.Optimizer],
        ] = {
            'adam': optim.Adam,
            'sgd': optim.SGD
        }

        try:
            optimizer_type = optimizers[self.training_config.optimizer]
        except KeyError:
            raise ValueError(
                f'Unsupported optimizer: '
                f'{self.training_config.optimizer!r}.'
                f'Expected one of {sorted(optimizers)}.'
            ) from None
        
        return optimizer_type(
            model.parameters(),
            lr=self.training_config.learning_rate
        )
    
    def _set_random_state(
        self,
        random_state: int | None,
    ) -> None:
        self._validate_random_state(random_state)
        if random_state is None:
            return
        
        np.random.seed(random_state)
        torch.manual_seed(random_state)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_state)

    def _train_model(
        self,
        model: nn.Module,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        *,
        random_state: int | None,
    ) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError('model must be a torch.nn.Module.')
        if not isinstance(X_train, torch.Tensor):
            raise TypeError('X_train must be a torch.Tensor.')
        if not isinstance(y_train, torch.Tensor):
            raise TypeError('y_train must be a torch.Tensor.')
        self._validate_random_state(random_state)
        if X_train.ndim != 2 or y_train.ndim != 2:
            raise ValueError('Training tensors must be two-dimensional.')
        if X_train.shape[0] == 0:
            raise ValueError('Training tensors must not be empty.')
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError(
                'X_train and y_train must have the same sample count.'
            )
        if X_train.shape[1] != self.fnn_builder.config.input_size:
            raise ValueError('X_train has an unexpected feature count.')
        if y_train.shape[1] != self.fnn_builder.config.output_size:
            raise ValueError('y_train has an unexpected output count.')
        if not torch.is_floating_point(X_train) or not torch.is_floating_point(
            y_train
        ):
            raise TypeError('Training tensors must use a floating-point dtype.')
        if not torch.isfinite(X_train).all() or not torch.isfinite(y_train).all():
            raise ValueError('Training tensors must contain only finite values.')
        dataset = TensorDataset(X_train, y_train)

        loader_generator = torch.Generator()
        if random_state is not None:
            loader_generator.manual_seed(random_state)

        loader = DataLoader(
            dataset,
            batch_size=self.training_config.batch_size,
            shuffle=True,
            generator=loader_generator
        )

        criterion = self._build_loss()
        optimizer = self._build_optimizer(model)

        model.train()

        for _ in np.arange(self.training_config.epochs):
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(self.training_config.resolved_device)
                batch_y = batch_y.to(self.training_config.resolved_device)
                optimizer.zero_grad()
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
                loss.backward()
                optimizer.step()

    def _predict(
        self,
        model: nn.Module,
        X_test: torch.Tensor,
        y_test: torch.Tensor,
    ) -> tuple[np.ndarray, float]:
        if not isinstance(model, nn.Module):
            raise TypeError('model must be a torch.nn.Module.')
        if not isinstance(X_test, torch.Tensor):
            raise TypeError('X_test must be a torch.Tensor.')
        if not isinstance(y_test, torch.Tensor):
            raise TypeError('y_test must be a torch.Tensor.')
        if X_test.ndim != 2 or y_test.ndim != 2:
            raise ValueError('Test tensors must be two-dimensional.')
        if X_test.shape[0] == 0:
            raise ValueError('Test tensors must not be empty.')
        if X_test.shape[0] != y_test.shape[0]:
            raise ValueError('X_test and y_test must have the same sample count.')
        if X_test.shape[1] != self.fnn_builder.config.input_size:
            raise ValueError('X_test has an unexpected feature count.')
        if y_test.shape[1] != self.fnn_builder.config.output_size:
            raise ValueError('y_test has an unexpected output count.')
        if not torch.isfinite(X_test).all() or not torch.isfinite(y_test).all():
            raise ValueError('Test tensors must contain only finite values.')
        model.eval()

        with torch.inference_mode():
            predictions = model(
                X_test.to(self.training_config.resolved_device)
            )
            test_loss = self._build_loss()(
                predictions,
                y_test.to(self.training_config.resolved_device)
            )

        return predictions.cpu().numpy(), float(test_loss.item())
    
    def _calculate_scores(
        self,
        actuals: np.ndarray,
        predictions: np.ndarray,
    ) -> dict[str, float]:
        if not isinstance(actuals, np.ndarray):
            raise TypeError('actuals must be a NumPy array.')
        if not isinstance(predictions, np.ndarray):
            raise TypeError('predictions must be a NumPy array.')
        if actuals.ndim != 2 or predictions.ndim != 2:
            raise ValueError('actuals and predictions must be two-dimensional.')
        if actuals.shape != predictions.shape:
            raise ValueError('actuals and predictions must have matching shapes.')
        if actuals.shape[0] < 2:
            raise ValueError('At least two test samples are required for metrics.')
        if not np.issubdtype(actuals.dtype, np.number) or not np.issubdtype(
            predictions.dtype, np.number
        ):
            raise TypeError('actuals and predictions must be numeric.')
        if not np.isfinite(actuals).all() or not np.isfinite(predictions).all():
            raise ValueError(
                'actuals and predictions must contain only finite values.'
            )
        mse = mean_squared_error(actuals, predictions)

        scores = {
            MetricName.MSE: float(mse),
            MetricName.RMSE: float(np.sqrt(mse)),
            MetricName.MAE: float(
                mean_absolute_error(actuals, predictions)
            ),
            MetricName.R2: float(
                r2_score(
                    actuals,
                    predictions,
                    multioutput='uniform_average',
                )
            )
        }

        return {
            str(metric): value
            for metric, value in scores.items()
            if metric in self.training_config.metrics
        }

    def _evaluate_methods(
        self,
        predictions: np.ndarray,
        *,
        pointwise_actuals: np.ndarray,
        averaging_actuals: np.ndarray,
        methods: Sequence[EvaluationMethod],
    ) -> dict[str, dict[str, object]]:
        evaluations: dict[str, dict[str, object]] = {}

        for method in methods:
            if method == EvaluationMethod.AVERAGING:
                evaluations[str(method)] = self._evaluate_averaging(
                    predictions,
                    averaging_actuals,
                )
            elif method == EvaluationMethod.POINTWISE:
                evaluations[str(method)] = self._evaluate_pointwise(
                    predictions,
                    pointwise_actuals,
                )
            else:
                # Defensive check in case this method is called internally
                # without prior validation.
                raise ValueError(
                    f"Unsupported evaluation method: {method!r}."
                )

        return evaluations

    def _evaluate_pointwise(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
    ) -> dict[str, np.ndarray]:
        if predictions.ndim != 3:
            raise ValueError(
                "predictions must have shape "
                "(models, test_points, outputs)."
            )

        if predictions.shape[1:] != actuals.shape:
            raise ValueError(
                "Pointwise predictions and actual test values do not align."
            )

        prediction_mean = predictions.mean(axis=0)
        prediction_variance = predictions.var(axis=0)
        bias = prediction_mean - actuals

        return {
            "pointwise_mean": prediction_mean,
            "pointwise_variance": prediction_variance,
            "pointwise_bias": bias,
            "pointwise_squared_bias": bias**2,
        }

    def _evaluate_averaging(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
    ) -> dict[str, np.ndarray]:
        if predictions.ndim != 3:
            raise ValueError(
                "predictions must have shape "
                "(models, test_points, outputs)."
            )

        if predictions.shape[2] != actuals.shape[1]:
            raise ValueError(
                "Predictions and averaging test values have different output dimensions."
            )

        # Shape: (models, outputs)
        model_means = predictions.mean(axis=1)

        # Shape: (outputs,)
        prediction_mean = model_means.mean(axis=0)
        prediction_variance = model_means.var(axis=0)
        actual_mean = actuals.mean(axis=0)
        bias = prediction_mean - actual_mean

        return {
            "averaging_mean": prediction_mean,
            "averaging_variance": prediction_variance,
            "averaging_actual_mean": actual_mean,
            "averaging_bias": bias,
            "averaging_squared_bias": bias**2,
        }

    def _get_test_result_and_data(
        self,
        split: tuple[
            pd.DataFrame,
            pd.DataFrame,
            pd.DataFrame,
            pd.DataFrame,
        ] | None = None,
        architecture: FnnArchitecture | None = None,
        *,
        random_state: int | None,
        test_size: float,
    ) -> tuple[dict[str, float], np.ndarray, pd.DataFrame]:
        '''
        Train and evaluate a model for one generated study variation.

        Parameters
        ----------
        split : tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame, pandas.DataFrame] | None
            Optional ``X_train``, ``X_test``, ``y_train``, and ``y_test`` data.
            When omitted, the analyzer's base data is split using its configured
            ``test_size`` and ``random_state``.
        hidden_layers : list[int] | None
            Hidden-layer sizes for a generated model architecture. When omitted,
            the analyzer's base model is reused with its initial weights restored.

        Returns
        -------
        tuple[dict[str, float], numpy.ndarray, pandas.DataFrame]
            Evaluation metrics, model predictions, and the corresponding actual
            output values.
        '''
        self._validate_random_state(random_state)
        self._validate_test_size(test_size)
        if architecture is not None and not isinstance(
            architecture, FnnArchitecture
        ):
            raise TypeError('architecture must be an FnnArchitecture or None.')

        if split is None:
            generated_split = train_test_split(
                self.inputs_df,
                self.outputs_df,
                test_size=test_size,
                random_state=random_state
            )
            X_train, X_test, y_train, y_test = self._validate_split(
                generated_split
            )
        else:
            X_train, X_test, y_train, y_test = self._validate_split(split)

        if list(X_train.columns) != list(self.inputs_df.columns):
            raise ValueError('X_train columns must match inputs_df.')
        if list(X_test.columns) != list(self.inputs_df.columns):
            raise ValueError('X_test columns must match inputs_df.')
        if list(y_train.columns) != list(self.outputs_df.columns):
            raise ValueError('y_train columns must match outputs_df.')
        if list(y_test.columns) != list(self.outputs_df.columns):
            raise ValueError('y_test columns must match outputs_df.')

        selected_architecture  = (
            self.baseline_architecture
            if architecture is None
            else architecture
        )

        self._set_random_state(random_state)
        model = self._build_model(selected_architecture)

        X_train_tensor = self._to_tensor(X_train)
        X_test_tensor = self._to_tensor(X_test)
        y_train_tensor = self._to_tensor(y_train)
        y_test_tensor = self._to_tensor(y_test)

        # train
        self._train_model(
            model,
            X_train_tensor,
            y_train_tensor,
            random_state=random_state
        )

        predictions, test_loss = self._predict(model, X_test_tensor, y_test_tensor)
        actuals_values = y_test.to_numpy(dtype=np.float32)

        scores = {
            LOSS_FIELD_NAME: test_loss,
            **self._calculate_scores(
                actuals_values,
                predictions
            )
        }

        predictions_values = predictions.reshape(-1)
        within_model_variance = float(np.var(predictions_values))
        within_model_mean = float(np.mean(predictions_values))
        standard_error = float(stats.sem(predictions_values))
        conf_interval = (
            (within_model_mean, within_model_mean)
            if standard_error == 0.0
            else stats.norm.interval(
                0.95,
                loc=within_model_mean,
                scale=standard_error,
            )
        )

        scores.update(
            {
                VARIANCE_FIELD_NAME: within_model_variance,
                MEAN_FIELD_NAME: within_model_mean,
                CONF_INTERVAL_LOWER_FIELD_NAME: float(
                    conf_interval[0]
                ),
                CONF_INTERVAL_UPPER_FIELD_NAME: float(
                    conf_interval[1]
                )
            }
        )

        return scores, predictions, y_test

    def _get_results(
        self,
        n_iter: int,
        generator: Generator[FnnArchitecture] | Generator[pd.DataFrame],
        study: str,
        *,
        base_split: tuple[
            pd.DataFrame,
            pd.DataFrame,
            pd.DataFrame,
            pd.DataFrame,
        ],
        random_state: int | None,
        test_size: float,
        save_predictions: bool = True,
    ) -> tuple[
        pd.DataFrame,
        dict[str, list[np.ndarray]],
        ]:
        '''
        Generate, train, and evaluate all variations for a configured study.

        Parameters
        ----------
        n_iter : int
            Number of times to invoke the generator.
        generator : Generator[tuple[int, ...]] | Generator[pandas.DataFrame] | Generator[NoiseVariation]
            Architecture, sampled-dataset, or noise generator used by the study.
        study : str
            Study type that determines how generated variations are prepared.
            Currently supports ``'model'``, ``'sampling'``, and ``'data'``.
        save_predictions : bool, default=True
            Whether to persist predictions and actual values for each variation.

        Returns
        -------
        pandas.DataFrame
            One result row per generated label and iteration.
        '''
        self._validate_n_iter(n_iter)
        self._validate_random_state(random_state)
        self._validate_test_size(test_size)
        if not isinstance(generator, Generator):
            raise TypeError('generator must implement Generator.')
        try:
            normalized_study = StudyName(study)
        except (TypeError, ValueError):
            raise ValueError(f'Unsupported study: {study!r}') from None
        if normalized_study not in {StudyName.MODEL, StudyName.SAMPLING, StudyName.DATA}:
            raise ValueError(f'Unsupported study: {study!r}')
        if not isinstance(save_predictions, bool):
            raise TypeError('save_predictions must be a boolean.')
        if self._run_id is None:
            raise ValueError('_run_id is None.')

        X_train_base, X_test_fixed, y_train_base, y_test_fixed = (
            self._validate_split(base_split)
        )

        study = normalized_study
        results = pd.DataFrame(columns=self.RESULT_COLUMNS)
        prediction_groups: dict[str, list[np.ndarray]] = {}

        for i in np.arange(n_iter):
            iteration_random_state = (
                None
                if random_state is None
                else random_state + int(i)
            )

            variations = generator.generate(random_state=iteration_random_state)
            if not isinstance(variations, list[Variation]):
                raise TypeError('generator.generate() must return a mapping.')
            if not variations:
                raise ValueError(
                    f'Generator produced no variations for study {study!r}.'
                )

            for j, variation in enumerate(variations):
                if not isinstance(variation.label, str) or not variation.label.strip():
                    raise ValueError(
                        'Generated variation labels must be non-empty strings.'
                    )
                model_random_state = (
                    None
                    if iteration_random_state is None
                    else iteration_random_state + j
                )
                architecture = None
                split = None

                if study == StudyName.MODEL:
                    if not isinstance(variation.generated, FnnArchitecture):
                        raise TypeError(
                            'Model studies must generate FnnArchitecture values.'
                        )
                    split = base_split
                    architecture = variation.generated

                elif study == StudyName.SAMPLING:
                    if not isinstance(variation.generated, pd.DataFrame):
                        raise TypeError(
                            'Sampling studies must generate pandas DataFrames.'
                        )
                    self._validate_dataframe(
                        variation,
                        name=f'sampling variation {variation.label!r}',
                    )
                    required_columns = set(self.inputs_df.columns) | set(
                        self.outputs_df.columns
                    )
                    missing_columns = required_columns - set(variation.columns)
                    if missing_columns:
                        raise ValueError(
                            f'Sampling variation {variation.label!r} is missing columns: '
                            f'{sorted(missing_columns)}.'
                        )

                    sampled_train = variation.generated
                    sampling_X_train = sampled_train[self.inputs_df.columns]
                    sampling_y_train = sampled_train[self.outputs_df.columns]
                    split = (
                        sampling_X_train,
                        X_test_fixed,
                        sampling_y_train,
                        y_test_fixed,
                    )
                elif study == StudyName.DATA:
                    if not isinstance(variation.generated, NoiseVariation):
                        raise TypeError(
                            "Data studies must generate NoiseVariation values."
                        )

                    noisy_train = variation.dataset

                    self._validate_dataframe(
                        noisy_train,
                        name=f"data variation {variation.label!r}",
                    )

                    required_columns = set(self.inputs_df.columns) | set(
                        self.outputs_df.columns
                    )
                    missing_columns = required_columns - set(noisy_train.columns)

                    if missing_columns:
                        raise ValueError(
                            f"Data variation {label!r} is missing columns: "
                            f"{sorted(missing_columns)}."
                        )

                    data_X_train = noisy_train[self.inputs_df.columns]
                    data_y_train = noisy_train[self.outputs_df.columns]

                    split = (
                        data_X_train,
                        X_test_fixed,
                        data_y_train,
                        y_test_fixed,
                    )

                result, predictions, actuals = self._get_test_result_and_data(
                    architecture=architecture,
                    split=split,
                    random_state=model_random_state,
                    test_size=test_size,
                )
                prediction_groups.setdefault(label, []).append(predictions)

                if save_predictions:
                    self._save_predictions_and_actuals(
                        predictions,
                        actuals,
                        study=study,
                        label=label,
                        iteration=int(i)
                    )
                
                df_row = {
                    RUN_ID_FIELD_NAME: self._run_id,
                    ITERATION_FIELD_NAME: int(i),
                    STUDY_FIELD_NAME: study,
                    VARIABLE_FIELD_NAME: label,
                    MODEL_SEED_FIELD_NAME: model_random_state,
                    ARCHITECTURE_FIELD_NAME: json.dumps(
                        list(
                            architecture.hidden_layers
                            if architecture is not None
                            else self.baseline_architecture.hidden_layers
                        )
                    ),
                } | result

                results = pd.concat([results, pd.DataFrame([df_row])], ignore_index=True)

        return results, prediction_groups

    def _build_generator(
        self,
        study: str,
        settings: dict[str, object],
        *,
        training_dataset: pd.DataFrame | None = None,
    ) -> FnnArchitectureGenerator | SamplingGenerator | NoiseGenerator:
        '''
        Construct the concrete generator configured for a study.

        Parameters
        ----------
        study : str
            Study type. Supported values are ``'model'``, ``'sampling'``, and ``'data'``.
        settings : dict[str, object]
            Generator settings for the selected study. Model settings describe
            architecture families; sampling settings contain strategy names.

        Returns
        -------
        ArchitectureGenerator | SamplingGenerator | NoiseGenerator
            Generator initialized with the study settings and, for sampling,
            the analyzer's combined input and output dataset.

        Raises
        ------
        ValueError
            If ``study`` is unsupported.
        '''
        if not isinstance(settings, Mapping):
            raise TypeError('settings must be a mapping.')
        try:
            study = StudyName(study)
        except (TypeError, ValueError):
            raise ValueError(f'Unsupported study: {study!r}') from None

        if study == StudyName.MODEL:
            return FnnArchitectureGenerator(settings=settings)
        if study == StudyName.SAMPLING:
            sampling_strategies = []
            strategies = settings.get('strategies', [])

            if isinstance(strategies, (str, bytes)) or not isinstance(
                strategies,
                Sequence,
            ):
                raise TypeError("sampling strategies must be a sequence.")

            if any(not isinstance(strategy, str) for strategy in strategies):
                raise TypeError("Every sampling strategy must be a string.")

            supported_strategies = set(SamplingStrategyName)
            unknown_strategies = set(strategies) - supported_strategies

            if unknown_strategies:
                raise ValueError(
                    f"Unsupported sampling strategies: {sorted(unknown_strategies)}"
                )

            if len(strategies) != len(set(strategies)):
                raise ValueError(
                    "sampling strategies must not contain duplicates."
                )

            if training_dataset is None:
                raise ValueError(
                    'Training dataset is required for the sampling study.'
                )

            if SamplingStrategyName.BOOTSTRAP in strategies:
                sampling_strategies.append(
                    SamplingStrategy(
                        label=SamplingStrategyName.BOOTSTRAP,
                        function=get_random_samples,
                        kwargs={
                            'sample_fraction': 1.0,
                            'with_replacement': True
                        }
                    )
                )

            if SamplingStrategyName.STRATIFIED in strategies:
                sampling_strategies.append(
                    SamplingStrategy(
                        label=SamplingStrategyName.STRATIFIED,
                        function=get_quantile_stratified_random_samples,
                        kwargs={
                            'stratify_col_index': self.inputs_df.shape[1],
                            'sample_fraction': 1.0,
                            'with_replacement': True
                        }
                    )
                )
            
            if SamplingStrategyName.LHS in strategies:
                sampling_strategies.append(
                    SamplingStrategy(
                        label=SamplingStrategyName.LHS,
                        function=generate_latin_hypercube_samples,
                        kwargs={
                            'sample_fraction': 1.0,
                        }
                    )
                )

            return SamplingGenerator(dataset=training_dataset, strategies=sampling_strategies)

        if study == StudyName.DATA:
            standard_deviations = settings.get(
                "standard_deviations",
                (0.1, 0.2, 0.3, 0.4, 0.5),
            )

            if training_dataset is None:
                raise ValueError(
                    f'Training dataset is required for the data study.'
                )

            return NoiseGenerator(
                dataset=training_dataset,
                standard_deviations=standard_deviations,
            )
        raise ValueError(f'Unsupported study: {study!r}')
    
    def run_bias_studies(
        self,
        settings: dict[str, Any] | None = None,
        *,
        save_results: bool = True,
        save_predictions: bool = True
    ) -> 'BiasAnalyzer':
        '''
        Default Settings
        -------------
        >>> settings = {
        ...     'n_iter': 100,
        ...     'random_state': None,
        ...     'test_size': 0.2,
        ...     'evaluation_method': ('averaging', 'pointwise'),
        ...     'studies': {
        ...         'model': {
        ...             'wide': {
        ...                 'layers': (1, 16),
        ...                 'neurons': (64, 256),
        ...             },
        ...             'narrow': {
        ...                 'layers': (16, 64),
        ...                 'neurons': (2, 64),
        ...             },
        ...             'taper': {
        ...                 'layers': (16, 64),
        ...                 'init_neurons': (1, 9),
        ...                 'taper_rate': (0.25, 0.5),
        ...                 'max_neurons': 256,
        ...             },
        ...             'reverse_taper': {
        ...                 'layers': (16, 64),
        ...                 'init_neurons': (128, 256),
        ...                 'taper_rate': (0.25, 0.5),
        ...                 'max_neurons': 256,
        ...             },
        ...             'combined_taper': {
        ...                 'layers': (16, 64),
        ...                 'init_neurons': (1, 9),
        ...                 'taper_rate': (0.25, 0.5),
        ...                 'max_neurons': 256,
        ...             },
        ...         },
        ...         'sampling': {
        ...             'strategies': [
        ...                 'bootstrap', 'stratified', 'lhs'
        ...             ]
        ...         },
        ...         'data': {
        ...             'standard_deviations': (0.1, 0.2, 0.3, 0.4, 0.5)
        ...         },
        ...     }
        ... }
        '''
        default_settings = {
            'n_iter': 100,
            'random_state': None,
            'test_size': 0.2,
            'evaluation_method': [
                'averaging',
                'pointwise',
            ],
            'studies': {
                'model': {
                    'wide': {
                        'layers': (1, 16),
                        'neurons': (64, 256),
                    },
                    'narrow': {
                        'layers': (16, 64),
                        'neurons': (2, 64),
                    },
                    'taper': {
                        'layers': (16, 64),
                        'init_neurons': (1, 9),
                        'taper_rate': (0.25, 0.5),
                        'max_neurons': 256,
                    },
                    'reverse_taper': {
                        'layers': (16, 64),
                        'init_neurons': (128, 256),
                        'taper_rate': (0.25, 0.5),
                        'max_neurons': 256,
                    },
                    'combined_taper': {
                        'layers': (16, 64),
                        'init_neurons': (1, 9),
                        'taper_rate': (0.25, 0.5),
                        'max_neurons': 256,
                    },
                },
                'sampling': {
                    'strategies': [
                        'bootstrap', 'stratified', 'lhs'
                    ]
                },
                'data': {
                    'standard_deviations': (0.1, 0.2, 0.3, 0.4, 0.5)
                },
            }
        }
        if settings is None:
            settings = default_settings
        elif not isinstance(settings, Mapping):
            raise TypeError('settings must be a mapping or None.')
        if not isinstance(save_results, bool):
            raise TypeError('save_results must be a boolean.')
        if not isinstance(save_predictions, bool):
            raise TypeError('save_predictions must be a boolean.')

        allowed_setting_names = {
            'n_iter',
            'random_state',
            'test_size',
            'studies',
            'evaluation_method'
        }
        unknown_setting_names = set(settings) - allowed_setting_names
        if unknown_setting_names:
            raise ValueError(
                f'Unsupported run settings: {sorted(unknown_setting_names)}'
            )
        missing_setting_names = {'n_iter', 'studies'} - set(settings)
        if missing_setting_names:
            raise ValueError(
                f'Missing run settings: {sorted(missing_setting_names)}'
            )

        n_iter = settings['n_iter']
        random_state = settings.get('random_state')
        test_size = settings.get('test_size', 0.2)

        self._validate_n_iter(n_iter)
        self._validate_random_state(random_state)
        self._validate_test_size(test_size)

        base_split = train_test_split(
            self.inputs_df,
            self.outputs_df,
            test_size=test_size,
            random_state=random_state
        )
        X_train_base, X_test_fixed, y_train_base, y_test_fixed = (
            self._validate_split(base_split)
        )

        base_training_dataset = pd.concat(
            [X_train_base, y_train_base],
            axis=1,
        )

        studies = settings['studies']
        if not isinstance(studies, Mapping):
            raise TypeError('studies must be a mapping.')
        if not studies:
            raise ValueError('At least one study must be configured')
        
        supported_studies = {'model', 'sampling', 'data'}
        unknown_studies = set(settings['studies']) - supported_studies
        if unknown_studies:
            raise ValueError(
                f'Unsupported studies: {sorted(unknown_studies)}'
            )

        evaluation_methods = self._validate_evaluation_methods(
            settings.get(
                'evaluation_method',
                (EvaluationMethod.AVERAGING, EvaluationMethod.POINTWISE),
            )
        )

        for study_name, study_settings in studies.items():
            if not isinstance(study_settings, Mapping):
                raise TypeError(
                    f'Settings for study {study_name!r} must be a mapping.'
                )

        sampling_settings = studies.get(StudyName.SAMPLING)
        if sampling_settings is not None:
            strategies = sampling_settings.get('strategies', [])
            if isinstance(strategies, (str, bytes)) or not isinstance(
                strategies, Sequence
            ):
                raise TypeError('sampling strategies must be a sequence.')
            if any(not isinstance(strategy, str) for strategy in strategies):
                raise TypeError('Every sampling strategy must be a string.')
            if not strategies:
                raise ValueError(
                    'At least one sampling strategy must be configured.'
                )
            requested_strategies = set(strategies)
            unknown_strategies = requested_strategies - supported_strategies

            if unknown_strategies:
                raise ValueError(
                f'Unsupported sampling strategies: {sorted(unknown_strategies)}'
            )
        
        self._run_id = f'run_{uuid.uuid4().hex}'
        if self._results_df is None:
            self._results_df = self._load_or_initialize_table(
                RESULTS_FILENAME,
                self.RESULT_COLUMNS
            )

        if self._runs_metadata_df is None:
            self._runs_metadata_df = self._load_or_initialize_table(
                RUN_METADATA_FILENAME,
                self.RUN_METADATA_COLUMNS
            )

        for study, study_settings in studies.items():
            generator = self._build_generator(
                study,
                study_settings,
                training_dataset=base_training_dataset,
            )
            results, prediction_groups = self._get_results(
                n_iter,
                generator,
                study,
                base_split=base_split,
                random_state=random_state,
                test_size=test_size,
                save_predictions=save_predictions,
            )
            self._results_df = pd.concat([self._results_df, results], ignore_index=True)

            for variable, prediction_list in prediction_groups.items():
                prediction_matrix = np.stack(
                    prediction_list,
                    axis=0,
                )

                # Evaluate the predictions using the specified evaluation methods.
                #  *Hiearchy* run_id -> study -> variable -> evaluation_method -> values
                evaluations = self._evaluate_methods(
                    prediction_matrix,
                    pointwise_actuals=y_test_fixed.to_numpy(dtype=np.float32),
                    averaging_actuals=self.outputs_df.to_numpy(dtype=np.float32),
                    methods=evaluation_methods,
                )
                run_evaluations = self._evaluations.setdefault(
                    self._run_id,
                    {},
                )
                study_evaluations = run_evaluations.setdefault(
                    str(study),
                    {},
                )
                study_evaluations[variable] = evaluations

        self._record_run_metadata(
            n_iter=n_iter,
            random_state=random_state,
            test_size=test_size,
        )

        if save_results:
            self._results_df.to_csv(RESULTS_FILENAME, index=False)
            self._runs_metadata_df.to_csv(RUN_METADATA_FILENAME, index=False)
        
        return self

    def decompose_variance(
        self,
        view: list[str] | None = None,
        *,
        confidence: float = 0.95
    ) -> dict:
        '''
        To provide a breakdown of bias variance of previous runs. If no runs were performed,
        the analyzer should not provide any results.

        Parameters
        -------------
        view : list[str], default = ['model','sampling','data']
            The selection of variances to view.
        
        confidence : float, default = 0.95
            The confidence level for each metric.

        Returns
        -------------
        dict
            Summary results from each study. Contains averages, maximums, minimums, and
            confidence intervals for each metric.
        '''
        if (
            not isinstance(confidence, (int, float))
            or isinstance(confidence, bool)
        ):
            raise TypeError('confidence must be numeric.')
        if confidence <= 0 or confidence >= 1:
            raise ValueError('confidence must be between 0 and 1.')

        if view is None:
            view = list(StudyName)
        elif isinstance(view, (str, bytes)) or not isinstance(view, Sequence):
            raise TypeError('view must be a sequence of study names or None.')
        if not view:
            raise ValueError('view must contain at least one study name.')
        if any(not isinstance(study, str) for study in view):
            raise TypeError('Every study view must be a string.')
        unknown_views = set(view) - set(StudyName)
        if unknown_views:
            raise ValueError(f'Unsupported study views: {sorted(unknown_views)}')

        df = self._results_df
        if df is None:
            if not os.path.exists(RESULTS_FILENAME):
                raise FileNotFoundError(
                    f'No results file exists at {RESULTS_FILENAME!r}.'
                )
            df = self._load_or_initialize_table(
                RESULTS_FILENAME,
                self.RESULT_COLUMNS,
            )
        if df.empty:
            raise ValueError('No results are available for decomposition.')
        required_columns = {STUDY_FIELD_NAME, VARIABLE_FIELD_NAME}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise ValueError(
                f'Results are missing required columns: '
                f'{sorted(missing_columns)}.'
            )
        
        views = {}

        for study_group_name, study_group_df in df.groupby(STUDY_FIELD_NAME):
            if study_group_name not in view:
                continue

            views[study_group_name] = {}
            for var_group_name, var_group_df in study_group_df.groupby(VARIABLE_FIELD_NAME):
                metric_cols = [
                    column
                    for column in (
                        LOSS_FIELD_NAME,
                        *MetricName,
                        VARIANCE_FIELD_NAME,
                        MEAN_FIELD_NAME,
                        CONF_INTERVAL_LOWER_FIELD_NAME,
                        CONF_INTERVAL_UPPER_FIELD_NAME,
                    )
                    if column in var_group_df.columns
                ]

                averages = {}
                maximums = {}
                minimums = {}
                confidence_intervals = {}
                for col_name in metric_cols:
                    col_data = pd.to_numeric(
                        var_group_df[col_name],
                        errors='coerce',
                    ).dropna()
                    if col_data.empty:
                        continue
                    averages[col_name] = col_data.mean()
                    maximums[col_name] = col_data.max()
                    minimums[col_name] = col_data.min()

                    if len(col_data) > 1:
                        confidence_intervals[col_name] = stats.norm.interval(
                            confidence, loc=col_data.mean(), scale=stats.sem(col_data)
                        )
                    else:
                        confidence_intervals[col_name] = (np.nan, np.nan)
                
                views[study_group_name][var_group_name] = {
                    var_group_name: {
                        'averages': averages,
                        'maximums': maximums,
                        'minimums': minimums,
                        'confidence_intervals': confidence_intervals
                    }
                }
        
        return views
    
    def plot_disagreement_map(
        self,
        view:list[str] | None = None,
        plot_type:list[str] | None = None,
        plot_settings:dict | None = None
    ) -> None:
        '''
        To provide a plot of bias variance results of previous runs. If no runs were performed,
        the analyzer should not provide any results.

        Parameters
        -------------
        view : list[str], default = ['model','sampling','data']
            The selection of variances to view.

        plot_type : str, default = ['heatmap','histogram','KDE','uncertainty_bands','scatter_disagreement']
            Selection of visualizations for study.
        
        Returns
        -------------
        None
        '''
        if view is None:
            view = list(StudyName)
        elif isinstance(view, (str, bytes)) or not isinstance(view, Sequence):
            raise TypeError('view must be a sequence of study names or None.')
        if not view:
            raise ValueError('view must contain at least one study name.')
        if any(not isinstance(study, str) for study in view):
            raise TypeError('Every study view must be a string.')
        unknown_views = set(view) - set(StudyName)
        if unknown_views:
            raise ValueError(f'Unsupported study views: {sorted(unknown_views)}')

        if plot_type is None:
            plot_type = [PlotType.VARIANCE_CONTRIBUTION]
        elif isinstance(plot_type, (str, bytes)) or not isinstance(
            plot_type, Sequence
        ):
            raise TypeError('plot_type must be a sequence of plot names or None.')
        if not plot_type:
            raise ValueError('plot_type must contain at least one plot name.')
        if any(not isinstance(plot_name, str) for plot_name in plot_type):
            raise TypeError('Every plot type must be a string.')
        unknown_plot_types = set(plot_type) - set(PlotType)
        if unknown_plot_types:
            raise ValueError(
                f'Unsupported plot types: {sorted(unknown_plot_types)}'
            )
        if plot_settings is not None and not isinstance(plot_settings, dict):
            raise TypeError('plot_settings must be a dictionary or None.')

        results_df = self._results_df
        if results_df is None:
            if not os.path.exists(RESULTS_FILENAME):
                raise FileNotFoundError(
                    f'No results file exists at {RESULTS_FILENAME!r}.'
                )
            results_df = self._load_or_initialize_table(
                RESULTS_FILENAME,
                self.RESULT_COLUMNS,
            )
        if STUDY_FIELD_NAME not in results_df.columns:
            raise ValueError(
                f'Results are missing required column {STUDY_FIELD_NAME!r}.'
            )
        
        filtered_df = results_df[results_df[STUDY_FIELD_NAME].isin(view)]

        if filtered_df.empty:
            raise ValueError(
                'No results are available for the selected studies.'
            )

        if PlotType.VARIANCE_CONTRIBUTION in plot_type:
            plot_variance_contribution(
                filtered_df,
                settings=plot_settings,
            )

        if PlotType.PREDICTION_MEANS_BY_R2_SCORES in plot_type:
            plot_prediction_means_by_r2_scores(
                filtered_df,
                settings=plot_settings,
            )

        if PlotType.VARIANCE_DISTRIBUTION in plot_type:
            plot_variance_distribution(
                filtered_df,
                settings=plot_settings,
            )

        if PlotType.MEAN_DISTRIBUTION in plot_type:
            plot_mean_distribution(
                filtered_df,
                settings=plot_settings,
            )

        plt.show()
