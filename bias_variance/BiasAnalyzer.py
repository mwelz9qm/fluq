from __future__ import annotations

import os
import uuid
import h5py
import pandas as pd
import numpy as np
import scipy.stats as stats
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.layers import Input, Dense
from keras.metrics import (
    R2Score,
    MeanSquaredError,
    RootMeanSquaredError,
    MeanAbsoluteError
)
from keras.models import Model
from sklearn.model_selection import train_test_split
from bias_variance._constants import (
    ACTUALS_DATASET_NAME,
    CONF_INTERVAL_LOWER_FIELD_NAME,
    CONF_INTERVAL_UPPER_FIELD_NAME,
    FIT_ITERATIONS_DIR_NAME,
    ITERATION_FIELD_NAME,
    LOSS_FIELD_NAME,
    MEAN_FIELD_NAME,
    MetricName,
    MODEL_NAME,
    PlotType,
    PREDICTIONS_DATASET_NAME,
    PREDICTIONS_LAYER_NAME,
    RESULTS_FILENAME,
    RUN_ID_FIELD_NAME,
    SamplingStrategyName,
    STUDY_FIELD_NAME,
    StudyName,
    TIMESTAMP_FIELD_NAME,
    VARIABLE_FIELD_NAME,
    VARIANCE_FIELD_NAME,
)
from bias_variance.generators.ArchitectureGenerator import ArchitectureGenerator
from bias_variance.generators.Generator import Generator
from bias_variance.generators.SamplingGenerator import (
    SamplingGenerator,
    SamplingStrategy
)
from bias_variance._plotting import (
    plot_mean_distribution,
    plot_prediction_means_by_r2_scores,
    plot_variance_contribution,
    plot_variance_distribution,
)
from common.sampling._sampling import (
    get_quantile_stratified_random_samples,
    get_random_samples,
    generate_latin_hypercube_samples
)


# General workflow
# analyzer = BiasAnalyzer(...)
# analyzer.run_model_bias_study(...)
# analyzer.run_sampling_bias_study(...)
# analyzer.run_data_bias_study(...)
# analyzer.decompose_variance(...)
# analyzer.plot_disagreement_map(...)

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
    
    test_size : float = 0.2
        Fraction used for initial train/test split.

    random_state : int | None = None
        Seed used for reproducibility.

    _model: tensorflow.keras.Model | None = None
        Holds the base model.
    
    _init_weights: list[numpy.ndarray] | None = None
        Holds the initial weights before any training data is fitted on the base model.
    
    _results_df: pandas.DataFrame | None = None
        Caches the results after a study has ran.
    
    _run_id: str | None = None
        The associated run id. Updates after run_bias_studies() is called.

    Questions
    ------------
    - Should we select all plots by default or only select the best representation w/ an auto selection feature?
    '''

    METRIC_OPTIONS = {
        MetricName.RMSE: RootMeanSquaredError(name=MetricName.RMSE),
        MetricName.MSE: MeanSquaredError(name=MetricName.MSE),
        MetricName.MAE: MeanAbsoluteError(name=MetricName.MAE),
        MetricName.R2: R2Score(name=MetricName.R2),
    }
    RESULTS_FILENAME = RESULTS_FILENAME
    FIT_ITERATIONS_DIR_NAME = FIT_ITERATIONS_DIR_NAME
    STUDY_FIELD_NAME = STUDY_FIELD_NAME
    VARIABLE_FIELD_NAME = VARIABLE_FIELD_NAME

    def __init__(
        self,
        inputs_df: pd.DataFrame,
        outputs_df: pd.DataFrame,
        *,
        model_settings: dict = {
            'hidden_layers': [
                32, 32
            ],
            'activation' : 'relu',
            'optimizer' : 'adam',
            'loss' : 'mse',
            'metrics' : ['rmse','r2','mse','mae'],
            'epochs' : 100,
            'batch_size' : 10,
            'verbose' : 0
        },
        test_size: float = 0.2,
        random_state: int | None = None,
        _model: tf.keras.Model | None = None,
        _init_weights: list[np.ndarray] | None = None,
        _results_df: pd.DataFrame | None = None,
        _run_id: str | None = None
    ) -> None:
        self.inputs_df = inputs_df
        self.outputs_df = outputs_df
        self.model_settings = model_settings
        self.test_size = test_size
        self.random_state = random_state
        self._model = _model
        self._init_weights = _init_weights
        self._results_df = _results_df
        self._run_id = _run_id
    
    def _build_model(self, hidden_layers: list[int]) -> Model:
        '''
        Build model for studies. Uses default values specified in self.model_settings.
        '''
        # use keras functional api to build base model
        inputs = Input(shape=(self.inputs_df.shape[1],))
        x = inputs
        for size in hidden_layers:
            x = Dense(size, activation=self.model_settings['activation'])(x)
        outputs = Dense(self.outputs_df.shape[1], name=PREDICTIONS_LAYER_NAME)(x)
        model = Model(inputs=inputs, outputs=outputs, name=MODEL_NAME)
        # compile base model with settings
        model.compile(
            optimizer=self.model_settings['optimizer'],
            loss=self.model_settings['loss'],
            metrics=[
                BiasAnalyzer.METRIC_OPTIONS[metric]
                for metric in self.model_settings['metrics']
            ],
        )
        return model
    
    def _init_model(self) -> None:
        '''
        Initializes the base model for studies. Use when starting study run.
        '''
        if self._model is None:
            self._model = self._build_model(self.model_settings['hidden_layers'])
            # store initial weights when model needs to be retrained
            self._init_weights = self._model.get_weights()

    def _init_results_csv(self) -> None:
        '''
        Initializes the results csv file. Use when retrieving a previous study run for
        bias-variance decomposition or plotting.
        '''
        columns = [
            RUN_ID_FIELD_NAME,
            ITERATION_FIELD_NAME,
            STUDY_FIELD_NAME,
            VARIABLE_FIELD_NAME,
            LOSS_FIELD_NAME,
            *BiasAnalyzer.METRIC_OPTIONS.keys(),
            VARIANCE_FIELD_NAME,
            MEAN_FIELD_NAME,
            CONF_INTERVAL_LOWER_FIELD_NAME,
            CONF_INTERVAL_UPPER_FIELD_NAME,
        ]

        if os.path.exists(RESULTS_FILENAME):
            self._results_df = pd.read_csv(
                RESULTS_FILENAME
            ).reindex(columns=columns)
        
        else:
            self._results_df = pd.DataFrame(columns=columns)
        
        os.makedirs(FIT_ITERATIONS_DIR_NAME, exist_ok=True)

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

    def _get_test_result_and_data(
        self,
        split: tuple[
            pd.DataFrame,
            pd.DataFrame,
            pd.DataFrame,
            pd.DataFrame,
        ] | None = None,
        hidden_layers: list[int] | None = None,
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
        if hidden_layers is None:
            self._init_model()
            model = self._model
            # reset weights
            model.set_weights(self._init_weights)
        else:
            model = self._build_model(hidden_layers)
        
        X_train, X_test, y_train, y_test = split or train_test_split(
            self.inputs_df,
            self.outputs_df,
            test_size=self.test_size,
            random_state=self.random_state
        )

        # train
        model.fit(
            X_train,
            y_train,
            epochs=self.model_settings['epochs'],
            batch_size=self.model_settings['batch_size'],
            verbose=self.model_settings['verbose']
        )

        predictions = model.predict(X_test)

        # evaluate
        scores = model.evaluate(X_test, y_test, batch_size=self.model_settings['batch_size'], return_dict=True)
        prediction_values = np.asarray(predictions).reshape(-1)
        variance = np.var(prediction_values)
        mean = np.mean(prediction_values)
        conf_interval = stats.norm.interval(
            0.95,
            loc=mean,
            scale=stats.sem(prediction_values),
        )

        metrics = {
            VARIANCE_FIELD_NAME: variance,
            MEAN_FIELD_NAME: mean,
            CONF_INTERVAL_LOWER_FIELD_NAME: conf_interval[0],
            CONF_INTERVAL_UPPER_FIELD_NAME: conf_interval[1],
        }

        return scores | metrics, predictions, y_test

    def _get_results(
        self,
        n_iter: int,
        generator: Generator[tuple[int, ...]] | Generator[pd.DataFrame],
        study: str,
        *,
        save_predictions: bool = True,
    ) -> pd.DataFrame:
        '''
        Generate, train, and evaluate all variations for a configured study.

        Parameters
        ----------
        n_iter : int
            Number of times to invoke the generator.
        generator : Generator[tuple[int, ...]] | Generator[pandas.DataFrame]
            Architecture or sampled-dataset generator used by the study.
        study : str
            Study type that determines how generated variations are prepared.
            Currently supports ``'model'`` and ``'sampling'``.
        save_predictions : bool, default=True
            Whether to persist predictions and actual values for each variation.

        Returns
        -------
        pandas.DataFrame
            One result row per generated label and iteration.
        '''
        results = pd.DataFrame()

        for i in np.arange(n_iter):
            iteration_random_state = (
                None
                if self.random_state is None
                else self.random_state + i
            )

            variations = generator.generate(random_state=iteration_random_state)

            for label, variation in variations.items():
                hidden_layers = None
                split = None

                if study == StudyName.MODEL:
                    hidden_layers = list(variation)
                
                elif study == StudyName.SAMPLING:
                    sampled_inputs_df = variation[self.inputs_df.columns]
                    sampled_outputs_df = variation[self.outputs_df.columns]
                    split = train_test_split(
                        sampled_inputs_df,
                        sampled_outputs_df,
                        test_size=self.test_size,
                        random_state=iteration_random_state
                    )
                
                result, predictions, actuals = self._get_test_result_and_data(hidden_layers=hidden_layers, split=split)

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
                    VARIABLE_FIELD_NAME: label
                } | result

                results = pd.concat([results, pd.DataFrame([df_row])], ignore_index=True)
        
        return results
    
    def _build_generator(
        self,
        study: str,
        settings: dict[str, object],
    ) -> ArchitectureGenerator | SamplingGenerator:
        '''
        Construct the concrete generator configured for a study.

        Parameters
        ----------
        study : str
            Study type. Supported values are ``'model'`` and ``'sampling'``.
        settings : dict[str, object]
            Generator settings for the selected study. Model settings describe
            architecture families; sampling settings contain strategy names.

        Returns
        -------
        ArchitectureGenerator | SamplingGenerator
            Generator initialized with the study settings and, for sampling,
            the analyzer's combined input and output dataset.

        Raises
        ------
        ValueError
            If ``study`` is unsupported.
        '''
        if study == StudyName.MODEL:
            return ArchitectureGenerator(settings=settings)
        if study == StudyName.SAMPLING:
            sampling_strategies = []
            strategies = settings.get('strategies', [])

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
            
            dataset = pd.concat([self.inputs_df, self.outputs_df], axis=1)
            return SamplingGenerator(dataset=dataset, strategies=sampling_strategies)
        
        raise ValueError(f'Unsupported study: {study!r}')
    
    def run_bias_studies(
        self,
        settings: dict | None = None,
        *,
        save_results: bool = True,
        save_predictions: bool = True
    ) -> 'BiasAnalyzer':
        '''
        Default Settings
        -------------
        >>> settings = {
        ...     'n_iter': 100,
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
        ...     }
        ... }
        '''
        default_settings = {
            'n_iter': 100,
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
            }
        }
        settings = settings or default_settings

        n_iter = settings['n_iter']
        if not isinstance(n_iter, int) or isinstance(n_iter, bool):
            raise TypeError('n_iter must be an integer.')
        if n_iter <= 0:
            raise ValueError('n_iter must be greater than 0.')
        
        if not settings['studies']:
            raise ValueError('At least one study must be configured')
        
        supported_studies = {StudyName.MODEL, StudyName.SAMPLING}
        unknown_studies = set(settings['studies']) - supported_studies
        if unknown_studies:
            raise ValueError(
                f'Unsupported studies: {sorted(unknown_studies)}'
            )
        
        supported_strategies = set(SamplingStrategyName)
        sampling_settings = settings['studies'].get(StudyName.SAMPLING)
        if sampling_settings is not None:
            requested_strategies = set(sampling_settings.get('strategies', []))
            unknown_strategies = requested_strategies - supported_strategies

            if unknown_strategies:
                raise ValueError(
                f'Unsupported sampling strategies: {sorted(unknown_strategies)}'
            )
        
        self._init_model()
        self._run_id = f'run_{uuid.uuid4().hex}'
        os.makedirs(FIT_ITERATIONS_DIR_NAME, exist_ok=True)
        
        for study, study_settings in settings['studies'].items():
            generator = self._build_generator(study, study_settings)
            results = self._get_results(n_iter, generator, study, save_predictions=save_predictions)
            self._results_df = pd.concat([self._results_df, results], ignore_index=True)
        
        if save_results:
            self._results_df.to_csv(RESULTS_FILENAME, index=False)
        
        return self

    def decompose_variance(
        self,
        view:list[str]=['model','sampling','data'],
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
        # Error conditions
        if confidence <= 0 or confidence >= 1:
            raise ValueError('confidence must be between 0 and 1.')

        df = self._results_df
        if df is None:
            df = pd.read_csv(RESULTS_FILENAME)
        
        views = {}

        for study_group_name, study_group_df in df.groupby(STUDY_FIELD_NAME):
            if study_group_name not in view:
                continue

            views[study_group_name] = {}
            for var_group_name, var_group_df in study_group_df.groupby(VARIABLE_FIELD_NAME):
                metric_cols = [
                    col for col in var_group_df.columns
                    if col not in {
                        STUDY_FIELD_NAME,
                        VARIABLE_FIELD_NAME,
                        RUN_ID_FIELD_NAME,
                        TIMESTAMP_FIELD_NAME,
                    }
                ]

                averages = {}
                maximums = {}
                minimums = {}
                confidence_intervals = {}
                for col_name in metric_cols:
                    col_data = var_group_df[col_name].dropna()
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
        # variance-contribution
        view = view or list(StudyName)
        plot_type = plot_type or [PlotType.VARIANCE_CONTRIBUTION]

        results_df = self._results_df
        if results_df is None:
            results_df = pd.read_csv(RESULTS_FILENAME)
        
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
