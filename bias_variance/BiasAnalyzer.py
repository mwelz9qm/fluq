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
from common.sampling.Sampler import Sampler
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

class BiasAnalyzerConfigMeta(type):
    '''
    Provides static, const member variables for the BiasAnalyzer class.
    '''
    @property
    def METRIC_OPTIONS(cls):
        '''
        Returns all keras functional model metric selections for analysis.
        '''
        return {
            'rmse' : RootMeanSquaredError(name='rmse'),
            'mse' : MeanSquaredError(name='mse'),
            'mae' : MeanAbsoluteError(name='mae'),
            'r2' : R2Score(name='r2')
        }
    
    @property
    def RESULTS_FILENAME(cls):
        '''
        Returns the saved results dataframe filename (w/ file extension) for later analysis.
        '''
        return 'bias_variance_results.csv'
    
    @property
    def FIT_ITERATIONS_DIR_NAME(cls):
        '''
        Returns directory name for saved predictions and actuals from model training/fitting.
        '''
        return 'iterations'
    
    @property
    def STUDY_FIELD_NAME(cls):
        '''
        Returns study field name for results table.
        '''
        return 'study'
    
    @property
    def VARIABLE_FIELD_NAME(cls):
        '''
        Returns variable field name for results table. This is dependent on the type of study.
        '''
        return 'variable'


# General workflow
# analyzer = BiasAnalyzer(...)
# analyzer.run_model_bias_study(...)
# analyzer.run_sampling_bias_study(...)
# analyzer.run_data_bias_study(...)
# analyzer.decompose_variance(...)
# analyzer.plot_disagreement_map(...)

class BiasAnalyzer(metaclass=BiasAnalyzerConfigMeta):
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
    

    Questions
    ------------
    - Should we select all plots by default or only select the best representation w/ an auto selection feature?
    '''

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

    def _init_results_csv(self):
        '''
        Initializes the results csv file. Use when retrieving a previous study run for
        bias-variance decomposition or plotting.
        '''
        columns = [
            'run_id',
            'iteration',
            BiasAnalyzer.STUDY_FIELD_NAME,
            BiasAnalyzer.VARIABLE_FIELD_NAME,
            'loss',
            *BiasAnalyzer.METRIC_OPTIONS.keys(),
            'variance',
            'mean',
            'conf_interval_lower',
            'conf_interval_upper',
        ]
        if os.path.exists(BiasAnalyzer.RESULTS_FILENAME):
            self._results_df = pd.read_csv(
                BiasAnalyzer.RESULTS_FILENAME
            ).reindex(columns=columns)
        else:
            self._results_df = pd.DataFrame(columns=columns)
        os.makedirs(BiasAnalyzer.FIT_ITERATIONS_DIR_NAME, exist_ok=True)
    
    def _build_model(self, hidden_layers) -> Model:
        '''
        Build model for studies. Uses default values specified in self.model_settings.
        '''
        # use keras functional api to build base model
        inputs = Input(shape=(self.inputs_df.shape[1],))
        x = inputs
        for hidden_layer in hidden_layers:
            x = Dense(hidden_layer, activation=self.model_settings['activation'])(x)
        outputs = Dense(self.outputs_df.shape[1], name='predictions')(x)
        model = Model(inputs=inputs, outputs=outputs, name='functional_model')
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
    
    def _init_model(self):
        '''
        Initializes the base model for studies. Use when starting study run.
        '''
        if self._model is None:
            self._model = self._build_model(self.model_settings['hidden_layers'])
            # store initial weights when model needs to be retrained
            self._init_weights = self._model.get_weights()

    def _save_predictions_and_actuals(
        self,
        predictions,
        actuals,
        *,
        study: str,
        label: str,
        iteration: int,
    ):
        '''
        Saves the predictions and actuals from a given trained model.

        Parameters
        -------------
        predictions
            model predictions
        actuals
            model actuals relative to predictions by row
        study: str
            study name
        label: str
            label or variable name in study
        iteration: int
            iteration id in a study run

        Returns
        --------------
        None
        '''
        if self._run_id is None:
            raise ValueError('_run_id is None.')
        pred_file_path = os.path.join(
            BiasAnalyzer.FIT_ITERATIONS_DIR_NAME,
            f'{self._run_id}.h5'
        )
        group_path = f'{study}/{label}/iteration_{iteration}'
        with h5py.File(pred_file_path, 'a') as hf:
            group = hf.create_group(group_path)
            group.create_dataset('predictions', data=predictions)
            group.create_dataset('actuals', data=actuals)

    class Architector:
        def __init__(self, settings: dict | None = None):
            supported = {
                'wide',
                'narrow',
                'taper',
                'reverse_taper',
                'combined_taper'
            }
            default_settings = {
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
            }
            settings = settings or default_settings
            normalized_settings = {}
            for architecture_name, overrides in settings.items():
                if architecture_name not in supported:
                    raise ValueError(
                        f'Unsupported architecture: {architecture_name}'
                    )
                normalized_settings[architecture_name] = (
                    default_settings[architecture_name]
                    | (overrides or {})
                )
            self.settings = normalized_settings
        
        def generate(self, random_state: int | None = None) -> dict[str, np.ndarray]:
            rng = np.random.default_rng(random_state)
            hidden_layers = {}

            for architecture_name, settings in self.settings.items():
                min_layers, max_layers = settings['layers']
                n_layers = rng.integers(min_layers, max_layers)
                sizes = np.zeros(n_layers, dtype=int)

                if architecture_name in ['wide', 'narrow']:
                    min_neurons, max_neurons = settings['neurons']
                    sizes = rng.integers(min_neurons, max_neurons, size=n_layers)
                elif architecture_name in ['taper', 'reverse_taper', 'combined_taper']:
                    min_neurons, max_neurons = settings['init_neurons']
                    init_neurons = rng.integers(min_neurons, max_neurons)
                    taper_rate = rng.uniform(*settings['taper_rate'])
                    max_allowed_neurons = settings['max_neurons']
                    if architecture_name == 'taper':
                        for i in np.arange(n_layers):
                            size = round(init_neurons * ((1 + taper_rate) ** i))
                            sizes[i] = min(size, max_allowed_neurons)
                    elif architecture_name == 'reverse_taper':
                        for i in np.arange(n_layers):
                            size = round(init_neurons * ((1 - taper_rate) ** i))
                            sizes[i] = max(size, 1)
                    elif architecture_name == 'combined_taper':
                        midpoint = max(1, int(np.ceil(n_layers / 2)))
                        for i in np.arange(n_layers):
                            if i < midpoint:
                                size = round(init_neurons * ((1 + taper_rate) ** i))
                            else:
                                peak = init_neurons * ((1 + taper_rate) ** (midpoint  - 1))
                                size = round(peak * ((1 - taper_rate) ** (i - midpoint + 1)))
                            sizes[i] = min(max(size, 1), max_allowed_neurons)
                
                hidden_layers[architecture_name] = sizes

            return hidden_layers

    def _get_test_result_and_data(self, split = None, hidden_layers = None):
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
            'variance': variance,
            'mean': mean,
            'conf_interval_lower': conf_interval[0],
            'conf_interval_upper': conf_interval[1],
        }

        return scores | metrics, predictions, y_test
    
    def _get_variations(self, study, generator, random_state):
        injected_kwargs = {} # For injecting the random_state into sampling function.
        if random_state is not None:
            injected_kwargs['random_state'] = random_state

        variations = {}
        if study == 'model':
            variations = generator.generate(random_state)
        elif study == 'sampling':
            variations = generator.generate(pd.concat([self.inputs_df, self.outputs_df], axis=1), injected_kwargs)
        elif study == 'data':
            variations = {}
        return variations

    def _get_results(self, n_iter, generator, study, save_predictions=True):
        results = pd.DataFrame()

        for i in np.arange(n_iter):
            iteration_random_state = None
            if self.random_state is not None:
                iteration_random_state = self.random_state + i
            variations = self._get_variations(study, generator, iteration_random_state)
            for label, variation in variations.items():
                hidden_layers = None
                split = None
                if study == 'model':
                    hidden_layers = variation[variation > 0].tolist()
                elif study == 'sampling':
                    sampled_inputs_df = variation[self.inputs_df.columns]
                    sampled_outputs_df = variation[self.outputs_df.columns]
                    split = train_test_split(
                        sampled_inputs_df,
                        sampled_outputs_df,
                        test_size=self.test_size,
                        random_state=iteration_random_state
                    )
                elif study == 'data':
                    split = None
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
                    'run_id': self._run_id,
                    'iteration': int(i),
                    BiasAnalyzer.STUDY_FIELD_NAME: study,
                    BiasAnalyzer.VARIABLE_FIELD_NAME: label
                } | result
                results = pd.concat([results, pd.DataFrame([df_row])], ignore_index=True)
        
        return results
    
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
        settings = {
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
        
        supported_studies = {'model', 'sampling'}
        unknown_studies = set(settings['studies']) - supported_studies
        if unknown_studies:
            raise ValueError(
                f'Unsupported studies: {sorted(unknown_studies)}'
            )
        
        supported_strategies = {'bootstrap', 'stratified', 'lhs'}
        sampling_settings = settings['studies'].get('sampling')
        if sampling_settings is not None:
            requested_strategies = set(sampling_settings.get('strategies', []))
            unknown_strategies = requested_strategies - supported_strategies
            if unknown_strategies:
                raise ValueError(
                f'Unsupported sampling strategies: {sorted(unknown_strategies)}'
            )
        
        self._init_model()
        self._run_id = f'run_{uuid.uuid4().hex}'
        os.makedirs(BiasAnalyzer.FIT_ITERATIONS_DIR_NAME, exist_ok=True)
        
        for study, study_settings in settings['studies'].items():
            results = pd.DataFrame()
            if study == 'model':
                architector = self.Architector(settings=study_settings)
                results = self._get_results(n_iter, architector, study, save_predictions=save_predictions)
            elif study == 'sampling':
                sampler = Sampler()
                strategies = study_settings.get('strategies', [])
                if 'bootstrap' in strategies:
                    kwargs = {
                        'sample_fraction': 1.0,
                        'with_replacement': True
                    }
                    sampler.add_strategy('bootstrap', get_random_samples, **kwargs)
                if 'stratified' in strategies:
                    kwargs = {
                        'stratify_col_index': self.inputs_df.shape[1],
                        'sample_fraction': 1.0,
                        'with_replacement': True
                    }
                    sampler.add_strategy('stratified', get_quantile_stratified_random_samples, **kwargs)
                if 'lhs' in strategies:
                    kwargs = {
                        'sample_fraction': 1.0,
                    }
                    sampler.add_strategy('lhs', generate_latin_hypercube_samples, **kwargs)
                results = self._get_results(n_iter, sampler, study, save_predictions=save_predictions)
            self._results_df = pd.concat([self._results_df, results], ignore_index=True)
        if save_results:
            self._results_df.to_csv(BiasAnalyzer.RESULTS_FILENAME, index=False)
        return self

    def _train_and_evaluate_model(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        *,
        hidden_layers=None,
    ) -> dict:
        '''
        Trains and evaluates the model on a given train-test split.
        
        Parameters
        --------------
        X_train
            Input data from train set.
        
        y_train
            Output data from train set.
        
        X_test
            Input data from test set.
        
        y_test
            Output data from test set.

        Returns
        ----------------
        A dictionary object with the loss and metric values defined in self.model_settings,
        and the mean, variance, and 95% confidence interval on the predictions.
        '''
        if hidden_layers is None:
            self._init_model()
            model = self._model
            # reset weights
            model.set_weights(self._init_weights)
        else:
            model = self._build_model(hidden_layers)

        # train
        model.fit(
            X_train,
            y_train,
            epochs=self.model_settings['epochs'],
            batch_size=self.model_settings['batch_size'],
            verbose=self.model_settings['verbose']
        )

        predictions = model.predict(X_test)

        # save predictions and actuals in directory
        self._save_predictions_and_actuals(predictions, y_test)

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
            'variance': variance,
            'mean': mean,
            'conf_interval_lower': conf_interval[0],
            'conf_interval_upper': conf_interval[1],
        }

        return scores | metrics


    def _generate_hidden_layer_sizes(
        self,
        architecture_settings: dict[str, dict],
        n_sizes: int,
    ) -> dict[str, np.ndarray]:
        '''
        Creates a 2D matrix of ints that define an array of architectures with a specified hidden layer size.

        Parameters
        -------------
        architecture_settings: dict
            Defines the architecture to generate and settings associated with it (i.e., max/min layer bounds, max/min neuron bounds, taper rate bounds)
        
        n_sizes: int
            Number of hidden layer sizes to generate per architecture.
        
        Returns
        --------------
        A dictionary with str keys of every architecture name that map to 2D numpy matrices of ints, specifying the hidden layer sizes
        
        Settings Example
        -----------------
        >>> architecture_settings = {
        ...     'wide': {
        ...         'layers': (1, 16), # (lower bound, upper bound): left val inclusive, right val exclusive
        ...         'neurons': (64, 256),
        ...     },
        ...     'narrow': {
        ...         'layers': (16, 64),
        ...         'neurons': (2, 64),
        ...     },
        ...     'taper': {
        ...         'layers': (16, 64),
        ...         'init_neurons': (1, 9), # starting layer's number of neurons
        ...         'taper_rate': (0.25, 0.5), # rate of layer size increase
        ...         'max_neurons': 256, # maximum neurons for any layer
        ...     },
        ...     'reverse_taper': {
        ...         'layers': (16, 64),
        ...         'init_neurons': (128, 256),
        ...         'taper_rate': (0.25, 0.5), # rate of layer size decrease
        ...         'max_neurons': 256,
        ...     },
        ...     'combined_taper': {
        ...         'layers': (16, 64),
        ...         'init_neurons': (1, 9),
        ...         'taper_rate': (0.25, 0.5), # rate of layer size increase and decrease
        ...         'max_neurons': 256,
        ...     }
        ... }
        '''
        if n_sizes <= 0:
            raise ValueError('n_sizes must be greater than 0.')

        default_settings = {
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
        }

        rng = np.random.default_rng(self.random_state)
        hidden_layer_sizes = {}

        for architecture_name, settings in architecture_settings.items():
            if architecture_name not in default_settings:
                raise ValueError(f'Unsupported architecture: {architecture_name}')

            settings = default_settings[architecture_name] | (settings or {})
            min_layers, max_layers = settings['layers']
            n_columns = max_layers - 1
            architecture_sizes = np.zeros((n_sizes, n_columns), dtype=int)

            for row_idx in np.arange(n_sizes):
                n_layers = rng.integers(min_layers, max_layers)

                if architecture_name in ['wide', 'narrow']:
                    min_neurons, max_neurons = settings['neurons']
                    sizes = rng.integers(min_neurons, max_neurons, size=n_layers)
                else:
                    min_neurons, max_neurons = settings['init_neurons']
                    init_neurons = rng.integers(min_neurons, max_neurons)
                    taper_rate = rng.uniform(*settings['taper_rate'])
                    max_allowed_neurons = settings['max_neurons']
                    sizes = np.zeros(n_layers, dtype=int)

                    if architecture_name == 'taper':
                        for layer_idx in np.arange(n_layers):
                            size = round(init_neurons * ((1 + taper_rate) ** layer_idx))
                            sizes[layer_idx] = min(size, max_allowed_neurons)

                    elif architecture_name == 'reverse_taper':
                        for layer_idx in np.arange(n_layers):
                            size = round(init_neurons * ((1 - taper_rate) ** layer_idx))
                            sizes[layer_idx] = max(size, 1)

                    else:
                        midpoint = max(1, int(np.ceil(n_layers / 2)))
                        for layer_idx in np.arange(n_layers):
                            if layer_idx < midpoint:
                                size = round(init_neurons * ((1 + taper_rate) ** layer_idx))
                            else:
                                peak = init_neurons * ((1 + taper_rate) ** (midpoint - 1))
                                size = round(peak * ((1 - taper_rate) ** (layer_idx - midpoint + 1)))
                            sizes[layer_idx] = min(max(size, 1), max_allowed_neurons)

                architecture_sizes[row_idx, :n_layers] = sizes

            hidden_layer_sizes[architecture_name] = architecture_sizes

        return hidden_layer_sizes

    def run_model_bias_study(
        self,
        architectures: list[str] = ['wide','narrow','taper','reverse_taper','combined_taper'], 
        n_iters: int = 100,
    ) -> 'BiasAnalyzer':
        '''
        To create multiple model architectures based on shape types (i.e., wide, narrow, taper,
        reverse_taper, and combined_taper), train base dataset on all models with base train dataset,
        and make predictions with base test dataset per model architecture for comparison results.

        Parameters
        ------------
        architectures: list[str]
            A list of architectures to use in study.
        
        n_iters: int = 100
            Number of iterations to run in study.

        Returns
        ------------
        BiasAnalyzer
        '''
        architecture_settings = dict.fromkeys(architectures)
        hidden_layer_sizes = self._generate_hidden_layer_sizes(architecture_settings, n_iters)
        X_train, X_test, y_train, y_test = train_test_split(
            self.inputs_df,
            self.outputs_df,
            test_size=self.test_size,
            random_state=self.random_state
        )
        results_df = pd.DataFrame()
        for label, sizes in hidden_layer_sizes.items():
            for size in sizes:
                hidden_layers = size[size > 0].tolist()
                results_row = {
                    BiasAnalyzer.STUDY_FIELD_NAME: 'model',
                    BiasAnalyzer.VARIABLE_FIELD_NAME: label,
                } | self._train_and_evaluate_model(
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    hidden_layers=hidden_layers
                )
                results_df = pd.concat([results_df, pd.DataFrame([results_row])], ignore_index=True)
        self._results_df = results_df # copy results
        self._results_df.to_csv(BiasAnalyzer.RESULTS_FILENAME, index=False) # then export to csv
        return self

    def run_sampling_bias_study(
        self,
        sampler: Sampler,
        *,
        n_iter: int = 100,
    ) -> 'BiasAnalyzer':
        '''
        Creates multiple, sampled datasets based on strategies (i.e., bootstrap, lhs, and
        stratified), trains on all generated train datasets with base model, and makes predictions
        with generated test dataset per train dataset for analysis.

        Parameters
        ------------
        sampler: Sampler
            Generates the sampled dataset based on the Sampler object's set strategies.
        
        n_iter: int = 100
            Number of iterations per sampling strategy.

        Returns
        ------------
        BiasAnalzer
        '''
        # Error conditions
        if n_iter <= 1:
            raise ValueError('n_iter must be greater than 1 to run study.')
        
        for label, _, _, kwargs in sampler.strategies:
            if 'random_state' in kwargs:
                raise ValueError(
                    f'Sampler strategy \'{label}\' cannot include random_state.'
                    'Set random_state on BiasAnalyzer instead.'
                )

        # before running study, initialize self._model based on self.model_settings
        self._init_model()

        results_df = pd.DataFrame() # To hold results
        dataset_df = pd.concat([self.inputs_df, self.outputs_df], axis=1) # merge inputs and outputs for sampling

        # Loop through all iterations and apply all sampling strategies from the Sampler.
        for i in np.arange(n_iter):
            # Set the random_state per iteration if the analyzer's random_state is not None.
            iteration_random_state = None
            if self.random_state is not None:
                iteration_random_state = self.random_state + i
            
            injected_kwargs = {} # For injecting the random_state into sampling function.
            if iteration_random_state is not None:
                injected_kwargs['random_state'] = iteration_random_state
            
            sample_sets = sampler.generate(dataset_df, injected_kwargs) # Get samples for each strategy.

            # Loop through all generated, sampled sets, get train-test splits, and save results.
            for label, sampled_df in sample_sets.items():
                sampled_inputs_df = sampled_df[self.inputs_df.columns]
                sampled_outputs_df = sampled_df[self.outputs_df.columns]

                X_train, X_test, y_train, y_test = train_test_split(
                    sampled_inputs_df,
                    sampled_outputs_df,
                    test_size=self.test_size,
                    random_state=iteration_random_state
                )

                results_row = {
                    BiasAnalyzer.STUDY_FIELD_NAME: 'sampling',
                    BiasAnalyzer.VARIABLE_FIELD_NAME: label
                } | self._train_and_evaluate_model(X_train, y_train, X_test, y_test)
                results_df = pd.concat([results_df, pd.DataFrame([results_row])], ignore_index=True)
            
        self._results_df = results_df # copy results
        self._results_df.to_csv(BiasAnalyzer.RESULTS_FILENAME, index=False) # then export to csv

        return self
        

    def run_data_bias_study(
        self, 
        cv_folds:int = 5, 
        n_iter:int = 10,
        shuffle:bool = True
    ) -> None:
        '''
        To create multiple train/test datasets based on cross validation folds,
        train on base model with generated train dataset, and make predictions
        with generated test dataset per paired train dataset for comparison results.

        Parameters
        ------------
        cv_folds : int, default = 5
            Must be greater than 1.
        
        n_iter : int, default = 10
            The number of cv iterations. Must be greater than 0.

        Returns
        ------------
        None

        TODO (implementation):
        ------------
        - Loop through each fold in folds.
            - Train and evaluate model on each fold (helper function call).
        '''
        pass

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
            df = pd.read_csv(BiasAnalyzer.RESULTS_FILENAME)
        
        views = {}

        for study_group_name, study_group_df in df.groupby(BiasAnalyzer.STUDY_FIELD_NAME):
            if study_group_name not in view:
                continue

            views[study_group_name] = {}
            for var_group_name, var_group_df in study_group_df.groupby(BiasAnalyzer.VARIABLE_FIELD_NAME):
                metric_cols = [
                    col for col in var_group_df.columns
                    if col not in {
                        BiasAnalyzer.STUDY_FIELD_NAME,
                        BiasAnalyzer.VARIABLE_FIELD_NAME,
                        'run_id',
                        'timestamp'
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

    def _get_result_df(self):
        results_df = self._results_df
        if results_df is None:
            results_df = pd.read_csv(BiasAnalyzer.RESULTS_FILENAME)
        return results_df
    
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
        view = view or ['model', 'sampling', 'data']
        plot_type = plot_type or ['variance_contribution']

        results_df = self._get_result_df()
        filtered_df = results_df[results_df[BiasAnalyzer.STUDY_FIELD_NAME].isin(view)]

        if filtered_df.empty:
            raise ValueError(
                'No results are available for the selected studies.'
            )

        if 'variance_contribution' in plot_type:
            plot_variance_contribution(
                filtered_df,
                settings=plot_settings,
            )

        if 'prediction_means_by_r2_scores' in plot_type:
            plot_prediction_means_by_r2_scores(
                filtered_df,
                settings=plot_settings,
            )

        if 'variance_distribution' in plot_type:
            plot_variance_distribution(
                filtered_df,
                settings=plot_settings,
            )

        if 'mean_distribution' in plot_type:
            plot_mean_distribution(
                filtered_df,
                settings=plot_settings,
            )

        plt.show()
