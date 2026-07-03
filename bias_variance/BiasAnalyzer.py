import os
import datetime
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
    ) -> None:
        self.inputs_df = inputs_df
        self.outputs_df = outputs_df
        self.model_settings = model_settings
        self.test_size = test_size
        self.random_state = random_state
        self._model = _model
        self._init_weights = _init_weights
        self._results_df = _results_df

    def _init_results_csv(self):
        '''
        Initializes the results csv file.
        '''
        if os.path.exists(BiasAnalyzer.RESULTS_FILENAME):
            self._results_df = pd.read_csv(BiasAnalyzer.RESULTS_FILENAME)
        else:
            columns = ['run_id', 'timestamp']
            columns.append(list(BiasAnalyzer.METRIC_OPTIONS.keys()))
            self._results_df = pd.DataFrame(columns=columns)
        os.makedirs(BiasAnalyzer.FIT_ITERATIONS_DIR_NAME, exist_ok=True)

    def _init_model(self):
        '''
        Initializes the base model for studies.
        '''
        if self._model is None:
            # use keras functional api to build base model
            inputs = Input(shape=(self.inputs_df.shape[1],))
            x = inputs
            for hidden_layer in self.model_settings['hidden_layers']:
                x = Dense(hidden_layer, activation=self.model_settings['activation'])(x)
            outputs = Dense(self.outputs_df.shape[1], name='predictions')(x)
            self._model = Model(inputs=inputs, outputs=outputs, name='functional_model')
            self._model.summary()
            # compile base model with settings
            self._model.compile(
                optimizer=self.model_settings['optimizer'],
                loss=self.model_settings['loss'],
                metrics=[BiasAnalyzer.METRIC_OPTIONS[metric] for metric in self.model_settings['metrics']]
            )
            # store initial weights when model needs to be retrained
            self._init_weights = self._model.get_weights()

    def _save_predictions_and_actuals(self, predictions, y_test):
        '''
        Saves the predictions and actuals from a given trained model.

        Parameters
        -------------
        X_test
            Input data from test set.

        y_test
            Output data from test set.

        Returns
        --------------
        None
        '''
        run_id = f'run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}'
        pred_file_path = f'{BiasAnalyzer.FIT_ITERATIONS_DIR_NAME}/{run_id}.h5'
        with h5py.File(pred_file_path, 'w') as hf:
            hf.create_dataset('predictions', data=predictions)
            hf.create_dataset('actuals', data=y_test)

    def _train_and_evaluate_model(self, X_train, y_train, X_test, y_test) -> dict:
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
        and the mean and variance on the predictions.
        '''
        # reset weights
        self._model.set_weights(self._init_weights)
        # train
        self._model.fit(
            X_train,
            y_train,
            epochs=self.model_settings['epochs'],
            verbose=self.model_settings['verbose']
        )

        predictions = self._model.predict(X_test)

        # save predictions and actuals in directory
        self._save_predictions_and_actuals(predictions, y_test)

        # evaluate
        scores = self._model.evaluate(X_test, y_test, batch_size=self.model_settings['batch_size'], return_dict=True)
        variance = np.var(predictions)
        mean = np.mean(predictions)
        
        return scores | {'variance': variance} | {'mean': mean}


    def run_model_bias_study(
        self, 
        architecture_types:list[str]=['wide','narrow','taper','reverse_taper','combined_taper'], 
        n_architectures=10,
        metrics:list[str]=['root_mean_squared_error','r2','mse','mae']
    ) -> None:
        '''
        To create multiple model architectures based on shape types (i.e., wide, narrow, taper,
        reverse_taper, and combined_taper), train base dataset on all models with base train dataset,
        and make predictions with base test dataset per model architecture for comparison results.

        Parameters
        ------------
        architecture_configs : dict | None = None
            Allows any user-defined configurations of architecture types for Keras model.

        architecture_types : list[str], default = ['wide','narrow','taper','reverse_taper','combined_taper'] i.e. all model arch types
            Must contain at least one architecture type to run.
        
        n_architectures : int, default = 10
            The number of architectures per type.

        metrics : list[str]=['rmse','r2','mse','mae']
            Used to evaluate model based on different metrics.

        Returns
        ------------
        None

        TODO (implementation):
        ------------
        - Loop through each architecture in architecture_types.
            - Build a model with that architecture.
            - Train and evaluate model (helper function call).
        '''
        pass

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
