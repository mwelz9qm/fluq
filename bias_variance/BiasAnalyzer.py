import os
import datetime
import h5py
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense
from keras.metrics import (
    R2Score,
    MeanSquaredError,
    RootMeanSquaredError,
    MeanAbsoluteError
)
from keras.models import Model
from sklearn.model_selection import train_test_split
from common.sampling._sampling import (
    get_random_samples,
    get_stratified_random_samples,
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
    - Should we store the prediction results in the BiasAnalyzer class as pandas dataframes? Or,
    should they be exported as a csv for deeper analysis? - I think internally, the data should be 
    stored as a pandas DataFrame, for the time being, since DataFrames make filtering, grouping, 
    variance analysis, plotting, and statistics much easier to view and implement. It is still a 
    good idea to implement exporting the results to a CSV for saving results later.
    
    - For study runs, should this return the ran BiasAnalyzer object with altered attributes or keep as same object with updated attributes?
    * This will consequently change our workflow i.e.,
        analyzer = BiasAnalyzer(...)
        analyzer_ran_study = analyzer.run_study(...) \\ OR analyzer = analyzer.run_study(...).copy()
        analyzer_ran_study.decompose_variance()
    
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

    def _save_predictions_and_actuals(self, X_test, y_test):
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
        predictions = self._model.predict(X_test)
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
        A dictionary object with the loss and metric values defined in self.model_settings.
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

        # save predictions
        self._save_predictions_and_actuals(X_test, y_test)

        # evaluate
        scores = self._model.evaluate(X_test, y_test, batch_size=self.model_settings['batch_size'], return_dict=True)
        
        return scores


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
        *,
        strategies:list[str]=['bootstrap','lhs','stratified'],
        n_iter: int = 100, 
        n_samples: int | None = None,
        sample_fraction: float | None = None,
        stratify_col_index: int | None = None,
        stratify_col_name: str | None = None,
        n_bins: int = 4,
        with_replacement: bool = False
    ) -> None:
        '''
        Creates multiple, sampled datasets based on strategies (i.e., bootstrap, lhs, and
        stratified), trains on all generated train datasets with base model, and makes predictions
        with generated test dataset per train dataset for analysis.

        Parameters
        ------------
        strategies: list[str] = ['bootstrap','lhs','stratified'] i.e. all strategies
            Must contain at least one strategy to run.
        
        n_iter: int = 100
            Number of iterations per sampling strategy.
        
        n_samples: int | None = None
            The number of sampled train datasets generated per strategy. Must be greater than 0.
            If n_samples is not None, then sample_fraction must be None.

        sample_fraction: float | None = None
            Control variable of sample size, used for bootstrap/LHS sampling. Must be within (0, 1].
            If sample_fraction is not None, then n_samples must be None.
        
        stratify_col_index: int | None = None
            The selected column index to apply the stratified method on. Is mutually exclusive with
            stratify_col_name, and either one must be not None if stratified method is selected.
        
        stratify_col_name: str | None = None
            The selected column name to apply the stratified method on. Is mutually exclusive with
            stratify_col_index, and either one must be to not None if stratified method is selected.
        
        n_bins: int = 4
            The number of stratas used in the stratified method.

        with_replacement: bool = True
            Control variable for bootstrap and stratified sampling.

        Returns
        ------------
        None

        Questions
        -----------
        - Should we have a results_df for each bias? (i.e., data_results_df, sampling_results_df, model_results_df)
        '''
        # Error conditions
        if (n_samples is None) and (sample_fraction is None):
            raise TypeError('You must provide exactly one of \'n_samples\' or \'sample_fraction\'.')
        
        if (n_samples is not None) and (sample_fraction is not None):
            raise TypeError('You must provide exactly one of \'n_samples\' or \'sample_fraction\'.')
        
        if (stratify_col_index is not None) and (stratify_col_name is not None):
            raise TypeError('You must provide one of \'stratify_col_index\' or \'stratify_col_name\', or leave both as None.')
        
        if ((stratify_col_index is not None) or (stratify_col_name is not None)) and ('stratified' in strategies):
            raise TypeError('You must provide \'stratify_col_index\' or \'stratify_col_name\' if \'strategies\' contains \'stratified\'.')

        # before running study, initialize self._model based on self.model_settings
        self._init_model()

        results_df = pd.DataFrame() # To hold results
        dataset_df = pd.concat([self.inputs_df, self.outputs_df], axis=1) # merge inputs and outputs for sampling
        
        # Loop through sampling options and perform sampling on dataset, training on sampled set, and gathering results.
        for strategy in strategies:
            if strategy.lower() == 'bootstrap':
                # Loop through the amount of iterators/data points for results_df.
                for i in np.arange(n_iter):
                    # apply sampling method, NOTE: we add i to random_state to ensure different, but reproducible, sampled datasets.
                    bootstrap_df = get_random_samples(dataset_df, n_samples=n_samples, sample_fraction=sample_fraction, random_state=self.random_state+i, with_replacement=with_replacement)
                    # split sampled dataset into input and output datasets
                    bootstrap_inputs_df = bootstrap_df[self.inputs_df.columns]
                    bootstrap_outputs_df = bootstrap_df[self.outputs_df.columns]
                    # split inputs and outputs for training and testing
                    X_train, X_test, y_train, y_test = train_test_split(bootstrap_inputs_df, bootstrap_outputs_df, test_size=self.test_size, random_state=self.random_state+i) # NOTE: Should random_state vary in train_test_split() at each step?
                    # build results content for results_df
                    results_row = {
                        'study': 'sampling',
                        'sampling_method': 'bootstrap',
                    } | self._train_and_evaluate_model(X_train, y_train, X_test, y_test)
                    results_df = pd.concat([results_df, pd.DataFrame([results_row])], ignore_index=True)

            # Same structure applied as in 'bootstrap' method
            if strategy.lower() == 'lhs':
                for i in np.arange(n_iter):
                    lhs_df = generate_latin_hypercube_samples(dataset_df, n_samples=n_samples, sample_fraction=sample_fraction, random_state=self.random_state+i)
                    lhs_inputs_df = lhs_df[self.inputs_df.columns]
                    lhs_outputs_df = lhs_df[self.outputs_df.columns]
                    X_train, X_test, y_train, y_test = train_test_split(lhs_inputs_df, lhs_outputs_df, test_size=self.test_size, random_state=self.random_state+i)
                    results_row = {
                        'study': 'sampling',
                        'sampling_method': 'lhs',
                    } | self._train_and_evaluate_model(X_train, y_train, X_test, y_test)
                    results_df = pd.concat([results_df, pd.DataFrame([results_row])], ignore_index=True)
            
            if strategy.lower() == 'stratified':
                # create the stratified column based on quantile rank on the selected index
                stratified_col_name = ''
                if stratify_col_index is not None:
                    stratified_col_name = f'quantile_rank_on_col_{stratify_col_index}'
                    dataset_df[stratified_col_name] = pd.qcut(dataset_df.iloc[:,stratify_col_index], q=n_bins, labels=False) # add column to dataset df
                if stratify_col_name is not None:
                    stratified_col_name = f'quantile_rank_on_{stratify_col_name}'
                    dataset_df[stratified_col_name] = pd.qcut(dataset_df[stratify_col_name], q=n_bins, labels=False)
                # apply same steps as in 'bootstrap' and 'lhs' methods
                for i in np.arange(n_iter):
                    stratified_df = get_stratified_random_samples(dataset_df, stratified_column_name=stratified_col_name, n_samples=n_samples, sample_fraction=sample_fraction, random_state=self.random_state+i, with_replacement=with_replacement)
                    stratified_inputs_df = stratified_df[self.inputs_df.columns]
                    stratified_outputs_df = stratified_df[self.outputs_df.columns]
                    X_train, X_test, y_train, y_test = train_test_split(stratified_inputs_df, stratified_outputs_df, test_size=self.test_size, random_state=self.random_state+i)
                    results_row = {
                        'study': 'sampling',
                        'sampling_method': 'stratified',
                    } | self._train_and_evaluate_model(X_train, y_train, X_test, y_test)
                    results_df = pd.concat([results_df, pd.DataFrame([results_row])], ignore_index=True)
                # remove stratified column after getting results
                dataset_df.drop(columns=stratified_col_name)
            
        self.results_df.to_csv(BiasAnalyzer.RESULTS_FILENAME, index=False)
        self.results_df = results_df # copy results
        

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
        view:list[str]=['model','sampling','data']
    ) -> pd.DataFrame:
        '''
        To provide a breakdown of bias variance of previous runs. If no runs were performed,
        the analyzer should not provide any results.

        Parameters
        -------------
        view : list[str], default = ['model','sampling','data']
            The selection of variances to view.

        Returns
        -------------
        pandas.DataFrame
            Summary results from each study.
        '''
        pass
    
    def plot_disagreement_map(
        self,
        view:list[str] = ['model','sampling','data'],
        plot_type:list[str] = ['heatmap','histogram','KDE','uncertainty_bands','scatter_disagreement'],
        plot_setttings:dict | None = None
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
        pass