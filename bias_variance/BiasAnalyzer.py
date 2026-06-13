import pandas as pd
import numpy as np
from common.sampling._sampling import get_random_samples, get_stratified_random_samples, generate_latin_hypercube_samples
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.metrics import R2Score, MeanSquaredError, RootMeanSquaredError, MeanAbsoluteError
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error, mean_absolute_error

class BiasAnalyzerConfigMeta(type):
    @property
    def METRIC_OPTIONS(cls):
        return {
            'rmse' : RootMeanSquaredError(name='rmse'),
            'mse' : MeanSquaredError(name='mse'),
            'mae' : MeanAbsoluteError(name='mae'),
            'r2' : R2Score(name='r2')
        }

# General workflow
# analyzer = BiasAnalyzer(...)
# analyzer.run_model_bias_study(...)
# analyzer.run_sampling_bias_study(...)
# analyzer.run_data_bias_study(...)
# analyzer.decompose_variance(...)
# analyzer.plot_disagreement_map(...)

class BiasAnalyzer(metaclass=BiasAnalyzerConfigMeta):
    '''
    Purpose
    ------------
    To analyze each bias by comparing to the base model and dataset to
    the generated predictions' 95% confidence interval.

    Parameters
    ------------
    base_architecture : callable | keras.Model
        Base model architecture or model factory.
    
    dataset : pandas.DataFrame

    test_size : float, default=0.2
        Fraction used for initial train/test split.

    random_state : int | None, default=None
        Seed used for reproducibility.

    save_predicitons : bool, default=True
        Determines if raw predictions are stored internally.

    Attributes
    ------------
    results : pandas.DataFrame
        Stores summarized study outputs.

    predictions : pandas.DataFrame
        Stores raw prediction arrays.

    Questions
    ------------
    - Should we pass the entire dataset and use a random seed for the base train/test split when
    comparing the sampling and data biases? Or, does it make more sense to pass the base train
    /test datasets for comparisons? - I think we should pass the entire dataset to prevent leakage,
    centralize randomness, and guarantee fair comparisons. Once passed, the train/test splits 
    can be reproducibly generated and the split indices can be stored.

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

    - Should we have dataset contain inputs and outputs or should each be separate variables?
    
    - Should the BiasAnalyzer hold a compiled base model?
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
            is_predictions_saved: bool = False,
            _model: tf.keras.Model | None = None,
            _init_weights: list[np.ndarray] | None = None,
            _results_df: pd.DataFrame | None = None,
            _predictions_df: pd.DataFrame | None = None,
            ) -> None:
        self.inputs_df = inputs_df
        self.outputs_df = outputs_df
        self.model_settings = model_settings
        self.test_size = test_size
        self.random_state = random_state
        self.is_predictions_saved = is_predictions_saved
        self._model = _model
        self._init_weights = _init_weights
        self._results_df = _results_df
        self._predictions_df = _predictions_df


    def _init_model(self):
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

    def _train_and_evaluate_model(self, X_train, y_train, X_test, y_test) -> dict:
        # reset weights
        self._model.set_weights(self._init_weights)
        # train
        self._model.fit(
            X_train,
            y_train,
            epochs=self.model_settings['epochs'],
            verbose=self.model_settings['verbose']
        )
        # evaluate
        results = self._model.evaluate(X_test, y_test, batch_size=self.model_settings['batch_size'], return_dict=True)
        
        return results


    def run_model_bias_study(
            self, 
            architecture_types:list[str]=['wide','narrow','taper','reverse_taper','combined_taper'], 
            n_architectures=10,
            metrics:list[str]=['root_mean_squared_error','r2','mse','mae']
            ) -> None:
        '''
        Purpose
        -----------
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
            n_samples: int | None = None,
            sample_fraction: float | None = None,
            n_iter_per_strategy: int = 10,
            stratified_col_name: str | None = None,
            with_replacement: bool = False
            ) -> None:
        '''
        Purpose
        -----------
        To create multiple train dataset samples based on strategies (i.e., bootstrap, lhs, and
        stratified), train on base model with all generated train datasets, and make predictions
        with base test dataset per train dataset for comparison results.

        Parameters
        ------------
        strategies : list[str], default = ['bootstrap','lhs','stratified'] i.e. all strategies
            Must contain at least one strategy to run.
        
        n_samples : int, default = 10
            The number of sampled train datasets generated per strategy. Must be greater than 0.

        sample_fraction : float, default = 1.0
            Control variable of sample size, used for bootstrap/LHS sampling.

        replacement : bool, default = True
            Control variable for bootstrap sampling.

        Returns
        ------------
        None

        TODO tasks
        ------------
        - Handle random_state when it is not None.
        - Refactor stratified method to allow for each method/strategy to be in helper function.
        - Find best implementation for handling stratified column in dataframe.

        Questions
        -----------
        - Should the dataframe contain the stratified column or should the get_stratified_random_samples
        function add and remove the stratified column in the implementation?
        - Should we have a results_df for each bias? (i.e., data_results_df, sampling_results_df, model_results_df)
        '''
        # Error checking
        if (n_samples is None) and (sample_fraction is None):
            raise TypeError('You must provide exactly one of \'n_samples\' or \'sample_fraction\'.')
        
        if (stratified_col_name is None) and ('stratified' in strategies):
            raise TypeError('You must provide \'stratified_col_name\' if \'strategies\' contains \'stratified\'.')

        # before running study, initialize self._model based on self.model_settings
        self._init_model()

        results_df = pd.DataFrame() # To hold results
        dataset_df = pd.concat([self.inputs_df, self.outputs_df], axis=1) # merge inputs and outputs for sampling
        
        # Loop through sampling options and perform sampling on dataset, training on sampled set, and gathering results.
        for strategy in strategies:
            if strategy.lower() == 'bootstrap':
                # Loop through the amount of iterators/data points for results_df.
                for _ in np.arange(n_iter_per_strategy):
                    # apply sampling method
                    bootstrap_df = get_random_samples(dataset_df, n_samples=n_samples, with_replacement=with_replacement)
                    # split sampled dataset into input and output datasets
                    bootstrap_inputs_df = bootstrap_df[self.inputs_df.columns]
                    bootstrap_outputs_df = bootstrap_df[self.outputs_df.columns]
                    # split inputs and outputs for training and testing
                    X_train, X_test, y_train, y_test = train_test_split(bootstrap_inputs_df, bootstrap_outputs_df, test_size=self.test_size)
                    # build results content for results_df
                    results_row = {
                        'study': 'sampling',
                        'sampling_method': 'bootstrap',
                    } | self._train_and_evaluate_model(X_train, y_train, X_test, y_test)
                    results_df = pd.concat([results_df, pd.DataFrame([results_row])])

            # Same structure applied as in 'bootstrap' method
            if strategy.lower() == 'lhs':
                for _ in np.arange(n_iter_per_strategy):
                    lhs_df = generate_latin_hypercube_samples(dataset_df, n_samples=n_samples)
                    lhs_inputs_df = lhs_df[self.inputs_df.columns]
                    lhs_outputs_df = lhs_df[self.outputs_df.columns]
                    X_train, X_test, y_train, y_test = train_test_split(lhs_inputs_df, lhs_outputs_df, test_size=self.test_size)
                    results_row = {
                        'study': 'sampling',
                        'sampling_method': 'lhs',
                    } | self._train_and_evaluate_model(X_train, y_train, X_test, y_test)
                    results_df = pd.concat([results_df, pd.DataFrame([results_row])])
            
            # NOTE: stratified method needs to be refactored for handling stratified_column_name
            if strategy.lower() == 'stratified':
                for _ in np.arange(n_iter_per_strategy):
                    stratified_df = get_stratified_random_samples(dataset_df, stratified_column_name=stratified_col_name, n_samples=n_samples, with_replacement=with_replacement)
                    stratified_inputs_df = stratified_df[self.inputs_df.columns]
                    stratified_outputs_df = stratified_df[self.outputs_df.columns]
                    X_train, X_test, y_train, y_test = train_test_split(stratified_inputs_df, stratified_outputs_df, test_size=self.test_size)
                    results_df = pd.concat([results_df, self._get_sample_run_results_row(X_train, X_test, y_train, y_test, self.model, self.model_settings, 'stratified')])
            
        self.results_df = results_df # copy results
        

    def run_data_bias_study(
            self, 
            cv_folds:int = 5, 
            n_iter:int = 10,
            shuffle:bool = True
            ) -> None:
        '''
        Purpose
        -----------
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
            ):
        '''
        Purpose
        -----------
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
    
    def export_results(
            self,
            filepath:str = 'results.csv'
    ) -> None:
        '''
        Purpose
        -----------
        To export the results of previous runs to a CSV file. If no runs were performed,
        the exporter should not create a new file.

        Parameters
        -----------
        filepath : str, default = 'results.csv'
            Filepath of CSV file to be saved.

        Returns
        -----------
        None
        '''
        pass
    
    def plot_disagreement_map(
            self,
            view:list[str] = ['model','sampling','data'],
            plot_type:list[str] = ['heatmap','histogram','KDE','uncertainty_bands','scatter_disagreement'],
            plot_setttings:dict | None = None
            ) -> None:
        '''
        Purpose
        -----------
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