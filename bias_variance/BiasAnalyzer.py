# General workflow
# analyzer = BiasAnalyzer(...)
# analyzer.run_model_bias_study(...)
# analyzer.run_sampling_bias_study(...)
# analyzer.run_data_bias_study(...)
# analyzer.decompose_variance(...)
# analyzer.plot_disagreement_map(...)

class BiasAnalyzer:
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
    '''

    def __init__(
            self, 
            base_architecture:callable, 
            dataset, 
            test_size:float = 0.2,
            random_state:int | None = None,
            save_predictions:bool = True
            ) -> None:
        pass

    def run_model_bias_study(
            self, 
            architecture_types:list[str]=['wide','narrow','taper','reverse_taper','combined_taper'], 
            n_architectures=10,
            metrics:list[str]=['rmse','r2','mse','mae']
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
            strategies:list[str]=['bootstrap','lhs','stratified'], 
            n_samples:int=10,
            sample_fraction:float = 1.0,
            replacement:bool = True
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

        TODO (implementaion):
        ------------
        - Create each train dataset w/ helper functions for sampling methods.
        - Loop through each sampled_dataset in dataset.
            - Train and evaluate model using base_model (helper function call).
        '''
        pass

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

    def _generate_bootstrap_dataset(self, dataset, sampling_settings:dict | None = None):
        '''
        Purpose
        -------------
        To generate and return a bootstrapped train dataset. 

        Parameters
        -------------
        dataset : pandas.DataFrame
            Base dataset to sample from

        sampling_settings : dict | None, default = None
            Additional configurations for sampling method
        
        Returns
        -------------
        pandas.DataFrame
            The generated dataset
        '''
        pass

    def _generate_stratified_dataset(self, dataset, sampling_settings:dict | None = None):
        '''
        Purpose
        -------------
        To generate and return a stratified train dataset.

        Parameters
        -------------
        dataset : pandas.DataFrame
            Base dataset to sample from

        sampling_settings : dict | None, default = None
            Additional configurations for sampling method
        
        Returns
        -------------
        pandas.DataFrame
            The generated dataset
        '''
        pass

    def _generate_lhs_dataset(self, dataset, sampling_settings:dict | None = None):
        '''
        Purpose
        -------------
        To generate and return a latin hypercube sampled train dataset. 

        Parameters
        -------------
        dataset : pandas.DataFrame
            Base dataset to sample from
            
        sampling_settings : dict | None, default = None
            Additional configurations for sampling method
        
        Returns
        -------------
        pandas.DataFrame
            The generated dataset
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