import pandas as pd
import numpy as np

def get_random_samples(
        dataset:pd.DataFrame,
        n_samples:int,
        random_state:int|None = None,
        with_replacement:bool=False,
    ) -> pd.DataFrame:
    '''
    Purpose
    -------------
    To sample and return n random samples from a given dataset.

    Parameters
    -------------
    dataset : pandas.DataFrame
        Base dataset to sample from.

    n_samples : int
        Number of samples to return.

    random_state : int | None = None
        To optionally specify a random state for reproducibility.

    with_replacement : bool
        To specify if sampling should include repeated values.
    
    Returns
    -------------
    pandas.DataFrame
        The sampled dataset.
    '''
    # Error conditions
    if (n_samples < 1):
        raise ValueError('The number of samples \'n_samples\' must be a positive interger.')

    if not with_replacement and (n_samples > dataset.shape[0]):
        raise ValueError('The number of samples \'n_samples\' exceeds the size of the dataset. Consider setting \'with_replacement\'=True for sampling with replacement')

    # check if random_state is set, then sample with random_state if given
    if random_state is None:
        return dataset.sample(n=n_samples, replace=with_replacement)
    else:
        return dataset.sample(n=n_samples, replace=with_replacement, random_state=random_state)

def get_stratified_random_samples(
        dataset:pd.DataFrame,
        n_samples:int,
        stratified_column_name:str,
        random_state:int|None = None,
        with_replacement:bool=False,
        is_print_statements_shown:bool=False
    ) -> pd.DataFrame:
    '''
    Purpose
    -------------
    To sample and return n random samples from a given dataset. Samples are chosen based on a stratified column name
    within the dataset. The stratified column partitions the dataset into stratas, based on the distinct categories
    in the column, and is sampled evenly.

    Parameters
    -------------
    dataset : pandas.DataFrame
        Base dataset to sample from.

    n_strata_samples : int
        Number of samples per strata to return.

    stratified_column_name : str
        The stratified column name used to bin the data points for sampling.

    random_state : int | None = None
        To optionally specify a random state for reproducibility.
        
    with_replacement : bool = False
        To specify if sampling should include repeated values.

    is_print_statements_shown : bool = False
        Temporary parameter for testing.
    
    Returns
    -------------
    pandas.DataFrame
        The sampled dataset.
    '''
    # Error conditions
    if (n_samples < 1):
        raise ValueError('The number of samples \'n_samples\' must be a positive interger.')

    if not with_replacement and (n_samples > dataset.shape[0]):
        raise ValueError('The number of samples \'n_samples\' exceeds the size of the dataset. Consider setting \'with_replacement\'=True for sampling with replacement')

    # create the grouped dataframe and determine the number of bins/stratas
    grouped_df = dataset.groupby(stratified_column_name)
    n_strata_samples = int(np.ceil(n_samples / grouped_df.ngroups))
    if is_print_statements_shown:
        print('number of stratas:', grouped_df.ngroups)
        print('number of strata samples:', n_strata_samples)

    samples_df = pd.DataFrame() # To hold the samples

    # check if random_state is set, then get samples per strata
    if random_state is None:
        samples_df = grouped_df.sample(n=n_strata_samples, replace=with_replacement)
    else:
        samples_df = grouped_df.sample(n=n_strata_samples, replace=with_replacement, random_state=random_state)
    
    grouped_samples_df = samples_df.groupby(stratified_column_name) # Regroup for trimming

    # create rng object
    rng = np.random.default_rng()
    if random_state is not None:
        rng = np.random.default_rng(seed=random_state)
    
    # iterate through groups and track removable samples until
    # the sample size equals n_samples
    drop_indicies = []
    counter = 0
    for _, group_df in grouped_samples_df:
        if samples_df.shape[0] - counter == n_samples:
            break
        random_index = rng.choice(group_df.index)
        drop_indicies.append(random_index)
        counter += 1
    
    # remove the rows (trimming sample set) and return samples
    return samples_df.drop(drop_indicies)
    
def generate_latin_hypercube_samples(
        regressor_dataset:pd.DataFrame,
        n_samples:int,
        random_state:int|None = None,
        is_print_statements_shown:bool=False
    ) -> pd.DataFrame:
    '''
    Purpose
    -------------
    To generate and return n latin hypercube samples from a given dataset.

    Parameters
    -------------
    regressor_dataset : pandas.DataFrame
        Base dataset to sample from. Note that the dataset must contain regressor data since
        this sampling method stratifies the data based on quantile values.
        
    n_samples : int
        Number of samples to return.
    
    random_state : int | None = None
        To optionally specify a random state for reproducibility.
    
    is_print_statements_shown : bool = False
        Temporary parameter for testing.
    
    Returns
    -------------
    pandas.DataFrame
        The generated dataset.
    '''
    # Error conditions
    if (n_samples < 1):
        raise ValueError('The number of samples \'n_samples\' must be a positive interger.')
    
    if (n_samples > regressor_dataset.shape[0]):
        raise ValueError('The number of samples \'n_samples\' must be less than the size of the regressor dataset.')

    # Get quantile steps to build strata intervals
    quantile_steps = np.linspace(0, 1, num=n_samples+1)

    # Get quantile values per column.
    df = regressor_dataset.quantile(quantile_steps, interpolation='midpoint')

    # Convert df to 2D matrix
    quantile_matrix = df.to_numpy()

    # Show quantile_matrix (for testing only)
    if is_print_statements_shown:
        print('Showing quantile_matrix')
        print('-'*30)
        print(quantile_matrix)
        print('quantile_matrix shape:', quantile_matrix.shape)

    # Create rng object for shuffling samples
    rng =  np.random.default_rng()
    if random_state is not None:
        rng = np.random.default_rng(seed=random_state)

    # Loop through quantile matrix to sample from the intervals and build df samples.
    df_samples = pd.DataFrame(columns=regressor_dataset.columns)
    for j in np.arange(quantile_matrix.shape[1]):
        col_samples = np.zeros(n_samples)
        for i in np.arange(1, quantile_matrix.shape[0]):
            lower = quantile_matrix[i-1,j]
            upper = quantile_matrix[i,j]
            sample = rng.uniform(lower, upper)
            col_samples[i-1] = sample
        rng.shuffle(col_samples)
        df_samples[regressor_dataset.columns[j]] = col_samples
    
    return df_samples
