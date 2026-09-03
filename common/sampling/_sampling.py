import numpy as np
import pandas as pd


def get_random_samples(
        dataset: pd.DataFrame,
        *,
        n_samples: int | None = None,
        sample_fraction: float | None = None,
        random_state: int | None = None,
        with_replacement: bool = False,
    ) -> pd.DataFrame:
    '''
    To sample and return n random samples from a given dataset.

    Parameters
    -------------
    dataset : pandas.DataFrame
        Base dataset to sample from.

    n_samples : int | None = None
        Number of samples to return. Is mutually exclusive with sample_fraction.
    
    sample_fraction : float | None = None
        Size of sample set to return as a fraction of dataset. Is mutually exclusive with n_samples.

    random_state : int | None = None
        Specify seed for reproducibility.

    with_replacement : bool
        Determines if samples include repeated values.
    
    Returns
    -------------
    pandas.DataFrame
        The sampled dataset.
    '''
    # Error conditions
    if (n_samples is None) and (sample_fraction is None):
        raise TypeError('You must provide exactly one of \'n_samples\' or \'sample_fraction\'.')
        
    if (n_samples is not None) and (sample_fraction is not None):
        raise TypeError('You must provide exactly one of \'n_samples\' or \'sample_fraction\'.')

    if (n_samples is not None) and (n_samples < 1):
        raise ValueError('The number of samples \'n_samples\' must be a positive interger.')

    if (n_samples is not None) and (not with_replacement) and (n_samples > dataset.shape[0]):
        raise ValueError('The number of samples \'n_samples\' cannot exceed the size of the dataset. Consider setting \'with_replacement\'=True for getting a number of samples greater than the size of the dataset.')
    
    if (sample_fraction is not None) and (sample_fraction <= 0):
        raise ValueError('The sample fraction \'sample_fraction\' must be greater than 0.')
    
    if (sample_fraction is not None) and (not with_replacement) and (sample_fraction > 1):
        raise ValueError('The sample fraction \'sample_fraction\' cannot exceed greater than 1. Consider setting \'with_replacement\'=True for sample fractions greater than 1.')

    # check if random_state is set, then sample with random_state if given
    if n_samples is None:
        return dataset.sample(frac=sample_fraction, replace=with_replacement, random_state=random_state)
    elif sample_fraction is None:
        return dataset.sample(n=n_samples, replace=with_replacement, random_state=random_state)

def get_stratified_random_samples(
        dataset: pd.DataFrame,
        *,
        stratified_column_name: str,
        n_samples: int | None = None,
        sample_fraction: float | None = None,
        random_state: int|None = None,
        with_replacement: bool = False,
        is_print_statements_shown: bool = False
    ) -> pd.DataFrame:
    '''
    To sample and return n random samples from a given dataset. Samples are chosen based on a stratified column name
    within the dataset. The stratified column partitions the dataset into stratas, based on the distinct categories
    in the column, and is sampled evenly.

    Parameters
    -------------
    dataset : pandas.DataFrame
        Base dataset to sample from.

    stratified_column_name : str
        Specified column to stratify the dataset.

    n_samples : int | None = None
        Number of samples to return. Is mutually exclusive with sample_fraction.
    
    sample_fraction : float | None = None
        Size of sample set to return as a fraction of dataset. Is mutually exclusive with n_samples.

    random_state : int | None = None
        Specify seed for reproducibility.

    with_replacement : bool
        Determines if samples include repeated values.

    is_print_statements_shown : bool = False
        Prints the number of stratas and samples per strata. Temporary parameter for testing.
    
    Returns
    -------------
    pandas.DataFrame
        The sampled dataset.
    '''
    # Error conditions
    if (n_samples is None) and (sample_fraction is None):
        raise TypeError('You must provide exactly one of \'n_samples\' or \'sample_fraction\'.')
        
    if (n_samples is not None) and (sample_fraction is not None):
        raise TypeError('You must provide exactly one of \'n_samples\' or \'sample_fraction\'.')

    if (n_samples is not None) and (n_samples < 1):
        raise ValueError('The number of samples \'n_samples\' must be a positive interger.')

    if (n_samples is not None) and (not with_replacement) and (n_samples > dataset.shape[0]):
        raise ValueError('The number of samples \'n_samples\' cannot exceed the size of the dataset. Consider setting \'with_replacement\'=True for getting a number of samples greater than the size of the dataset.')
    
    if (sample_fraction is not None) and (sample_fraction <= 0):
        raise ValueError('The sample fraction \'sample_fraction\' must be greater than 0.')
    
    if (sample_fraction is not None) and (not with_replacement) and (sample_fraction > 1):
        raise ValueError('The sample fraction \'sample_fraction\' cannot exceed greater than 1. Consider setting \'with_replacement\'=True for sample fractions greater than 1.')

    # create the grouped dataframe and determine the number of bins/stratas
    grouped_df = dataset.groupby(stratified_column_name)
    n = 1 # To hold the number of samples
    if sample_fraction is None:
        n = n_samples
    elif n_samples is None:
        n = int(np.round(dataset.shape[0] * sample_fraction))
    n_strata_samples = int(np.ceil( n / grouped_df.ngroups))
    
    if is_print_statements_shown:
        print('number of stratas:', grouped_df.ngroups)
        print('number of strata samples:', n_strata_samples)

    # check if random_state is set, then get samples per strata
    samples_df = grouped_df.sample(n=n_strata_samples, replace=with_replacement, random_state=random_state)
    
    grouped_samples_df = samples_df.groupby(stratified_column_name) # Regroup for trimming

    # create rng object
    rng = np.random.default_rng()
    if random_state is not None:
        rng = np.random.default_rng(seed=random_state)
    
    # iterate through groups and track removable samples until the sample size equals n
    drop_indicies = []
    for i, (_, group_df) in enumerate(grouped_samples_df):
        if samples_df.shape[0] - i == n:
            break
        random_index = rng.choice(group_df.index)
        drop_indicies.append(random_index)
    
    # remove the rows (trimming sample set) and return samples
    return samples_df.drop(drop_indicies)


def get_quantile_stratified_random_samples(
    dataset: pd.DataFrame,
    *,
    stratify_col_index: int | None = None,
    stratify_col_name: str | None = None,
    n_bins: int = 4,
    n_samples: int | None = None,
    sample_fraction: float | None = None,
    random_state: int | None = None,
    with_replacement: bool = False,
) -> pd.DataFrame:
    '''
    To sample and return n random samples from a given dataset. Samples are chosen based on a stratified column name or
    index within the dataset. The stratified column partitions the dataset into stratas, based on quantile rankings.

    Parameters
    -------------
    dataset : pandas.DataFrame
        Base dataset to sample from.

    stratify_col_index: int | None = None
        Specified column to stratify the dataset. The column must contain numerical values. You must
        provide exactly one of stratify_col_index or stratify_col_name.

    stratify_col_name : str | None = None
        Specified column to stratify the dataset. The column must contain numerical values. You must
        provide exactly one of stratify_col_index or stratify_col_name.

    n_bins : int | None = 4
        Number of quantile ranks applied on the stratify column data.

    n_samples : int | None = None
        Number of samples to return. Is mutually exclusive with sample_fraction.
    
    sample_fraction : float | None = None
        Size of sample set to return as a fraction of dataset. Is mutually exclusive with n_samples.

    random_state : int | None = None
        Specify seed for reproducibility.

    with_replacement : bool
        Determines if samples include repeated values.
    
    Returns
    -------------
    pandas.DataFrame
        The sampled dataset.
    '''
    # Error conditions
    if (stratify_col_index is None) == (stratify_col_name is None):
        raise ValueError(
            'Provide exactly one of stratify_col_index or stratify_col_name.'
        )

    stratified_df = dataset.copy()
    quantile_col_name = '__quantile_strata__'

    if quantile_col_name in stratified_df.columns:
        raise ValueError(f'Temporary column name {quantile_col_name!r} already exists.')

    if stratify_col_index is not None:
        stratify_values = stratified_df.iloc[:, stratify_col_index]
    else:
        stratify_values = stratified_df[stratify_col_name]

    # Create new column based on quantile ranking.
    stratified_df[quantile_col_name] = pd.qcut(
        stratify_values,
        q=n_bins,
        labels=False,
        duplicates='drop', # Safeguard in case bin edges happen to be equal since qcut() enforces unique bin edges.
    )

    # Get samples
    samples_df = get_stratified_random_samples(
        stratified_df,
        stratified_column_name=quantile_col_name,
        n_samples=n_samples,
        sample_fraction=sample_fraction,
        random_state=random_state,
        with_replacement=with_replacement,
    )

    return samples_df.drop(columns=quantile_col_name) # Remove quantile column after sampling.
    
def generate_latin_hypercube_samples(
        regressor_dataset: pd.DataFrame,
        *,
        n_samples: int | None = None,
        sample_fraction: float | None = None,
        random_state: int | None = None,
        is_print_statements_shown: bool = False
    ) -> pd.DataFrame:
    '''
    To generate and return n latin hypercube samples from a given dataset.

    Parameters
    -------------
    regressor_dataset : pandas.DataFrame
        Base dataset to sample from. Note that the dataset must contain regressor data since
        this sampling method stratifies the data based on quantile values.
        
    n_samples : int | None = None
        Number of samples to return. Is mutually exclusive with sample_fraction.
    
    sample_fraction : float | None = None
        Size of sample set to return as a fraction of dataset. Is mutually exclusive with n_samples.

    random_state : int | None = None
        Specify seed for reproducibility.
    
    is_print_statements_shown : bool = False
        Prints the number of stratas and samples per strata. Temporary parameter for testing.
    
    Returns
    -------------
    pandas.DataFrame
        The generated dataset.
    '''
    # Error conditions
    if (n_samples is None) and (sample_fraction is None):
        raise TypeError('You must provide exactly one of \'n_samples\' or \'sample_fraction\'.')
        
    if (n_samples is not None) and (sample_fraction is not None):
        raise TypeError('You must provide exactly one of \'n_samples\' or \'sample_fraction\'.')

    if (n_samples is not None) and ((n_samples < 1) or (n_samples > regressor_dataset.shape[0])):
        raise ValueError('The number of samples \'n_samples\' must be a positive interger and cannot exceed the size of the dataset.')
    
    if (sample_fraction is not None) and ((sample_fraction <= 0) or (sample_fraction > 1)):
        raise ValueError('The sample fraction \'sample_fraction\' must be greater than 0 and less than or equal to 1.')
    
    n = 1 # To hold the number of samples
    if sample_fraction is None:
        n = n_samples
    elif n_samples is None:
        n = int(np.round(regressor_dataset.shape[0] * sample_fraction))

    # Get quantile steps to build strata intervals
    quantile_steps = np.linspace(0, 1, num=n+1)

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
        col_samples = np.zeros(n)
        for i in np.arange(1, quantile_matrix.shape[0]):
            lower = quantile_matrix[i-1,j]
            upper = quantile_matrix[i,j]
            sample = rng.uniform(lower, upper)
            col_samples[i-1] = sample
        rng.shuffle(col_samples)
        df_samples[regressor_dataset.columns[j]] = col_samples
    
    return df_samples
