import pandas as pd
from collections.abc import Callable
from typing  import Generic, ParamSpec, TypeVar

# Define type variables for argument signatures and return types
P = ParamSpec('P')
R = TypeVar('R', bound=pd.DataFrame)

class Sampler(Generic[P, R]):
    '''
    Generates datasets given a list of sampling strategies.

    Parameters
    ------------------
    strategies: list[tuple[str, Callable[P, R], tuple, dict]]
        The list of sampling stategies used to generate a new dataset.
    '''
    def __init__(self):
        self.strategies: list[tuple[str, Callable[P, R], tuple, dict]] = []

    def __init__(self, strategies: list[tuple[str, Callable[P, R], tuple, dict]]):
        self.strategies = strategies
    
    def add_strategy(self, label: str, sampling_func: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> 'Sampler':
        '''
        Adds a sampling function for the Sampler to apply and generate new datasets.
        '''
        self.strategies.append((label, sampling_func, args, kwargs))
        return self
    
    def remove_strategy(self, label: str) -> 'Sampler':
        '''
        Removes a sampling function from the Sampler.
        '''
        self.strategies = [strategy for strategy in self.strategies if label not in strategy]
        return self
    
    def generate(self, df: pd.DataFrame) ->  dict[str, pd.DataFrame]:
        '''
        Generates datasets with the provided strategies. Set strategies via the constructor or
        with the add_strategy() method.
        '''
        sample_sets = {}
        for label, sampling_func, args, kwargs in self.strategies:
            sample_sets[label] = sampling_func(df.copy(), *args, **kwargs)
        return sample_sets