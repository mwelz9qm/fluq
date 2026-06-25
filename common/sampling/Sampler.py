import inspect
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
    
    def add_strategy(self, label: str, sampling_func: Callable[P, R], **kwargs: P.kwargs) -> 'Sampler':
        '''
        Adds a sampling function for the Sampler to apply and generate new datasets.
        '''
        self.strategies.append((label, sampling_func, kwargs))
        return self
    
    def remove_strategy(self, label: str) -> 'Sampler':
        '''
        Removes a sampling function from the Sampler.
        '''
        self.strategies = [strategy for strategy in self.strategies if label not in strategy]
        return self
    
    def generate(self, df: pd.DataFrame, injected_kwargs: dict | None = None) ->  dict[str, pd.DataFrame]:
        '''
        Generates datasets with the provided strategies. Set strategies via the constructor or
        with the add_strategy() method.
        '''
        injected_kwargs = injected_kwargs or {}
        sample_sets = {}
        for label, sampling_func, kwargs in self.strategies:
            final_kwargs  = kwargs.copy()
            sig = inspect.signature(sampling_func)
            # Loop through injected_kwargs and add to final_kwargs if not defined
            # to prevent overwriting
            for key, value in injected_kwargs.items():
                if key in sig.parameters and key not in final_kwargs:
                    final_kwargs[key] = value
            sample_sets[label] = sampling_func(df.copy(), **final_kwargs)
        return sample_sets