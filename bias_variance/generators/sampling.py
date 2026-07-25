from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .base import Generator, Variation

SamplingFunction = Callable[..., pd.DataFrame]


@dataclass(frozen=True)
class SamplingStrategy:
    label: str
    function: SamplingFunction
    kwargs: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SamplingVariation(Variation):
    dataset: pd.DataFrame


class SamplingGenerator(Generator[pd.DataFrame]):
    def __init__(
        self,
        dataset: pd.DataFrame,
        strategies: Iterable[SamplingStrategy] = (),
    ) -> None:
        self._dataset = dataset.copy()
        self._strategies: dict[str, SamplingStrategy] = {}

        for strategy in strategies:
            self.add_strategy(strategy)

    def add_strategy(
        self,
        strategy: SamplingStrategy,
    ) -> 'SamplingGenerator':
        if strategy.label in self._strategies:
            raise ValueError(
                f'Duplicate sampling strategy label: {strategy.label!r}'
            )
        
        if 'random_state' in strategy.kwargs:
            raise ValueError(
                'Configure random_state on BiasAnalyzer, not on a strategy.'
            )
        
        self._strategies[strategy.label] = strategy
        return self
    
    def remove_strategy(self, label: str) -> 'SamplingGenerator':
        self._strategies.pop(label, None)
        return self
    
    def generate(
        self,
        *,
        random_state:  int | None = None,
    ) -> list[SamplingVariation]:
        variations = []

        for label, strategy in self._strategies.items():
            variation = SamplingVariation(
                label=label,
                random_state=random_state,
                dataset=strategy.function(
                    self._dataset.copy(),
                    random_state=random_state,
                    **strategy.kwargs,
                ),
            )
            variations.append(variation)
        
        return variations