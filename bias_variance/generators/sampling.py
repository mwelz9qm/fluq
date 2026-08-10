from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any

import pandas as pd

from common.sampling._sampling import (
    generate_latin_hypercube_samples,
    get_quantile_stratified_random_samples,
    get_random_samples,
)

from .base import Generator, GeneratorConfig, Variation

SamplingFunction = Callable[..., pd.DataFrame]

class SamplingStrategyName(StrEnum):
    BOOTSTRAP = 'bootstrap'
    STRATIFIED = 'stratified'
    LHS = 'lhs'


@dataclass(frozen=True)
class SamplingStrategy:
    function: SamplingFunction
    kwargs: Mapping[str, Any] = field(default_factory=dict)


DEFAULT_SAMPLING_STRATEGIES = MappingProxyType({
    SamplingStrategyName.BOOTSTRAP: SamplingStrategy(
        function=get_random_samples,
        kwargs=MappingProxyType({
            'sample_fraction': 1.0,
            'with_replacement': True,
        }),
    ),
    SamplingStrategyName.STRATIFIED: SamplingStrategy(
        function=get_quantile_stratified_random_samples,
        kwargs=MappingProxyType({
            'stratify_col_index': 0,
            'sample_fraction': 1.0,
        }),
    ),
    SamplingStrategyName.LHS: SamplingStrategy(
        function=generate_latin_hypercube_samples,
        kwargs=MappingProxyType({
            'sample_fraction': 1.0,
        }),
    ),
})


@dataclass(frozen=True, slots=True)
class SamplingGeneratorConfig(GeneratorConfig):
    dataset: pd.DataFrame
    sampling_strategies: Mapping[
        SamplingStrategyName,
        SamplingStrategy,
    ] = field(default_factory=lambda: DEFAULT_SAMPLING_STRATEGIES)

    @property
    def variation_labels(self) -> tuple[str]:
        labels = (
            name.value
            for name
            in self.sampling_strategies
        )
        return labels

@dataclass(frozen=True, slots=True)
class SamplingVariation(Variation[pd.DataFrame]):
    @property
    def dataset(self) -> pd.DataFrame:
        """The generated dataset (kept as a compatibility alias)."""
        return self.generated


class SamplingGenerator(Generator[pd.DataFrame]):
    def __init__(
        self,
        settings: SamplingGeneratorConfig,
    ) -> None:
        self._dataset = settings.dataset.copy()
        self._strategies = dict(settings.sampling_strategies)

    def add_strategy(
        self,
        strategy_name: SamplingStrategyName,
        strategy: SamplingStrategy,
    ) -> 'SamplingGenerator':
        if strategy_name in self._strategies:
            raise ValueError(
                f'Duplicate sampling strategy label: {strategy.label!r}'
            )
        
        if 'random_state' in strategy.kwargs:
            raise ValueError(
                'Configure random_state on BiasAnalyzer, not on a strategy.'
            )
        
        self._strategies[strategy_name] = strategy

        return self
    
    def remove_strategy(self, strategy_name: SamplingStrategyName) -> 'SamplingGenerator':
        self._strategies.pop(strategy_name, None)

        return self
    
    def generate(
        self,
        *,
        random_state:  int | None = None,
    ) -> list[Variation[pd.DataFrame]]:
        variations = []

        for label, strategy in self._strategies.items():
            variation = SamplingVariation(
                label=label.value,
                random_state=random_state,
                generated=strategy.function(
                    self._dataset.copy(),
                    random_state=random_state,
                    **strategy.kwargs,
                ),
            )
            variations.append(variation)
        
        return variations
