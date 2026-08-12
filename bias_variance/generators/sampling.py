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

from .base import Variation, VariationGenerator, VariationGeneratorConfig

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
class SamplingGeneratorConfig(VariationGeneratorConfig):
    _base_dataset: pd.DataFrame
    sampling_strategies: Mapping[
        SamplingStrategyName,
        SamplingStrategy,
    ] = field(default_factory=lambda: DEFAULT_SAMPLING_STRATEGIES)

    @property
    def variation_labels(self) -> tuple[str, ...]:
        return tuple(
            name.value
            for name
            in self.sampling_strategies
        )

    @property
    def dataset(self) -> pd.DataFrame:
        return self._base_dataset.copy()

    def __post_init__(self) -> None:
        self._validate_dataset(self._base_dataset)

@dataclass(frozen=True, slots=True)
class SamplingVariation(Variation[pd.DataFrame]):
    @property
    def dataset(self) -> pd.DataFrame:
        """The generated dataset (kept as a compatibility alias)."""
        return self.generated


class SamplingGenerator(VariationGenerator[pd.DataFrame]):
    def __init__(
        self,
        settings: SamplingGeneratorConfig,
    ) -> None:
        self.settings = settings

    @property
    def variation_labels(self):
        return self.settings.variation_labels
    
    def generate(
        self,
        *,
        random_state: int | None = None,
    ) -> list[SamplingVariation]:
        variations = [
            SamplingVariation(
                label=label.value,
                random_state=random_state,
                generated=strategy.function(
                    self.settings.dataset,
                    random_state=random_state,
                    **strategy.kwargs,
                ),
            )
            for label, strategy
            in self.settings.sampling_strategies.items()
        ]
        
        return variations
