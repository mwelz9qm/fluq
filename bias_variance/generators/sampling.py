from collections.abc import Callable, Iterable, Mapping
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


@dataclass(frozen=True, slots=True)
class SamplingVariation(Variation[pd.DataFrame]):

    @property
    def dataset(self) -> pd.DataFrame:
        return self.generated


class SamplingGenerator(VariationGenerator[pd.DataFrame]):

    def __init__(
        self,
        settings: SamplingGeneratorConfig | None = None,
    ) -> None:
        self.settings = SamplingGeneratorConfig() if settings is None else settings
        self._base_dataset: pd.DataFrame | None = None

    @property
    def variation_labels(self) -> tuple[str, ...]:
        return self.settings.variation_labels

    @property
    def base_dataset(self) -> pd.DataFrame | None:
        return self._base_dataset

    @base_dataset.setter
    def base_dataset(self, value: pd.DataFrame) -> None:
        self._validate_dataset(value)
        self._base_dataset = value

    @property
    def dataset(self) -> pd.DataFrame:
        if self._base_dataset is None:
            raise ValueError(
                'Base dataset is not set.'
            )
        return self._base_dataset.copy()

    def generate(
        self,
        *,
        random_state: int | None = None,
    ) -> Iterable[SamplingVariation]:
        for label, strategy in self.settings.sampling_strategies.items():
            variation = SamplingVariation(
                label=label.value,
                random_state=random_state,
                generated=strategy.function(
                    self.dataset,
                    random_state=random_state,
                    **strategy.kwargs,
                ),
            )

            yield variation
        
