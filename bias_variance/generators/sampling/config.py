from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any

import pandas as pd

from bias_variance.generators.base import VariationGeneratorConfig
from common.sampling._sampling import (
    generate_latin_hypercube_samples,
    get_quantile_stratified_random_samples,
    get_random_samples,
)

type SamplingFunction = Callable[..., pd.DataFrame]


class SamplingStrategyName(StrEnum):
    BOOTSTRAP = 'bootstrap'
    STRATIFIED = 'stratified'
    LHS = 'lhs'

    @property
    def function(self) -> SamplingFunction:
        match self:
            case SamplingStrategyName.BOOTSTRAP:
                return get_random_samples
            case SamplingStrategyName.STRATIFIED:
                return get_quantile_stratified_random_samples
            case SamplingStrategyName.LHS:
                return generate_latin_hypercube_samples


@dataclass(frozen=True, slots=True)
class SamplingStrategy:
    function: SamplingFunction
    kwargs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not callable(self.function):
            raise TypeError('Sampling strategy function must be callable.')
        if not isinstance(self.kwargs, Mapping):
            raise TypeError('Sampling strategy kwargs must be a mapping.')
        if 'random_state' in self.kwargs:
            raise ValueError(
                'Sampling strategy kwargs cannot contain random_state.'
            )
        object.__setattr__(
            self,
            'kwargs',
            MappingProxyType(dict(self.kwargs)),
        )


@dataclass(frozen=True, slots=True)
class SamplingGeneratorConfig(VariationGeneratorConfig):
    sampling_strategies: Mapping[
        SamplingStrategyName,
        SamplingStrategy,
    ]

    def __post_init__(self) -> None:
        if not isinstance(self.sampling_strategies, Mapping):
            raise TypeError('sampling_strategies must be a mapping.')
        if not self.sampling_strategies:
            raise ValueError('At least one sampling strategy must be configured.')

        for name, strategy in self.sampling_strategies.items():
            if not isinstance(name, SamplingStrategyName):
                raise TypeError(
                    'sampling_strategies keys must be SamplingStrategyName '
                    'members.'
                )
            if not isinstance(strategy, SamplingStrategy):
                raise TypeError(
                    f'sampling_strategies[{name.value!r}] must be a '
                    'SamplingStrategy.'
                )
            if strategy.function is not name.function:
                raise ValueError(
                    f'{name.value!r} must use {name.function.__name__}.'
                )

        object.__setattr__(
            self,
            'sampling_strategies',
            MappingProxyType(dict(self.sampling_strategies)),
        )

    @property
    def variation_labels(self) -> tuple[str, ...]:
        return tuple(name.value for name in self.sampling_strategies)
