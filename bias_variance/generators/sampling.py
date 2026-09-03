'''Sampling-based data generator for bias-variance studies.

This module defines sampling strategies, configuration, variation, and generator
objects for creating sampled copies of a base dataset.
'''

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
    '''Names of supported sampling strategies.'''

    BOOTSTRAP = 'bootstrap'
    STRATIFIED = 'stratified'
    LHS = 'lhs'


@dataclass(frozen=True)
class SamplingStrategy:
    '''Stores a sampling function and its keyword arguments.

    Attributes
    ----------
    function : SamplingFunction
        Function used to generate a sampled dataset.
    kwargs : Mapping[str, Any]
        Keyword arguments passed to the sampling function.
    '''
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
    '''Configures sampling strategies for dataset generation.

    Attributes
    ----------
    sampling_strategies : Mapping[SamplingStrategyName, SamplingStrategy]
        Sampling strategies used to generate dataset variations.
    '''
    sampling_strategies: Mapping[
        SamplingStrategyName,
        SamplingStrategy,
    ] = field(default_factory=lambda: DEFAULT_SAMPLING_STRATEGIES)

    @property
    def variation_labels(self) -> tuple[str, ...]:
        '''Return labels for the configured sampling strategies.

        Returns
        -------
        tuple[str, ...]
            Labels for each configured sampling strategy.
        '''
        return tuple(
            name.value
            for name
            in self.sampling_strategies
        )


@dataclass(frozen=True, slots=True)
class SamplingVariation(Variation[pd.DataFrame]):
    '''Represents a sampled dataset variation.

    Attributes
    ----------
    dataset : pd.DataFrame
        The generated sampled dataset.
    '''

    @property
    def dataset(self) -> pd.DataFrame:
        '''The generated dataset, kept as a compatibility alias.'''
        return self.generated


class SamplingGenerator(VariationGenerator[pd.DataFrame]):
    '''Generate sampled copies of a base dataset.

    Each configured sampling strategy creates one sampled dataset variation.

    Attributes
    ----------
    settings : SamplingGeneratorConfig | None, default = None
        Settings that define which sampling strategies are used.
    base_dataset : pd.DataFrame | None
        Dataset used as the source for generating sampled variations.
    '''

    def __init__(
        self,
        settings: SamplingGeneratorConfig | None = None,
    ) -> None:
        self.settings = SamplingGeneratorConfig() if settings is None else settings
        self._base_dataset: pd.DataFrame | None = None

    @property
    def variation_labels(self) -> tuple[str, ...]:
        '''Return labels for the configured sampling variations.

        Returns
        -------
        tuple[str, ...]
            Labels from the generator settings.
        '''
        return self.settings.variation_labels

    @property
    def base_dataset(self) -> pd.DataFrame | None:
        '''Return the base dataset used for sampling.

        Returns
        -------
        pd.DataFrame | None
            The base dataset, or ``None`` when one has not been set.
        '''
        return self._base_dataset

    @base_dataset.setter
    def base_dataset(self, value: pd.DataFrame) -> None:
        '''Set and validate the base dataset.

        Parameters
        ----------
        value : pd.DataFrame
            Dataset to use for sampling.
        '''
        self._validate_dataset(value)
        self._base_dataset = value

    @property
    def dataset(self) -> pd.DataFrame:
        '''Return a copy of the base dataset.

        Returns
        -------
        pd.DataFrame
            A copy of the base dataset.

        Raises
        ------
        ValueError
            If the base dataset has not been set.
        '''
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
        '''Generate sampled dataset variations.

        Parameters
        ----------
        random_state : int | None, default = None
            Seed used to make sampling reproducible.

        Returns
        -------
        Iterable[SamplingVariation]
            Generated sampled dataset variations.
        '''
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
        
