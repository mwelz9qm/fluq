'''Noise-based data generator for bias-variance studies.

This module defines configuration, variation, and generator objects for
creating noisy copies of a base dataset.
'''

from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Real

import numpy as np
import pandas as pd

from .base import Variation, VariationGenerator, VariationGeneratorConfig


@dataclass(frozen=True, slots=True)
class NoiseGeneratorConfig(VariationGeneratorConfig):
    '''Configures standard deviations for noise generation.

    Attributes
    ----------
    standard_deviations : tuple[float, ...], default = (0.1, 0.2, 0.3, 0.4, 0.5)
        Standard deviation values used to create noisy dataset variations.
    '''
    standard_deviations: tuple[float, ...] = (
        0.1, 0.2, 0.3, 0.4, 0.5
    )

    @property
    def variation_labels(self) -> tuple[str, ...]:
        '''Return labels for each configured standard deviation.

        Returns
        -------
        tuple[str, ...]
            Labels in the form ``std_<standard_deviation>``.
        '''
        return tuple(
            f"std_{standard_deviation:g}"
            for standard_deviation
            in self.standard_deviations
        )

    @staticmethod
    def _validate_standard_deviations(values: tuple[float, ...]) -> None:
        '''Validate configured standard deviation values.

        Parameters
        ----------
        values : tuple[float, ...]
            Standard deviation values to validate.

        Raises
        ------
        TypeError
            If ``values`` is not a tuple or contains non-real values.
        ValueError
            If ``values`` is empty, non-finite, non-positive, or contains
            duplicates.
        '''
        if not isinstance(values, tuple):
            raise TypeError("Standard deviations must be provided as a tuple.")

        if any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, Real)
            for value in values
        ):
            raise TypeError("Standard deviations must contain only real numbers.")

        standard_deviations = tuple(float(value) for value in values)

        if not standard_deviations:
            raise ValueError(
                "At least one standard deviation must be configured."
            )

        if any(not np.isfinite(value) for value in standard_deviations):
            raise ValueError(
                "Standard deviations must be finite."
            )

        if any(value <= 0 for value in standard_deviations):
            raise ValueError(
                "Standard deviations must be greater than zero."
            )

        if len(set(standard_deviations)) != len(standard_deviations):
            raise ValueError(
                "Standard deviations must be unique."
            )

    def __post_init__(self) -> None:
        '''Validate the noise generator configuration.'''
        self._validate_standard_deviations(self.standard_deviations)


@dataclass(frozen=True, slots=True)
class NoiseVariation(Variation[pd.DataFrame]):
    '''Represents a noisy dataset variation.

    Attributes
    ----------
    dataset : pd.DataFrame
        The generated noisy dataset.
    '''

    @property
    def dataset(self) -> pd.DataFrame:
        '''The generated dataset, kept as a compatibility alias.'''
        return self.generated


class NoiseGenerator(VariationGenerator[pd.DataFrame]):
    '''Generate noisy copies of a base dataset.

    Each generated variation multiplies the base dataset by normally
    distributed scale factors centered at 1.0. Each configured standard
    deviation creates one noisy dataset variation.

    Attributes
    ----------
    settings : NoiseGeneratorConfig | None, default = None
        Settings that define the standard deviations used for generation.
    base_dataset : pd.DataFrame | None
        Dataset used as the source for generating noisy variations.
    '''

    def __init__(
        self,
        settings: NoiseGeneratorConfig | None = None,
    ) -> None:
        self.settings = NoiseGeneratorConfig() if settings is None else settings
        self._base_dataset: pd.DataFrame | None = None

    @property
    def variation_labels(self) -> tuple[str, ...]:
        '''Return labels for the configured noise variations.

        Returns
        -------
        tuple[str, ...]
            Labels from the generator settings.
        '''
        return self.settings.variation_labels

    @property
    def base_dataset(self) -> pd.DataFrame | None:
        '''Return the base dataset used for noise generation.

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
            Dataset to use for noise generation.
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
    ) -> Iterable[NoiseVariation]:
        '''Generate noisy dataset variations.

        Parameters
        ----------
        random_state : int | None, default = None
            Seed used to make noise generation reproducible.

        Returns
        -------
        Iterable[NoiseVariation]
            Generated noisy dataset variations.
        '''
        rng = np.random.default_rng(random_state)

        for standard_deviation in self.settings.standard_deviations:
            noisy_dataset = self.dataset

            scale_factor_matrix = rng.normal(
                loc=1.0,
                scale=standard_deviation,
                size=noisy_dataset.shape,
            )

            noisy_dataset = noisy_dataset.mul(
                scale_factor_matrix,
                axis="columns",
            )

            variation = NoiseVariation(
                label=f"std_{standard_deviation:g}",
                random_state=random_state,
                generated=noisy_dataset,
            )

            yield variation