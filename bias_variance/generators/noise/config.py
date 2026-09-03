from dataclasses import dataclass
from numbers import Real

import numpy as np

from bias_variance.generators.base import VariationGeneratorConfig


@dataclass(frozen=True, slots=True)
class NoiseGeneratorConfig(VariationGeneratorConfig):
    standard_deviations: tuple[float, ...]

    @property
    def variation_labels(self) -> tuple[str, ...]:
        return tuple(
            f'std_{standard_deviation:g}'
            for standard_deviation in self.standard_deviations
        )

    @staticmethod
    def _validate_standard_deviations(values: tuple[float, ...]) -> None:
        if not isinstance(values, tuple):
            raise TypeError('Standard deviations must be provided as a tuple.')

        if any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, Real)
            for value in values
        ):
            raise TypeError(
                'Standard deviations must contain only real numbers.'
            )

        standard_deviations = tuple(float(value) for value in values)
        if not standard_deviations:
            raise ValueError(
                'At least one standard deviation must be configured.'
            )

        if any(not np.isfinite(value) for value in standard_deviations):
            raise ValueError('Standard deviations must be finite.')

        if any(value <= 0 for value in standard_deviations):
            raise ValueError('Standard deviations must be greater than zero.')

        if any(value >= 1 for value in standard_deviations):
            raise ValueError('Standard deviations must be less than one.')

        if len(set(standard_deviations)) != len(standard_deviations):
            raise ValueError('Standard deviations must be unique.')

    def __post_init__(self) -> None:
        self._validate_standard_deviations(self.standard_deviations)
        object.__setattr__(
            self,
            'standard_deviations',
            tuple(float(value) for value in self.standard_deviations),
        )
