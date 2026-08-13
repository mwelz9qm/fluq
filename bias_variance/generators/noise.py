from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Real

import numpy as np
import pandas as pd

from .base import Variation, VariationGenerator, VariationGeneratorConfig


@dataclass(frozen=True, slots=True)
class NoiseGeneratorConfig(VariationGeneratorConfig):
    standard_deviations: tuple[float] = (
        0.1, 0.2, 0.3, 0.4, 0.5
    )

    @property
    def variation_labels(self) -> tuple[str, ...]:
        return tuple(
            f"std_{standard_deviation:g}"
            for standard_deviation
            in self.standard_deviations
        )

    @staticmethod
    def _validate_standard_deviations(values: tuple[float]) -> None:
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
        self._validate_standard_deviations(self.standard_deviations)


@dataclass(frozen=True, slots=True)
class NoiseVariation(Variation[pd.DataFrame]):
    @property
    def dataset(self) -> pd.DataFrame:
        """The generated dataset (kept as a compatibility alias)."""
        return self.generated


class NoiseGenerator(VariationGenerator[pd.DataFrame]):
    def __init__(
        self,
        settings: NoiseGeneratorConfig | None = None,
    ) -> None:
        self.settings = NoiseGeneratorConfig() if settings is None else settings
        self._base_dataset: pd.DataFrame | None = None

    @property
    def variation_labels(self) -> tuple[str, ...]:
        return self.settings.variation_labels

    @property
    def base_dataset(self) -> pd.DataFrame | None:
        return self._base_dataset

    @base_dataset.setter
    def base_dataset(self, value) -> None:
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
    ) -> Iterable[NoiseVariation]:
        rng = np.random.default_rng(random_state)

        # generation loop
        for standard_deviation in self.settings.standard_deviations:
            # get copy from base dataset
            noisy_dataset = self.dataset

            # construct scale factor matrix
            scale_factor_matrix = rng.normal(
                loc=1.0,
                scale=standard_deviation,
                size=noisy_dataset.shape,
            )

            # multiply by scale factor for each
            # corresponding entry in dataset and matrix
            noisy_dataset = noisy_dataset.mul(
                scale_factor_matrix,
                axis="columns",
            )

            # add the generated variation
            variation = NoiseVariation(
                label=f"std_{standard_deviation:g}",
                random_state=random_state,
                generated=noisy_dataset,
            )

            yield variation
