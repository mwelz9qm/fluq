from dataclasses import dataclass
from numbers import Real

import numpy as np
import pandas as pd

from .base import Variation, VariationGenerator, VariationGeneratorConfig


@dataclass(frozen=True, slots=True)
class NoiseGeneratorConfig(VariationGeneratorConfig):
    _base_dataset: pd.DataFrame
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

    @property
    def dataset(self) -> pd.DataFrame:
        return self._base_dataset.copy()

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
        self._validate_dataset(self._base_dataset)
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
        settings: NoiseGeneratorConfig
    ) -> None:
        self.settings = settings

    @property
    def variation_labels(self):
        return self.settings.variation_labels

    def generate(
        self,
        *,
        random_state: int | None = None,
    ) -> list[NoiseVariation]:
        rng = np.random.default_rng(random_state)
        variations: list[NoiseVariation] = []

        # generation loop
        for standard_deviation in self.settings.standard_deviations:
            # get copy from base dataset
            noisy_dataset = self.settings.dataset

            # construct scale factor matrix
            scale_factor_matrix = rng.normal(
                loc=1.0,
                scale=standard_deviation,
                size=noisy_dataset.shape,
            )

            # multiply by scale factor for each
            # corresponding entry in dataset and matrix
            noisy_dataset.mul(
                scale_factor_matrix,
                axis="columns",
            )

            # add the generated variation
            variations.append(
                NoiseVariation(
                    label=f"std_{standard_deviation:g}",
                    random_state=random_state,
                    generated=noisy_dataset,
                )
            )

        return variations
