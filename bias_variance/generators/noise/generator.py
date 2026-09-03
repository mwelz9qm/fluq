from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from bias_variance.generators.base import Variation, VariationGenerator
from bias_variance.generators.noise.config import NoiseGeneratorConfig
from bias_variance.generators.noise.config_builder import (
    NoiseGeneratorConfigBuilder,
)


@dataclass(frozen=True, slots=True)
class NoiseVariation(Variation[pd.DataFrame]):
    @property
    def dataset(self) -> pd.DataFrame:
        return self.generated


class NoiseGenerator(VariationGenerator[pd.DataFrame]):
    def __init__(
        self,
        settings: NoiseGeneratorConfig | Mapping[str, Any] | None = None,
    ) -> None:
        if isinstance(settings, NoiseGeneratorConfig):
            self.settings = settings
        elif settings is None or isinstance(settings, Mapping):
            self.settings = (
                NoiseGeneratorConfigBuilder()
                .apply_settings(settings)
                .build()
            )
        else:
            raise TypeError(
                'settings must be a NoiseGeneratorConfig, a mapping, or None.'
            )
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
            raise ValueError('Base dataset is not set.')
        return self._base_dataset.copy()

    def generate(
        self,
        *,
        random_state: int | None = None,
    ) -> Iterable[NoiseVariation]:
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
                axis='columns',
            )

            yield NoiseVariation(
                label=f'std_{standard_deviation:g}',
                random_state=random_state,
                generated=noisy_dataset,
            )
