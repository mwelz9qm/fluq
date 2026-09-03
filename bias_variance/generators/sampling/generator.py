from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd

from bias_variance.generators.base import Variation, VariationGenerator
from bias_variance.generators.sampling.config import (
    SamplingGeneratorConfig,
    SamplingStrategy,
    SamplingStrategyName,
)
from bias_variance.generators.sampling.config_builder import (
    SamplingGeneratorConfigBuilder,
)
from bias_variance.seeding import derive_keyed_seed, resolve_seed


@dataclass(frozen=True, slots=True)
class SamplingVariation(Variation[pd.DataFrame]):
    @property
    def dataset(self) -> pd.DataFrame:
        return self.generated


class SamplingGenerator(VariationGenerator[pd.DataFrame]):
    def __init__(
        self,
        settings: (
            SamplingGeneratorConfig
            | Mapping[
                SamplingStrategyName | str,
                Mapping[str, Any] | SamplingStrategy,
            ]
            | None
        ) = None,
    ) -> None:
        if isinstance(settings, SamplingGeneratorConfig):
            self.settings = settings
        elif settings is None or isinstance(settings, Mapping):
            self.settings = (
                SamplingGeneratorConfigBuilder()
                .apply_settings(settings)
                .build()
            )
        else:
            raise TypeError(
                'settings must be a SamplingGeneratorConfig, a mapping, or '
                'None.'
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
    ) -> Iterable[SamplingVariation]:
        parent_seed = resolve_seed(random_state)
        for name, strategy in self.settings.sampling_strategies.items():
            variation_seed = derive_keyed_seed(
                parent_seed,
                'sampling',
                name.value,
            )
            yield SamplingVariation(
                label=name.value,
                variation_seed=variation_seed,
                generated=strategy.function(
                    self.dataset,
                    random_state=variation_seed,
                    **strategy.kwargs,
                ),
            )
