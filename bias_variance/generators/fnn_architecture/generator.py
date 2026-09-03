from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from bias_variance.generators.base import Variation, VariationGenerator
from bias_variance.generators.fnn_architecture.config import (
    ArchitectureName,
    FnnArchitectureGeneratorConfig,
    FnnRandomArchitectureConfig,
    FnnTaperArchitectureConfig,
)
from bias_variance.generators.fnn_architecture.config_builder import (
    FnnArchitectureGeneratorConfigBuilder,
)
from bias_variance.models.fnn import FnnArchitecture


@dataclass(frozen=True, slots=True)
class FnnArchitectureVariation(Variation[FnnArchitecture]):
    def __post_init__(self) -> None:
        Variation.__post_init__(self)

        if not isinstance(self.generated, FnnArchitecture):
            raise TypeError('generated must be an FnnArchitecture.')

    @property
    def architecture(self) -> FnnArchitecture:
        return self.generated

class FnnArchitectureGenerator(VariationGenerator[FnnArchitecture]):
    def __init__(
        self,
        settings: (
            FnnArchitectureGeneratorConfig
            | Mapping[
                ArchitectureName | str,
                Mapping[str, Any]
                | FnnRandomArchitectureConfig
                | FnnTaperArchitectureConfig,
            ]
            | None
        ) = None,
    ) -> None:
        if isinstance(settings, FnnArchitectureGeneratorConfig):
            self.settings = settings
        elif settings is None or isinstance(settings, Mapping):
            self.settings = (
                FnnArchitectureGeneratorConfigBuilder()
                .apply_settings(settings)
                .build()
            )
        else:
            raise TypeError(
                'settings must be an FnnArchitectureGeneratorConfig, a '
                'mapping, or None.'
            )

    @property
    def variation_labels(self) -> tuple[str, ...]:
        return self.settings.variation_labels

    @staticmethod
    def _generate_random_sizes(
        config: FnnRandomArchitectureConfig,
        rng: np.random.Generator,
    ) -> FnnArchitecture:
        low_layers, high_layers = config.layer_range
        n_layers = rng.integers(low_layers, high_layers, dtype=int)
        low_size, high_size = config.size_range

        return FnnArchitecture(
            hidden_layers=tuple(
                int(size)
                for size in rng.integers(
                    low_size,
                    high_size,
                    size=n_layers
                )
            )
        )

    @staticmethod
    def _generate_taper_sizes(
        config: FnnTaperArchitectureConfig,
        taper_type: ArchitectureName,
        rng: np.random.Generator,
    ) -> FnnArchitecture:
        low_layers, high_layers = config.layer_range
        n_layers = rng.integers(low_layers, high_layers, dtype=int)

        low_size, high_size = config.start_size_range
        start_size = rng.integers(low_size, high_size, dtype=int)

        low_taper_rate, high_taper_rate = config.taper_rate_range
        match taper_type:
            case ArchitectureName.TAPER:
                size_rate = rng.uniform(low_taper_rate + 1, high_taper_rate + 1)

            case ArchitectureName.REVERSE_TAPER:
                size_rate = rng.uniform(1 - high_taper_rate, 1 - low_taper_rate)

            case ArchitectureName.COMBINED_TAPER:
                size_rate = rng.uniform(low_taper_rate, high_taper_rate)

            case _:
                raise ValueError(
                    f'Unknown taper_type: {taper_type!r}.'
                )

        sizes = []

        if taper_type in (
            ArchitectureName.TAPER,
            ArchitectureName.REVERSE_TAPER,
        ):
            for i in np.arange(n_layers):
                size = round(start_size * (size_rate ** i))
                sizes.append(min(max(size, 1), config.max_size))

        else:
            midpoint = max(1, int(np.ceil(n_layers / 2)))
            for i in np.arange(n_layers):
                if i < midpoint:
                    size = round(start_size * ((1 + size_rate) ** i))
                else:
                    peak = start_size * ((1 + size_rate) ** (midpoint  - 1))
                    size = round(peak * ((1 - size_rate) ** (i - midpoint + 1)))
                sizes.append(min(max(size, 1), config.max_size))

        return FnnArchitecture(
            hidden_layers=tuple(sizes)
        )

    def generate(
        self,
        *,
        random_state: int | None = None,
    ) -> Iterable[FnnArchitectureVariation]:
        rng = np.random.default_rng(random_state)

        for name, config in (self.settings.range_architectures or {}).items():
            variation = FnnArchitectureVariation(
                label=name.value,
                random_state=random_state,
                generated=self._generate_random_sizes(config, rng)
            )
            yield variation

        for name, config in (self.settings.taper_architectures or {}).items():
            variation = FnnArchitectureVariation(
                label=name.value,
                random_state=random_state,
                generated=self._generate_taper_sizes(config, name, rng)
            )
            yield variation
