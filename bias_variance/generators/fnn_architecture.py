from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from numbers import Real
from types import MappingProxyType

import numpy as np

from ..models.fnn import FnnArchitecture
from .base import Variation, VariationGenerator, VariationGeneratorConfig


class ArchitectureName(StrEnum):

    WIDE = 'wide'
    NARROW = 'narrow'
    TAPER = 'taper'
    REVERSE_TAPER = 'reverse_taper'
    COMBINED_TAPER = 'combined_taper'


@dataclass(frozen=True, slots=True)
class FnnRandomArchitectureConfig:
    layer_range: tuple[int, int] | None = None
    size_range: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        self._validate_integer_range('layer_range', self.layer_range)
        self._validate_integer_range('size_range', self.size_range)

    @staticmethod
    def _validate_integer_range(
        name: str,
        value: tuple[int, int] | None,
    ) -> None:
        if value is None:
            return

        if not isinstance(value, tuple):
            raise TypeError(f'{name} must be a tuple or None.')

        if len(value) != 2:
            raise ValueError(f'{name} must contain exactly two bounds.')

        if any(
            not isinstance(bound, int) or isinstance(bound, bool)
            for bound in value
        ):
            raise TypeError(f'{name} bounds must be integers.')

        lower, upper = value
        if lower < 1:
            raise ValueError(f'{name} lower bound must be at least 1.')

        if lower >= upper:
            raise ValueError(
                f'{name} lower bound must be less than its upper bound.'
            )


@dataclass(frozen=True, slots=True)
class FnnTaperArchitectureConfig:
    layer_range: tuple[int, int] | None = None
    start_size_range: tuple[int, int] | None = None
    taper_rate_range: tuple[float, float] | None = None
    max_size: int | None = None

    def __post_init__(self) -> None:
        FnnRandomArchitectureConfig._validate_integer_range(
            'layer_range',
            self.layer_range,
        )
        FnnRandomArchitectureConfig._validate_integer_range(
            'start_size_range',
            self.start_size_range,
        )
        self._validate_taper_rate_range()
        self._validate_max_size()

    def _validate_taper_rate_range(self) -> None:
        value = self.taper_rate_range
        if value is None:
            return

        if not isinstance(value, tuple):
            raise TypeError('taper_rate_range must be a tuple or None.')

        if len(value) != 2:
            raise ValueError(
                'taper_rate_range must contain exactly two bounds.'
            )

        if any(
            not isinstance(bound, Real) or isinstance(bound, bool)
            for bound in value
        ):
            raise TypeError('taper_rate_range bounds must be numeric.')

        lower, upper = value
        if not 0 < lower < upper < 1:
            raise ValueError(
                'taper_rate_range bounds must satisfy 0 < lower < upper < 1.'
            )

    def _validate_max_size(self) -> None:
        if self.max_size is None:
            return

        if not isinstance(self.max_size, int) or isinstance(self.max_size, bool):
            raise TypeError('max_size must be an integer or None.')

        if self.max_size < 1:
            raise ValueError('max_size must be at least 1.')

        if (
            self.start_size_range is not None
            and self.max_size < self.start_size_range[1]
        ):
            raise ValueError(
                'max_size must be at least the upper bound of '
                'start_size_range.'
            )


DEFAULT_RANDOM_CONFIG = MappingProxyType({
    ArchitectureName.WIDE: FnnRandomArchitectureConfig(
        layer_range=(1, 4),
        size_range=(64, 256),
    ),
    ArchitectureName.NARROW: FnnRandomArchitectureConfig(
        layer_range=(4, 16),
        size_range=(2, 64),
    ),
})

DEFAULT_TAPER_CONFIG = MappingProxyType({
    ArchitectureName.TAPER: FnnTaperArchitectureConfig(
        layer_range=(1, 16),
        start_size_range=(1, 9),
        taper_rate_range=(0.25, 0.5),
        max_size=256,
    ),
    ArchitectureName.REVERSE_TAPER: FnnTaperArchitectureConfig(
        layer_range=(1, 16),
        start_size_range=(128, 256),
        taper_rate_range=(0.25, 0.5),
        max_size=256,
    ),
    ArchitectureName.COMBINED_TAPER: FnnTaperArchitectureConfig(
        layer_range=(1, 16),
        start_size_range=(1, 9),
        taper_rate_range=(0.25, 0.5),
        max_size=256,
    ),
})


@dataclass(frozen=True, slots=True)
class FnnArchitectureGeneratorConfig(VariationGeneratorConfig):
    range_architectures: Mapping[
        ArchitectureName,
        FnnRandomArchitectureConfig,
    ] = field(default_factory=lambda: DEFAULT_RANDOM_CONFIG)
    taper_architectures: Mapping[
        ArchitectureName,
        FnnTaperArchitectureConfig,
    ] = field(default_factory=lambda: DEFAULT_TAPER_CONFIG)

    def __post_init__(self) -> None:
        self._validate_architecture_mapping(
            'range_architectures',
            self.range_architectures,
            frozenset((ArchitectureName.WIDE, ArchitectureName.NARROW)),
            FnnRandomArchitectureConfig,
        )
        self._validate_architecture_mapping(
            'taper_architectures',
            self.taper_architectures,
            frozenset((
                ArchitectureName.TAPER,
                ArchitectureName.REVERSE_TAPER,
                ArchitectureName.COMBINED_TAPER,
            )),
            FnnTaperArchitectureConfig,
        )

        normalized_range_config = {}
        for name, architecture in self.range_architectures.items():
            layer_range = architecture.layer_range or DEFAULT_RANDOM_CONFIG[name].layer_range
            size_range = architecture.size_range or DEFAULT_RANDOM_CONFIG[name].size_range
            normalized_range_config[name] = FnnRandomArchitectureConfig(layer_range, size_range)

        normalized_taper_config = {}
        for name, architecture in self.taper_architectures.items():
            layer_range = architecture.layer_range or DEFAULT_TAPER_CONFIG[name].layer_range
            start_size_range = architecture.start_size_range or DEFAULT_TAPER_CONFIG[name].start_size_range
            taper_rate_range = architecture.taper_rate_range or DEFAULT_TAPER_CONFIG[name].taper_rate_range
            max_size = architecture.max_size or DEFAULT_TAPER_CONFIG[name].max_size
            normalized_taper_config[name] = FnnTaperArchitectureConfig(layer_range, start_size_range, taper_rate_range, max_size)

        self._validate_random_architecture_relationships(
            normalized_range_config
        )

        object.__setattr__(
            self,
            'range_architectures',
            MappingProxyType(normalized_range_config),
        )
        object.__setattr__(
            self,
            'taper_architectures',
            MappingProxyType(normalized_taper_config),
        )

    @staticmethod
    def _validate_architecture_mapping(
        name: str,
        value: Mapping,
        supported_names: frozenset[ArchitectureName],
        config_type: type,
    ) -> None:
        if not isinstance(value, Mapping):
            raise TypeError(f'{name} must be a mapping.')

        for architecture_name, config in value.items():
            if not isinstance(architecture_name, ArchitectureName):
                raise TypeError(
                    f'{name} keys must be ArchitectureName members.'
                )

            if architecture_name not in supported_names:
                raise ValueError(
                    f'{architecture_name.value!r} is not supported in {name}.'
                )

            if not isinstance(config, config_type):
                raise TypeError(
                    f'{name}[{architecture_name.value!r}] must be a '
                    f'{config_type.__name__}.'
                )

    @staticmethod
    def _validate_random_architecture_relationships(
        configurations: Mapping[
            ArchitectureName,
            FnnRandomArchitectureConfig,
        ],
    ) -> None:
        if not {
            ArchitectureName.WIDE,
            ArchitectureName.NARROW,
        }.issubset(configurations):
            return

        wide = configurations[ArchitectureName.WIDE]
        narrow = configurations[ArchitectureName.NARROW]

        if wide.layer_range[1] > narrow.layer_range[0]:
            raise ValueError(
                'WIDE and NARROW layer_range values must be disjoint, with '
                'WIDE below NARROW.'
            )

        if narrow.size_range[1] > wide.size_range[0]:
            raise ValueError(
                'WIDE and NARROW size_range values must be disjoint, with '
                'NARROW below WIDE.'
            )

    @property
    def variation_labels(self) -> tuple[str, ...]:
        labels = [
            architecture.value
            for architecture
            in self.range_architectures
        ]
        labels.extend([
            architecture.value
            for architecture
            in self.taper_architectures
        ])

        return tuple(labels)


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
        settings: FnnArchitectureGeneratorConfig | None = None,
    ) -> None:
        self.settings = (
            FnnArchitectureGeneratorConfig()
            if settings is None
            else settings
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

        for name, config in self.settings.range_architectures.items():
            variation = FnnArchitectureVariation(
                label=name.value,
                random_state=random_state,
                generated=self._generate_random_sizes(config, rng)
            )
            yield variation

        for name, config in self.settings.taper_architectures.items():
            variation = FnnArchitectureVariation(
                label=name.value,
                random_state=random_state,
                generated=self._generate_taper_sizes(config, name, rng)
            )
            yield variation
