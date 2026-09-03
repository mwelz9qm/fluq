from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from numbers import Real
from types import MappingProxyType

from bias_variance.generators.base import VariationGeneratorConfig


class ArchitectureName(StrEnum):

    WIDE = 'wide'
    NARROW = 'narrow'
    TAPER = 'taper'
    REVERSE_TAPER = 'reverse_taper'
    COMBINED_TAPER = 'combined_taper'


@dataclass(frozen=True, slots=True)
class FnnRandomArchitectureConfig:
    layer_range: tuple[int, int]
    size_range: tuple[int, int]

    def __post_init__(self) -> None:
        self._validate_integer_range('layer_range', self.layer_range)
        self._validate_integer_range('size_range', self.size_range)

    @staticmethod
    def _validate_integer_range(
        name: str,
        value: tuple[int, int],
    ) -> None:
        
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
    layer_range: tuple[int, int]
    start_size_range: tuple[int, int]
    taper_rate_range: tuple[float, float]
    max_size: int

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

        if not isinstance(self.max_size, int) or isinstance(self.max_size, bool):
            raise TypeError('max_size must be an integer or None.')

        if self.max_size < 1:
            raise ValueError('max_size must be at least 1.')

        if self.max_size < self.start_size_range[1]:
            raise ValueError(
                'max_size must be at least the upper bound of '
                'start_size_range.'
            )


@dataclass(frozen=True, slots=True)
class FnnArchitectureGeneratorConfig(VariationGeneratorConfig):
    range_architectures: Mapping[
        ArchitectureName,
        FnnRandomArchitectureConfig,
    ] | None = None
    taper_architectures: Mapping[
        ArchitectureName,
        FnnTaperArchitectureConfig,
    ] | None = None

    def __post_init__(self) -> None:
        for name, architectures in (
            ('range_architectures', self.range_architectures),
            ('taper_architectures', self.taper_architectures),
        ):
            if architectures is not None and not isinstance(
                architectures,
                Mapping,
            ):
                raise TypeError(f'{name} must be a mapping.')

        if not self.range_architectures and not self.taper_architectures:
            raise ValueError(
                'range_architectures and taper_architectures cannot both be '
                'None or empty.'
            )

        if self.range_architectures is not None:
            self._validate_architecture_mapping(
                'range_architectures',
                self.range_architectures,
                frozenset((ArchitectureName.WIDE, ArchitectureName.NARROW)),
                FnnRandomArchitectureConfig,
            )
            self._validate_random_architecture_relationships(
                self.range_architectures
            )
            object.__setattr__(
                self,
                'range_architectures',
                MappingProxyType(dict(self.range_architectures)),
            )

        if self.taper_architectures is not None:
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
            object.__setattr__(
                self,
                'taper_architectures',
                MappingProxyType(dict(self.taper_architectures)),
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
            in (self.range_architectures or {})
        ]
        labels.extend([
            architecture.value
            for architecture
            in (self.taper_architectures or {})
        ])

        return tuple(labels)
