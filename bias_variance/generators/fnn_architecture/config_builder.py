from collections.abc import Mapping
from dataclasses import asdict
from types import MappingProxyType
from typing import Any, ClassVar, Self

from bias_variance.generators.fnn_architecture.config import (
    ArchitectureName,
    FnnArchitectureGeneratorConfig,
    FnnRandomArchitectureConfig,
    FnnTaperArchitectureConfig,
)


class FnnArchitectureGeneratorConfigBuilder:
    DEFAULT_RANDOM_CONFIG: ClassVar[
        MappingProxyType[ArchitectureName, FnnRandomArchitectureConfig]
    ] = MappingProxyType({
        ArchitectureName.WIDE: FnnRandomArchitectureConfig(
            layer_range=(1, 4),
            size_range=(64, 256),
        ),
        ArchitectureName.NARROW: FnnRandomArchitectureConfig(
            layer_range=(4, 16),
            size_range=(2, 64),
        ),
    })

    DEFAULT_TAPER_CONFIG: ClassVar[
        MappingProxyType[ArchitectureName, FnnTaperArchitectureConfig]
    ] = MappingProxyType({
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

    def __init__(self) -> None:
        self._config_data: dict[ArchitectureName, dict[str, Any]]
        self.reset()

    @classmethod
    def _default_configs(
        cls,
    ) -> dict[ArchitectureName, dict[str, Any]]:
        defaults = {
            name: asdict(config)
            for name, config in cls.DEFAULT_RANDOM_CONFIG.items()
        }
        defaults.update({
            name: asdict(config)
            for name, config in cls.DEFAULT_TAPER_CONFIG.items()
        })
        return defaults

    def reset(self) -> Self:
        self._config_data = self._default_configs()
        return self

    @staticmethod
    def _architecture_name(value: ArchitectureName | str) -> ArchitectureName:
        if isinstance(value, ArchitectureName):
            return value
        if not isinstance(value, str):
            raise TypeError(
                'Architecture configuration keys must be ArchitectureName '
                'members or strings.'
            )
        try:
            return ArchitectureName(value)
        except ValueError as error:
            raise ValueError(f'Unknown architecture name: {value!r}.') from error

    @classmethod
    def _config_type(
        cls,
        name: ArchitectureName,
    ) -> type[FnnRandomArchitectureConfig | FnnTaperArchitectureConfig]:
        if name in cls.DEFAULT_RANDOM_CONFIG:
            return FnnRandomArchitectureConfig
        return FnnTaperArchitectureConfig

    @classmethod
    def _merged_settings(
        cls,
        name: ArchitectureName,
        settings: Mapping[str, Any]
        | FnnRandomArchitectureConfig
        | FnnTaperArchitectureConfig,
    ) -> dict[str, Any]:
        config_type = cls._config_type(name)
        if isinstance(settings, config_type):
            overrides = asdict(settings)
        elif isinstance(settings, Mapping):
            overrides = dict(settings)
        else:
            raise TypeError(
                f'Configuration for {name.value!r} must be a mapping or '
                f'{config_type.__name__}.'
            )

        return {
            **cls._default_configs()[name],
            **overrides,
        }

    def apply_settings(
        self,
        settings: Mapping[
            ArchitectureName | str,
            Mapping[str, Any]
            | FnnRandomArchitectureConfig
            | FnnTaperArchitectureConfig,
        ] | None,
    ) -> Self:
        if settings is None:
            return self
        if not isinstance(settings, Mapping):
            raise TypeError('settings must be a mapping or None.')
        if not settings:
            raise ValueError('settings must select at least one architecture.')

        selected: dict[ArchitectureName, dict[str, Any]] = {}
        for raw_name, architecture_settings in settings.items():
            name = self._architecture_name(raw_name)
            if name in selected:
                raise ValueError(
                    f'Duplicate architecture configuration: {name.value!r}.'
                )
            selected[name] = self._merged_settings(
                name,
                architecture_settings,
            )

        self._config_data = selected
        return self

    def build(self) -> FnnArchitectureGeneratorConfig:
        if not self._config_data:
            raise ValueError('At least one architecture must be configured.')

        range_architectures = {
            name: FnnRandomArchitectureConfig(**settings)
            for name, settings in self._config_data.items()
            if name in self.DEFAULT_RANDOM_CONFIG
        }
        taper_architectures = {
            name: FnnTaperArchitectureConfig(**settings)
            for name, settings in self._config_data.items()
            if name in self.DEFAULT_TAPER_CONFIG
        }

        return FnnArchitectureGeneratorConfig(
            range_architectures=(
                MappingProxyType(range_architectures)
                if range_architectures
                else None
            ),
            taper_architectures=(
                MappingProxyType(taper_architectures)
                if taper_architectures
                else None
            ),
        )
