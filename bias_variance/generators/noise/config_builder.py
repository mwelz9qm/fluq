from collections.abc import Mapping
from typing import Any, ClassVar, Self

from bias_variance.generators.noise.config import NoiseGeneratorConfig


class NoiseGeneratorConfigBuilder:
    DEFAULT_STANDARD_DEVIATIONS: ClassVar[tuple[float, ...]] = (
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
    )

    def __init__(self) -> None:
        self._standard_deviations: tuple[float, ...]
        self.reset()

    def reset(self) -> Self:
        self._standard_deviations = self.DEFAULT_STANDARD_DEVIATIONS
        return self

    def set_standard_deviations(
        self,
        standard_deviations: tuple[float, ...],
    ) -> Self:
        self._standard_deviations = standard_deviations
        return self

    def apply_settings(
        self,
        settings: Mapping[str, Any] | None,
    ) -> Self:
        if settings is None:
            return self
        if not isinstance(settings, Mapping):
            raise TypeError('settings must be a mapping or None.')

        unknown_settings = set(settings) - {'standard_deviations'}
        if unknown_settings:
            raise ValueError(
                f'Unknown noise settings: {sorted(unknown_settings)!r}.'
            )
        if 'standard_deviations' in settings:
            self.set_standard_deviations(settings['standard_deviations'])

        return self

    def build(self) -> NoiseGeneratorConfig:
        return NoiseGeneratorConfig(self._standard_deviations)
