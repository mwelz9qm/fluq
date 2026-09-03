from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, ClassVar, Self

from bias_variance.generators.sampling.config import (
    SamplingGeneratorConfig,
    SamplingStrategy,
    SamplingStrategyName,
)


class SamplingGeneratorConfigBuilder:
    DEFAULT_SAMPLING_STRATEGIES: ClassVar[
        MappingProxyType[SamplingStrategyName, SamplingStrategy]
    ] = MappingProxyType({
        SamplingStrategyName.BOOTSTRAP: SamplingStrategy(
            function=SamplingStrategyName.BOOTSTRAP.function,
            kwargs={
                'sample_fraction': 1.0,
                'with_replacement': True,
            },
        ),
        SamplingStrategyName.STRATIFIED: SamplingStrategy(
            function=SamplingStrategyName.STRATIFIED.function,
            kwargs={
                'stratify_col_index': 0,
                'sample_fraction': 1.0,
            },
        ),
        SamplingStrategyName.LHS: SamplingStrategy(
            function=SamplingStrategyName.LHS.function,
            kwargs={
                'sample_fraction': 1.0,
            },
        ),
    })

    def __init__(self) -> None:
        self._strategy_kwargs: dict[SamplingStrategyName, dict[str, Any]]
        self.reset()

    def reset(self) -> Self:
        self._strategy_kwargs = {
            name: dict(strategy.kwargs)
            for name, strategy in self.DEFAULT_SAMPLING_STRATEGIES.items()
        }
        return self

    @staticmethod
    def _strategy_name(
        value: SamplingStrategyName | str,
    ) -> SamplingStrategyName:
        if isinstance(value, SamplingStrategyName):
            return value
        if not isinstance(value, str):
            raise TypeError(
                'Sampling strategy keys must be SamplingStrategyName members '
                'or strings.'
            )
        try:
            return SamplingStrategyName(value)
        except ValueError as error:
            raise ValueError(f'Unknown sampling strategy: {value!r}.') from error

    @classmethod
    def _merged_kwargs(
        cls,
        name: SamplingStrategyName,
        settings: Mapping[str, Any] | SamplingStrategy,
    ) -> dict[str, Any]:
        if isinstance(settings, SamplingStrategy):
            if settings.function is not name.function:
                raise ValueError(
                    f'{name.value!r} must use {name.function.__name__}.'
                )
            overrides = dict(settings.kwargs)
        elif isinstance(settings, Mapping):
            overrides = dict(settings)
        else:
            raise TypeError(
                f'Configuration for {name.value!r} must be a mapping or '
                'SamplingStrategy.'
            )

        if 'random_state' in overrides:
            raise ValueError(
                'Sampling strategy kwargs cannot contain random_state.'
            )
        return {
            **cls.DEFAULT_SAMPLING_STRATEGIES[name].kwargs,
            **overrides,
        }

    def apply_settings(
        self,
        settings: Mapping[
            SamplingStrategyName | str,
            Mapping[str, Any] | SamplingStrategy,
        ] | None,
    ) -> Self:
        if settings is None:
            return self
        if not isinstance(settings, Mapping):
            raise TypeError('settings must be a mapping or None.')
        if not settings:
            raise ValueError('settings must select at least one strategy.')

        selected: dict[SamplingStrategyName, dict[str, Any]] = {}
        for raw_name, strategy_settings in settings.items():
            name = self._strategy_name(raw_name)
            if name in selected:
                raise ValueError(
                    f'Duplicate sampling strategy: {name.value!r}.'
                )
            selected[name] = self._merged_kwargs(name, strategy_settings)

        self._strategy_kwargs = selected
        return self

    def build(self) -> SamplingGeneratorConfig:
        if not self._strategy_kwargs:
            raise ValueError('At least one sampling strategy must be configured.')

        return SamplingGeneratorConfig(MappingProxyType({
            name: SamplingStrategy(
                function=name.function,
                kwargs=kwargs,
            )
            for name, kwargs in self._strategy_kwargs.items()
        }))


DEFAULT_SAMPLING_STRATEGIES = (
    SamplingGeneratorConfigBuilder.DEFAULT_SAMPLING_STRATEGIES
)
