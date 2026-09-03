from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, ClassVar, Self

import pandas as pd
from sklearn.model_selection import train_test_split

from bias_variance.generators.base import VariationGenerator, VariationGeneratorConfig
from bias_variance.generators.fnn_architecture import (
    FnnArchitectureGenerator,
    FnnArchitectureGeneratorConfig,
    FnnArchitectureGeneratorConfigBuilder,
)
from bias_variance.generators.noise import (
    NoiseGenerator,
    NoiseGeneratorConfig,
    NoiseGeneratorConfigBuilder,
)
from bias_variance.generators.sampling import (
    SamplingGenerator,
    SamplingGeneratorConfig,
    SamplingGeneratorConfigBuilder,
)
from bias_variance.models.evaluation import EvaluationMethod, MetricName
from bias_variance.models.fnn import FnnArchitecture


class StudyBias(StrEnum):
    MODEL = 'model'
    SAMPLING = 'sampling'
    DATA = 'data'


@dataclass(frozen=True, slots=True)
class Study:
    study_bias: StudyBias
    variation_generator: VariationGenerator


@dataclass(frozen=True, slots=True)
class RunBaseline:
    architecture: FnnArchitecture
    X: pd.DataFrame
    Y: pd.DataFrame
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    Y_train: pd.DataFrame
    Y_test: pd.DataFrame

    @property
    def split(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        return (self.X_train, self.X_test, self.Y_train, self.Y_test)
    
    @property
    def train_set(
        self,
    ) -> pd.DataFrame:
        return pd.concat([self.X_train, self.Y_train], axis='columns')

    @property
    def dataset(
        self,
    ) -> pd.DataFrame:
        return pd.concat([self.X, self.Y], axis='columns')


@dataclass(frozen=True, slots=True)
class RunConfig:
    baseline: RunBaseline
    studies: tuple[Study, ...]
    evaluation_methods: tuple[EvaluationMethod, ...]
    n_iter: int
    test_size: float
    test_metrics: frozenset[MetricName]
    random_state: int | None


class RunConfigBuilder:
    _RUN_SETTING_SETTERS: ClassVar[Mapping[str, str]] = {
        'variation_generator_configs': 'set_variation_generator_configs',
        'base_architecture': 'set_base_architecture',
        'n_iter': 'set_n_iter',
        'test_size': 'set_test_size',
        'test_metrics': 'set_test_metrics',
        'evaluation_methods': 'set_evaluation_methods',
        'random_state': 'set_random_state',
    }

    def __init__(self):
        self._config_data: dict[str, Any] = {}
        self.reset()

    def reset(self) -> None:
        self._config_data = {
            'base_architecture': FnnArchitecture((64, 64, 64, 64, 64, 64, 64, 64)),
            'evaluation_methods': [
                EvaluationMethod.AVERAGING,
                EvaluationMethod.POINTWISE
            ],
            'n_iter': 100,
            'test_size': 0.2,
            'test_metrics': [MetricName.MSE, MetricName.R2],
        }

    def _set(self, key: str, value: Any) -> Self:
        self._config_data[key] = value
        return self

    def apply_run_settings(
        self,
        run_settings: Mapping[str, Any] | None,
    ) -> Self:
        if run_settings is None:
            return self
        if not isinstance(run_settings, Mapping):
            raise TypeError('run_settings must be a mapping or None.')

        unknown_settings = set(run_settings) - set(self._RUN_SETTING_SETTERS)
        if unknown_settings:
            raise ValueError(
                'Unknown run settings: '
                f'{sorted(unknown_settings)!r}.'
            )

        for key, value in run_settings.items():
            setter = getattr(self, self._RUN_SETTING_SETTERS[key])
            setter(value)

        return self

    def set_X(self, X: pd.DataFrame) -> Self:
        return self._set('X', X)

    def set_Y(self, Y: pd.DataFrame) -> Self:
        return self._set('Y', Y)

    def set_variation_generator_configs(
        self,
        variation_generator_configs: Iterable[VariationGeneratorConfig] | Mapping[str, Mapping[str, Any]]
    ) -> Self:
        if variation_generator_configs is None:
            raise ValueError(
                'Empty variation_generator_configs not allowed.'
                'Must contain at least one variation generator config.'
            )

        if isinstance(variation_generator_configs, Mapping):
            converted_configs: list[VariationGeneratorConfig] = []
            for key, kwargs in variation_generator_configs.items():
                if not isinstance(kwargs, Mapping):
                    raise TypeError(
                        f'Configuration for {key!r} must be a mapping.'
                    )
                match key:
                    case StudyBias.MODEL.value:
                        config = (
                            FnnArchitectureGeneratorConfigBuilder()
                            .apply_settings(kwargs)
                            .build()
                        )

                    case StudyBias.SAMPLING.value:
                        config = (
                            SamplingGeneratorConfigBuilder()
                            .apply_settings(kwargs)
                            .build()
                        )

                    case StudyBias.DATA.value:
                        config = (
                            NoiseGeneratorConfigBuilder()
                            .apply_settings(kwargs)
                            .build()
                        )

                    case _:
                        raise ValueError(
                            f'Unknown variation generator config: {key!r}.'
                        )

                converted_configs.append(config)

            return self._set('variation_generator_configs', converted_configs)

        if isinstance(variation_generator_configs, Iterable) and not isinstance(
            variation_generator_configs, (str, bytes)
        ):
            configs = tuple(variation_generator_configs)
            if not all(
                isinstance(config, VariationGeneratorConfig)
                for config in configs
            ):
                raise TypeError(
                    'Variation generator configs must contain only '
                    'VariationGeneratorConfig instances.'
                )
            return self._set('variation_generator_configs', configs)

        raise TypeError('Unknown variation generator type.')
        
    def set_evaluation_methods(
        self,
        evaluation_methods: Iterable[EvaluationMethod] | Iterable[str]
    ) -> Self:
        if evaluation_methods is None:
            raise ValueError(
                'Empty evaluation_methods not allowed.'
                'Must contain at least one evaluation method.'
            )
        if isinstance(evaluation_methods, (str, bytes)):
            raise TypeError('evaluation_methods must be an iterable of methods.')
        
        converted_methods: list[EvaluationMethod] = []
        for method in evaluation_methods:
            if isinstance(method, EvaluationMethod):
                converted_methods.append(method)
                continue
            match method:
                case EvaluationMethod.AVERAGING.value:
                    converted_methods.append(EvaluationMethod.AVERAGING)

                case EvaluationMethod.POINTWISE.value:
                    converted_methods.append(EvaluationMethod.POINTWISE)

                case _:
                    raise ValueError(
                        f'Unknown evaluation method: {method}'
                    )
        return self._set('evaluation_methods', converted_methods)

    def set_base_architecture(self, base_architecture: FnnArchitecture | tuple[int, ...]) -> Self:
        if isinstance(base_architecture, tuple):
            return self._set('base_architecture', FnnArchitecture(base_architecture))

        elif isinstance(base_architecture, FnnArchitecture):
            return self._set('base_architecture', base_architecture)

        else:
            raise TypeError(
                'Unknown base_architecture type.'
            )

    def set_n_iter(self, n_iter: int) -> Self:
        return self._set('n_iter', n_iter)

    def set_test_size(self, test_size: float) -> Self:
        return self._set('test_size', test_size)

    def set_test_metrics(self, test_metrics: Iterable[MetricName] | Iterable[str]) -> Self:
        if isinstance(test_metrics, Iterable) and not isinstance(
            test_metrics, (str, bytes)
        ):
            converted_metrics: list[MetricName] = []
            for metric in test_metrics:
                if isinstance(metric, MetricName):
                    converted_metrics.append(metric)
                    continue
                match metric:
                    case MetricName.MSE.value:
                        converted_metrics.append(MetricName.MSE)

                    case MetricName.RMSE.value:
                        converted_metrics.append(MetricName.RMSE)

                    case MetricName.MAE.value:
                        converted_metrics.append(MetricName.MAE)

                    case MetricName.R2.value:
                        converted_metrics.append(MetricName.R2)

                    case _:
                        raise ValueError(
                            f'Unsupported test metric: {metric!r}.'
                        )

            return self._set('test_metrics', converted_metrics)

        else:
            raise TypeError(
                'Unknown test_metrics type.'
            )

    def set_random_state(self, random_state: int) -> Self:
        return self._set('random_state', random_state)

    def build(self) -> RunConfig:
        X = self._config_data.get('X', None)
        Y = self._config_data.get('Y', None)
        test_size = self._config_data.get('test_size', None)
        random_state = self._config_data.get('random_state', None)

        if not isinstance(X, pd.DataFrame) or not isinstance(Y, pd.DataFrame):
            raise TypeError('X and Y must be pandas DataFrames.')
        if X.empty or Y.empty:
            raise ValueError('X and Y must not be empty.')
        if len(X) != len(Y):
            raise ValueError('X and Y must contain the same number of rows.')

        n_iter = self._config_data.get('n_iter', None)
        if not isinstance(n_iter, int) or isinstance(n_iter, bool):
            raise TypeError('n_iter must be an integer.')
        if n_iter <= 0:
            raise ValueError('n_iter must be greater than zero.')
        if not isinstance(test_size, (int, float)) or isinstance(test_size, bool):
            raise TypeError('test_size must be numeric.')
        if not 0 < test_size < 1:
            raise ValueError('test_size must be between zero and one.')

        split = train_test_split(
            X,
            Y,
            test_size=test_size,
            random_state=random_state
        )

        base_architecture = self._config_data.get('base_architecture', None)
        
        baseline = RunBaseline(
            base_architecture,
            X,
            Y,
            *split,        
        )

        variation_generator_configs = self._config_data.get('variation_generator_configs', None)

        studies = []
        
        if variation_generator_configs is None:
            default_studies = [
                Study(StudyBias.MODEL, FnnArchitectureGenerator()),
                Study(StudyBias.SAMPLING, SamplingGenerator()),
                Study(StudyBias.DATA, NoiseGenerator())
            ]
            studies = default_studies
        else:
            for config in variation_generator_configs:
                match config:
                    case FnnArchitectureGeneratorConfig():
                        studies.append(Study(StudyBias.MODEL, FnnArchitectureGenerator(config)))
        
                    case SamplingGeneratorConfig():
                        studies.append(Study(StudyBias.SAMPLING, SamplingGenerator(config)))
        
                    case NoiseGeneratorConfig():
                        studies.append(Study(StudyBias.DATA, NoiseGenerator(config)))

        if not studies:
            raise ValueError('At least one study must be configured.')

        evaluation_methods = self._config_data.get('evaluation_methods', None)
        test_metrics = self._config_data.get('test_metrics', None)

        if not evaluation_methods:
            raise ValueError('At least one evaluation method must be configured.')
        if not test_metrics:
            raise ValueError('At least one test metric must be configured.')

        test_metrics = set(test_metrics)
        if EvaluationMethod.AVERAGING in evaluation_methods:
            test_metrics.add(MetricName.MSE)

        self.reset()

        return RunConfig(
            baseline=baseline,
            studies=tuple(studies),
            evaluation_methods=tuple(evaluation_methods),
            n_iter=n_iter,
            test_size=test_size,
            test_metrics=frozenset(test_metrics),
            random_state=random_state
        )
