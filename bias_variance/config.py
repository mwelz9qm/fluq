from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Self

import pandas as pd
from sklearn.model_selection import train_test_split

from bias_variance.generators.base import VariationGenerator, VariationGeneratorConfig
from bias_variance.generators.fnn_architecture import (
    FnnArchitectureGenerator,
    FnnArchitectureGeneratorConfig,
)
from bias_variance.generators.noise import NoiseGenerator, NoiseGeneratorConfig
from bias_variance.generators.sampling import SamplingGenerator, SamplingGeneratorConfig
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

        if isinstance(variation_generator_configs, Iterable[VariationGeneratorConfig]):
            return self._set('variation_generator_configs', variation_generator_configs)

        elif isinstance(variation_generator_configs, Mapping[str, Mapping[str, Any]]):
            converted_configs: list[VariationGeneratorConfig] = []
            for key, kwargs in variation_generator_configs.items():
                match key:
                    case StudyBias.MODEL.value:
                        converted_configs.append(FnnArchitectureGeneratorConfig(**kwargs))

                    case StudyBias.SAMPLING.value:
                        converted_configs.append(SamplingGeneratorConfig(**kwargs))

                    case StudyBias.DATA.value:
                        converted_configs.append(NoiseGeneratorConfig(**kwargs))

            return self._set('variation_generator_configs', converted_configs)

        else:
            raise TypeError(
                'Unknown variation generator type.'
            )
        
    def set_evaluation_methods(
        self,
        evaluation_methods: Iterable[EvaluationMethod] | Iterable[str]
    ) -> Self:
        if evaluation_methods is None:
            raise ValueError(
                'Empty evaluation_methods not allowed.'
                'Must contain at least one evaluation method.'
            )
        
        if isinstance(evaluation_methods, Iterable[EvaluationMethod]):
            return self._set('evaluation_methods', evaluation_methods)

        converted_methods: list[EvaluationMethod] = []
        for method in evaluation_methods:
            match method:
                case EvaluationMethod.AVERAGING.value:
                    converted_methods.append(EvaluationMethod.AVERAGING)

                case EvaluationMethod.POINTWISE.value:
                    converted_methods.append(EvaluationMethod.POINTWISE)

                case _:
                    raise ValueError(
                        f'Unknown evaluation method: {method}'
                    )
        return converted_methods

    def set_base_architecture(self, base_architecture: FnnArchitecture | tuple[int, ...]) -> Self:
        if isinstance(base_architecture, tuple[int, ...]):
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
        if isinstance(test_metrics, Iterable[MetricName]):
            return self._set('test_metrics', test_metrics)

        elif isinstance(test_metrics, Iterable[str]):
            converted_metrics: list[MetricName] = []
            for metric in test_metrics:
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

        evaluation_methods = self._config_data.get('evaluation_methods', None)
        n_iter = self._config_data.get('n_iter', None)
        test_metrics = self._config_data.get('test_metrics', None)

        self.reset()

        return RunConfig(
            baseline=baseline,
            studies=tuple(studies),
            evaluation_methods=evaluation_methods,
            n_iter=n_iter,
            test_size=test_size,
            test_metrics=test_metrics,
            random_state=random_state
        )
