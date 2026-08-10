from dataclasses import dataclass
from enum import StrEnum

import pandas as pd
from sklearn.model_selection import train_test_split

from bias_variance.generators.base import GeneratorConfig
from bias_variance.generators.fnn_architecture import (
    DEFAULT_RANDOM_CONFIG,
    DEFAULT_TAPER_CONFIG,
    FnnArchitectureGeneratorConfig,
)
from bias_variance.generators.noise import NoiseGeneratorConfig
from bias_variance.generators.sampling import SamplingGeneratorConfig
from bias_variance.models.evaluation import EvaluationMethod, MetricName
from bias_variance.models.fnn import FnnArchitecture


class StudyName(StrEnum):
    MODEL = 'model'
    SAMPLING = 'sampling'
    DATA = 'data'


@dataclass(frozen=True, slots=True)
class Study:
    study_name: StudyName
    evaluation_method: EvaluationMethod
    generator_config: GeneratorConfig

    def __post_init__(self) -> None:
        match self.study_name:
            case StudyName.MODEL:
                if not isinstance(self.generator_config, FnnArchitectureGeneratorConfig):
                    raise TypeError(
                        f'Mismatched generator_config type with study_name: {self.generator_config!r}, {self.study_name!r}.'
                    )
                
            case StudyName.SAMPLING:
                if not isinstance(self.generator_config, SamplingGeneratorConfig):
                    raise TypeError(
                        f'Mismatched generator_config type with study_name: {self.generator_config!r}, {self.study_name!r}.'
                    )

            case StudyName.DATA:
                if not isinstance(self.generator_config, NoiseGeneratorConfig):
                    raise TypeError(
                        f'Mismatched generator_config type with study_name: {self.generator_config!r}, {self.study_name!r}.'
                    )

    @property
    def description(self) -> tuple[StudyName, EvaluationMethod]:
        return (self.study_name, self.evaluation_method)


@dataclass(frozen=True, slots=True)
class DatasetSplit:
    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.DataFrame
    y_test: pd.DataFrame

    @property
    def full(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        return (self.x_train, self.x_test, self.y_train, self.y_test)

    @property
    def train_set(
        self,
    ) -> pd.DataFrame:
        return pd.concat([self.x_train, self.y_train], axis='columns')


@dataclass(frozen=True, slots=True)
class RunBaseline:
    inputs: pd.DataFrame
    outputs: pd.DataFrame
    split: DatasetSplit
    architecture: FnnArchitecture

    @property
    def dataset(
        self,
    ) -> pd.DataFrame:
        return pd.concat([self.inputs, self.outputs], axis='columns')


@dataclass(frozen=True, slots=True)
class RunConfig:
    baseline: RunBaseline
    studies: tuple[Study, ...]
    n_iter: int
    test_size: float
    test_metrics: frozenset[MetricName]
    random_state: int | None


def _build_run_config(
    inputs: pd.DataFrame,
    outputs: pd.DataFrame,
    studies: tuple[Study, ...] | None = None,
    base_architecture: tuple[int, ...] = (64, 64, 64, 64, 64, 64, 64, 64),
    n_iter: int = 100,
    test_size: float = 0.2,
    test_metrics: frozenset[MetricName] = frozenset(
        (MetricName.MSE, MetricName.R2)
    ),
    random_state: int | None = None
) -> RunConfig:
    split = train_test_split(
        inputs,
        outputs,
        test_size=test_size,
        random_state=random_state
    )

    baseline = RunBaseline(
        inputs=inputs,
        outputs=outputs,
        split=DatasetSplit(*split),
        architecture=FnnArchitecture(base_architecture)
    )

    if studies is None:
        default_studies: list[Study] = []
        default_studies.append(
            Study(
                StudyName.MODEL,
                EvaluationMethod.AVERAGING,
                FnnArchitectureGeneratorConfig(
                    range_architectures=DEFAULT_RANDOM_CONFIG,
                    taper_architectures=DEFAULT_TAPER_CONFIG
                )
            )
        )
        default_studies.append(
            Study(
                StudyName.SAMPLING,
                EvaluationMethod.AVERAGING,
                SamplingGeneratorConfig(baseline.dataset)
            )
        )
        default_studies.append(
            Study(
                StudyName.DATA,
                EvaluationMethod.AVERAGING,
                NoiseGeneratorConfig(baseline.dataset)
            )
        )
        default_studies.append(
            Study(
                StudyName.MODEL,
                EvaluationMethod.POINTWISE,
                FnnArchitectureGeneratorConfig(
                    range_architectures=DEFAULT_RANDOM_CONFIG,
                    taper_architectures=DEFAULT_TAPER_CONFIG
                )
            )
        )
        default_studies.append(
            Study(
                StudyName.SAMPLING,
                EvaluationMethod.POINTWISE,
                SamplingGeneratorConfig(baseline.split.train_set)
            )
        )
        default_studies.append(
            Study(
                StudyName.DATA,
                EvaluationMethod.POINTWISE,
                NoiseGeneratorConfig(baseline.split.train_set)
            )
        )
        studies = tuple(default_studies)

    return RunConfig(
        baseline=baseline,
        studies=studies,
        n_iter=n_iter,
        test_size=test_size,
        test_metrics=test_metrics,
        random_state=random_state
    )