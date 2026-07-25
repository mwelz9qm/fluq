from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from bias_variance.generators.base import Generator
from bias_variance.models.fnn.FnnArchitecture import FnnArchitecture
from bias_variance.models.training import Trainer, TrainingConfig


class StudyName(StrEnum):
    MODEL = 'model'
    SAMPLING = 'sampling'
    DATA = 'data'


class EvaluationMethod(StrEnum):
    AVERAGING = 'averaging'
    POINTWISE = 'pointwise'


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    x_test: np.ndarray[float]
    y_test: np.ndarray[float]
    predictions: np.ndarray[float]


@dataclass(frozen=True, slots=True)
class DatasetSplit:
    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.DataFrame
    y_test: pd.DataFrame


@dataclass(frozen=True, slots=True)
class StudyBaseline:
    inputs: pd.DataFrame
    outputs: pd.DataFrame
    split: DatasetSplit
    architecture: FnnArchitecture


@dataclass(frozen=True, slots=True)
class StudyConfig:
    n_iter: int = 100
    test_size: float = 0.2
    random_state: int | None = None
    baseline: StudyBaseline
    evaluation_method: frozenset[EvaluationMethod] = {EvaluationMethod.AVERAGING, EvaluationMethod.POINTWISE}
    studies: frozenset[StudyName] = {StudyName.MODEL, StudyName.SAMPLING, StudyName.DATA}


class BiasAnalyzer:
    def __init__(
        self,
        training_config: TrainingConfig,
        result_store,
    ):
        self.result_store = result_store
        self.trainer = Trainer(training_config)

    def _decompose_bias_and_variance(
        self,
    ) -> None:
        pass

    def _build_generator(
        self,
        study_name: StudyName,
        evaluation_method: EvaluationMethod,
        test_size: float,
        baseline: StudyBaseline,
    ) -> Generator[FnnArchitecture] | Generator[pd.DataFrame]:
        pass

    def _run_bias_study(
        self,
        study_name: StudyName,
        evaluation_method: EvaluationMethod,
        n_iter: int,
        test_size: float,
        random_state: int | None,
        baseline: StudyBaseline,
    ) -> None:
        for i in np.arange(n_iter):
            generator = self._build_generator(study_name, evaluation_method, test_size, baseline)
            variations = generator.generate(random_state=random_state)
            for j, variation in enumerate(variations):
                if evaluation_method == EvaluationMethod.POINTWISE:
                    if study_name == StudyName.MODEL:
                        trained_model = self.trainer.train(
                            architecture=variation.generated,
                            x_train=baseline.split.x_train,
                            y_train=baseline.split.y_train,
                            random_state=random_state + j
                        )
                        predictions = self.trainer.predict(trained_model, baseline.split.x_test)
                        
                    else:
                        trained_model = self.trainer.train(
                            architecture=baseline.architecture,
                            x_train=variation.generated[baseline.inputs.columns],
                            y_train=variation.generated[baseline.outputs.columns],
                            random_state=random_state + j
                        )
                        predictions = self.trainer.predict(trained_model, baseline.split.x_test)

                    evaluation_result = EvaluationResult(
                        x_test=baseline.split.x_test,
                        y_test=baseline.split.y_test,
                        predictions=predictions,
                    )

                else:
                    if study_name == StudyName.MODEL:
                        x_train, x_test, y_train, y_test = train_test_split(
                            baseline.inputs,
                            baseline.outputs,
                            test_size=test_size,
                            random_state=random_state
                        )
                        trained_model = self.trainer.train(
                            architecture=variation.generated,
                            x_train=x_train,
                            y_train=y_train,
                            random_state=random_state + j
                        )
                        predictions = self.trainer.predict(trained_model, x_test)

                    else:
                        x_train, x_test, y_train, y_test = train_test_split(
                            variation.generated[baseline.inputs.columns],
                            variation.generated[baseline.outputs.columns],
                            test_size=test_size,
                            random_state=random_state
                        )
                        trained_model = self.trainer.train(
                            architecture=baseline.architecture,
                            x_train=variation.generated[baseline.inputs.columns],
                            y_train=variation.generated[baseline.outputs.columns],
                            random_state=random_state + j
                        )
                        predictions = self.trainer.predict(trained_model, x_test)

                    evaluation_result = EvaluationResult(
                        x_test=x_test,
                        y_test=y_test,
                        predictions=predictions,
                    )

                self.result_store.add(evaluation_result)

            variation_result = {}
            self.result_store.add(variation_result)

    def run_bias_studies(
        self,
        study_config: StudyConfig,
    ) -> None:
        self.trainer.set_model_builder(
            study_config.baseline.inputs.shape[1],
            study_config.baseline.outputs.shape[1]
        )
        
        for study_name in study_config.studies:
            self._run_bias_study(
                study_name,
                study_config.evaluation_method,
                study_config.n_iter,
                study_config.test_size,
                study_config.random_state,
                study_config.baseline
            )
            self._decompose_bias_and_variance()

        run_result = {}
        self.result_store.add(run_result)
        self.result_store.commit()
