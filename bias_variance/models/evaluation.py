from dataclasses import dataclass
from enum import StrEnum
from typing import Literal, overload

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)
from torch import nn

from bias_variance.persistence.records import EvaluationRecord
from bias_variance.persistence.store import ResultStore, StoredTestPointPrediction


class MetricName(StrEnum):

    RMSE = "rmse"
    MSE = "mse"
    MAE = "mae"
    R2 = "r2"

DEFAULT_METRICS: frozenset[MetricName] = frozenset(
    (
        MetricName.RMSE,
        MetricName.MSE,
        MetricName.MAE,
        MetricName.R2,
    )
)


class EvaluationMethod(StrEnum):
    AVERAGING = 'averaging'
    POINTWISE = 'pointwise'


@dataclass(frozen=True, slots=True)
class GroupUpdateData:
    group_id: int
    bias: tuple[float, ...]
    variance: tuple[float, ...]

    @property
    def data_row(self) -> tuple[int, tuple[float, ...], tuple[float, ...]]:
        return (self.group_id, self.bias, self.variance)


@dataclass(frozen=True, slots=True)
class EvaluationData:
    update_groups: tuple[GroupUpdateData]
    evaluations: tuple[EvaluationRecord]


class Evaluator:
    def __init__(
        self,
        result_store: ResultStore,
    ):
        self.result_store = result_store

    def _evaluate_averaging(
        self,
        group_id: int,
    ) -> GroupUpdateData:

        mse_scores, model_variances = self.result_store.get_averaging_evaluation_data(group_id)
        mse_array = np.asarray(mse_scores, dtype=float)
        variance_array = np.asarray(model_variances, dtype=float)
        if (
            mse_array.ndim != 2
            or variance_array.ndim != 2
            or mse_array.shape != variance_array.shape
            or mse_array.size == 0
        ):
            raise ValueError(
                'Averaging MSE scores and variances must have matching shape '
                '(models, outputs).'
            )

        return GroupUpdateData(
            group_id=group_id,
            bias=tuple(float(value) for value in mse_array.mean(axis=0)),
            variance=tuple(
                float(value) for value in variance_array.mean(axis=0)
            ),
        )

    def _evaluate_pointwise(
        self,
        group_id: int,
    ) -> tuple[GroupUpdateData, tuple[EvaluationRecord, ...]]:
        records: list[EvaluationRecord] = []
        rows_by_position: dict[int, dict[int, StoredTestPointPrediction]] = {}

        for row in self.result_store.get_pointwise_evaluation_data(group_id):
            position_rows = rows_by_position.setdefault(row.set_position, {})

            if row.model_id in position_rows:
                raise ValueError(
                    f'Duplicate results for model {row.model_id}, '
                    f'position {row.set_position}.'
                )

            position_rows[row.model_id] = row

        expected_model_ids = set(self.result_store.get_models(group_id))

        for position, model_rows in rows_by_position.items():
            if set(model_rows) != expected_model_ids:
                missing = expected_model_ids - set(model_rows)
                raise ValueError(
                    f'Position {position} is missing models: {sorted(missing)}.'
                )

            ordered_rows = [model_rows[mid] for mid in sorted(expected_model_ids)]
            reference = ordered_rows[0]

            for row in ordered_rows[1:]:
                if row.input != reference.input:
                    raise ValueError(
                        f'Models use different test inputs at position {position}.'
                    )
                if row.output != reference.output:
                    raise ValueError(
                        f'Models use different actual outputs at position {position}.'
                    )

            actual = np.asarray(reference.output, dtype=float)
            predictions = np.asarray(
                [row.prediction for row in ordered_rows],
                dtype=float,
            )

            if actual.ndim != 1:
                raise ValueError('Actual output must be one-dimensional.')

            if predictions.ndim != 2:
                raise ValueError('Predictions must be two-dimensional.')

            if actual.shape != predictions.shape[1:]:
                raise ValueError(
                    'Actuals and predictions must have matching shapes; '
                    f'got {actual.shape} and {predictions.shape[1:]}.'
                )

            point_mean = predictions.mean(axis=0)
            squared_bias = (point_mean - actual) ** 2
            variance = predictions.var(axis=0)

            records.append(
                EvaluationRecord(
                    group_id=group_id,
                    test_set_position=position,
                    y_true=tuple(float(value) for value in actual),
                    point_mean_prediction=tuple(
                        float(value) for value in point_mean
                    ),
                    bias=tuple(float(value) for value in squared_bias),
                    variance=tuple(float(value) for value in variance),
                )
            )

        if not records:
            raise ValueError(f'No evaluation data found for group {group_id}.')

        return (
            GroupUpdateData(
                group_id=group_id,
                bias=tuple(
                    float(value)
                    for value in np.mean([record.bias for record in records], axis=0)
                ),
                variance=tuple(
                    float(value)
                    for value in np.mean(
                        [record.variance for record in records], axis=0
                    )
                ),
            ),
            tuple(records),
        )

    def evaluate(
        self,
        run_id: str,
    ) -> EvaluationData:
        studies = self.result_store.get_studies(run_id)
        update_groups: list[GroupUpdateData] = []
        evaluation_records: list[EvaluationRecord] = []
        for study_id in studies:
            groups = self.result_store.get_groups(study_id)
            for group_id in groups:
                match self.result_store.get_method(group_id):
                    case EvaluationMethod.AVERAGING.value:
                        group = self._evaluate_averaging(group_id)
                        records = ()
                    case EvaluationMethod.POINTWISE.value:
                        group, records = self._evaluate_pointwise(group_id)
                    case method:
                        raise ValueError(f'Unknown method: {method!r}.')

                update_groups.append(group)
                evaluation_records.extend(records)

        return EvaluationData(
            update_groups=tuple(update_groups),
            evaluations=tuple(evaluation_records)
        )


########## HELPER FUNCTIONS ##########

def get_model_predictions(
    model: nn.Module,
    x_test: torch.Tensor | pd.DataFrame,
    resolved_device: torch.device,
) -> np.ndarray:
    if not isinstance(x_test, torch.Tensor):
        x_test = torch.as_tensor(
            x_test.to_numpy(dtype=np.float32, copy=True),
            dtype=torch.float32,
        )
    else:
        x_test = x_test.to(dtype=torch.float32)

    model.eval()

    with torch.inference_mode():
        predictions = model(
            x_test.to(resolved_device)
        )

    return predictions.cpu().numpy()

@overload
def get_model_scores(
    predictions: np.ndarray,
    y_test: torch.Tensor | pd.DataFrame,
    metrics: frozenset[MetricName],
    is_uniform: Literal[True] = True,
) -> dict[str, float]:
    ...


@overload
def get_model_scores(
    predictions: np.ndarray,
    y_test: torch.Tensor | pd.DataFrame,
    metrics: frozenset[MetricName],
    is_uniform: Literal[False],
) -> dict[str, tuple[float, ...]]:
    ...


def get_model_scores(
    predictions: np.ndarray,
    y_test: torch.Tensor | pd.DataFrame,
    metrics: frozenset[MetricName],
    is_uniform: bool = True,
) -> dict[str, float | tuple[float, ...]]:
    if isinstance(y_test, torch.Tensor):
        y_test = y_test.detach().cpu().numpy()

    scores: dict[str, float | tuple[float, ...]] = {}
    multioutput = 'uniform_average' if is_uniform else 'raw_values'
    for metric in metrics:
        match(metric):
            case MetricName.RMSE:
                score = root_mean_squared_error(
                    y_test,
                    predictions,
                    multioutput=multioutput,
                )

            case MetricName.MSE:
                score = mean_squared_error(
                    y_test,
                    predictions,
                    multioutput=multioutput,
                )

            case MetricName.MAE:
                score = mean_absolute_error(
                    y_test,
                    predictions,
                    multioutput=multioutput,
                )

            case MetricName.R2:
                score = r2_score(
                    y_test,
                    predictions,
                    multioutput=multioutput,
                )

            case _:
                raise ValueError(
                    f'metrics contains unknown metric: {metric!r}.'
                )

        if is_uniform:
            scores[metric.value] = float(score)
        else:
            raw_score = np.asarray(score, dtype=float).reshape(-1)
            scores[metric.value] = tuple(
                float(value)
                for value in raw_score
            )

    return scores
