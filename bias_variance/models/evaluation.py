from dataclasses import dataclass
from enum import StrEnum

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
from bias_variance.persistence.store import ResultStore


class MetricName(StrEnum):
    """Supported model evaluation metrics."""

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
    '''
    Evaluates the run data across all models under a variation and study.

    Uses the result store to query data for evaluation. For a given
    evaluation method (i.e. averaging or pointwise), the Evaluator
    decomposes the bias and variance based on the method or methods and
    returns the overall values.
    
    Note that the Evaluator is not responsible for building the variation
    record after getting the results.

    Attributes
    -------------
    result_store: ResultStore
        The interface for inserting and accessing data from the cache tables.
    '''
    def __init__(
        self,
        result_store: ResultStore,
    ):
        self.result_store = result_store

    def _evaluate_averaging(
        self,
        run_id: str,
        group_id: int,
    ) -> GroupUpdateData:
        """Average stored per-model MSE and prediction variance by output."""
        mse_scores, model_variances = (
            self.result_store.get_averaging_evaluation_data(run_id, group_id)
        )
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
        """Evaluate and record predictions at each shared test position."""
        records: list[EvaluationRecord] = []
        for position in self.result_store.get_test_set_positions(group_id):
            actuals, predictions = self.result_store.get_actuals_and_predictions(
                group_id_and_set_pos=(group_id, position)
            )

            actuals_array = np.asarray(actuals, dtype=float)
            predictions_array = np.asarray(predictions, dtype=float)

            if actuals_array.ndim != 2 or predictions_array.ndim != 2:
                raise ValueError(
                    'Actuals and predictions must each have shape '
                    '(observations, outputs).'
                )
            if actuals_array.shape != predictions_array.shape:
                raise ValueError(
                    'Actuals and predictions must have matching shapes; '
                    f'got {actuals_array.shape} and {predictions_array.shape}.'
                )
            if not np.allclose(actuals_array, actuals_array[0]):
                raise ValueError(
                    'Inconsistent actual outputs for '
                    f'group {group_id}, position {position}.'
                )
            point_mean = np.mean(predictions_array, axis=0)
            squared_bias = (point_mean - actuals_array[0]) ** 2.0
            variance = np.var(predictions_array, axis=0)
            records.append(
                EvaluationRecord(
                    group_id=group_id,
                    test_set_position=position,
                    y_true=tuple(float(value) for value in actuals_array[0]),
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
        '''
        Evaluates the biases and variances for all variation groups in a study.
        There are two types of bias and variance pairs:
        - averaging
        - pointwise

        The method will insert the results based on the selected evaluation
        methods.

        Parameters
        -----------
        run_id: str
            The run identifier used to query results in the cache tables.

        Returns
        ------------
        tuple[GroupUpdateData]
        '''
        studies = self.result_store.get_studies(run_id)
        update_groups: list[GroupUpdateData] = []
        evaluation_records: list[EvaluationRecord] = []
        for study_id in studies:
            groups = self.result_store.get_groups(study_id)
            for group_id in groups:
                match self.result_store.get_method(group_id):
                    case EvaluationMethod.AVERAGING.value:
                        group = self._evaluate_averaging(run_id, group_id)
                        records = ()
                    case EvaluationMethod.POINTWISE.value:
                        group, records = self._evaluate_pointwise(group_id)
                    case method:
                        raise ValueError(f'Unknown method: {method!r}.')

                self.result_store.update_group(*group.data_row)
                for record in records:
                    self.result_store.add(record)
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
    """Generate model predictions for a test input set.

    This helper prepares the test inputs as a torch tensor when needed, places
    the inputs on the resolved device, runs the model in evaluation mode without
    tracking gradients, and returns the predictions as a NumPy array.

    Parameters
    ----------
    model : nn.Module
        Trained model used to generate predictions.
    x_test : torch.Tensor | pd.DataFrame
        Test inputs used for prediction.
    resolved_device : torch.device
        Device where the test inputs should be placed for prediction.

    Returns
    -------
    np.ndarray
        Model predictions returned as a NumPy array.
    """
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

def get_model_scores(
    predictions: np.ndarray,
    y_test: torch.Tensor | pd.DataFrame,
    metrics: frozenset[MetricName]
) -> dict[str, float]:
    if isinstance(y_test, torch.Tensor):
        y_test = y_test.detach().cpu().numpy()

    scores = {}
    for metric in metrics:
        match(metric):
            case MetricName.RMSE:
                scores[metric.value] = root_mean_squared_error(y_test, predictions)

            case MetricName.MSE:
                scores[metric.value] = mean_squared_error(y_test, predictions)

            case MetricName.MAE:
                scores[metric.value] = mean_absolute_error(y_test, predictions)

            case MetricName.R2:
                scores[metric.value] = r2_score(y_test, predictions)

            case _:
                raise ValueError(
                    f'metrics contains unknown metric: {metric!r}.'
                )
            
    return scores
