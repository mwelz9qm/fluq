from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import torch
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)
from torch import nn

from bias_variance.persistence.store import ResultStore


class MetricName(StrEnum):
    """Supported model evaluation metrics."""

    RMSE = "rmse"
    MSE = "mse"
    MAE = "mae"
    R2 = "r2"

DEFAULT_METRICS: frozenset[MetricName] = (
    MetricName.RMSE,
    MetricName.MSE,
    MetricName.MAE,
    MetricName.R2,
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
    def data_row(self) -> tuple[str, tuple[float, ...], tuple[float, ...]]:
        return (self.group_id, self.bias, self.variance)


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

    def _evaluate_strategy_bias_and_variance(self, group_id: int) -> GroupUpdateData:
        method = self.result_store.get_method(group_id)
        match method:
            case EvaluationMethod.AVERAGING.value:
                items = self.result_store.get_models(group_id)
            case EvaluationMethod.POINTWISE.value:
                items = self.result_store.get_test_set_positions(group_id)
            case _:
                raise ValueError(
                    f'Unknown method: {method!r}.'
                )

        variances: list[np.ndarray] = []
        squared_errors: list[np.ndarray] = []
        for item in items:
            if method == EvaluationMethod.AVERAGING.value:
                actuals, predictions = (
                    self.result_store.get_actuals_and_predictions(model_id=item)
                )
            else:
                actuals, predictions = (
                    self.result_store.get_actuals_and_predictions(
                        group_id_and_set_pos=(group_id, item)
                    )
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

            if method == EvaluationMethod.POINTWISE.value:
                if not np.allclose(actuals_array, actuals_array[0]):
                    raise ValueError(
                        'Inconsistent actual outputs for '
                        f'group {group_id}, position {item}.'
                    )
                mean_error = (
                    np.mean(predictions_array, axis=0) - actuals_array[0]
                )
            else:
                mean_error = (
                    np.mean(predictions_array, axis=0)
                    - np.mean(actuals_array, axis=0)
                )

            variance = np.var(predictions_array, axis=0)
            squared_error = mean_error ** 2.0

            variances.append(variance)
            squared_errors.append(squared_error)

        if not variances:
            raise ValueError(f'No evaluation data found for group {group_id}.')

        strategy_variance = np.mean(np.stack(variances), axis=0)
        strategy_bias = np.mean(np.stack(squared_errors), axis=0)

        return GroupUpdateData(
            group_id,
            tuple(float(value) for value in strategy_bias),
            tuple(float(value) for value in strategy_variance),
        )

    def evaluate(
        self,
        run_id: str,
    ) -> tuple[GroupUpdateData]:
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
        group_results = []
        for study_id in studies:
            groups = self.result_store.get_groups(study_id)
            for group_id in groups:
                group_update = self._evaluate_strategy_bias_and_variance(group_id)
                group_results.append(group_update)

        return tuple(group_results)


########## HELPER FUNCTIONS ##########

def get_model_predictions(
    model: nn.Module,
    x_test: torch.Tensor | np.ndarray,
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
    x_test : torch.Tensor | np.ndarray
        Test inputs used for prediction.
    resolved_device : torch.device
        Device where the test inputs should be placed for prediction.

    Returns
    -------
    np.ndarray
        Model predictions returned as a NumPy array.
    """
    if not isinstance(x_test, torch.Tensor):
        x_test = torch.from_numpy(x_test)

    model.eval()

    with torch.inference_mode():
        predictions = model(
            x_test.to(resolved_device)
        )

    return predictions.cpu().numpy()

def get_model_scores(
    predictions:  np.ndarray,
    y_test: np.ndarray,
    metrics: frozenset[MetricName]
) -> dict[str, float]:
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
