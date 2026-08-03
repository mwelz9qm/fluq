from collections.abc import Mapping
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
    group_id: str
    bias: float
    variance: float

    @property
    def data_row(self) -> tuple[str, float, float]:
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
        result_store,
    ):
        self.result_store = result_store

    def _calculate_averaging(self, group_id: str) -> GroupUpdateData:
        '''
        Calculates the strategy bias and variance with the averaging method.
        - Bias: For a variation group in a study, the mean of predictions
        is subtracted by the sample mean corresponding to the points within
        the testing set used in model evaluation and then squared for each
        model within the variation group. The bias is the mean of these
        squared errors across all models within the group.
        - Variance: For a variaion group in a study, the variance is the
        mean of variances for all models in the group, where the variance
        is calculated per model's set of predictions.

        The method uses the run_id as an entry point to query the run data
        for evaluation. All variation group biases and variances are then
        inserted into the cache results tables.

        Parameters
        -----------
        run_id: str
            The run identifier used to query results in the cache tables.

        Returns
        ------------
        GroupUpdateData
        '''
        models = self.result_store.get_models(group_id)
        
        variances = []
        squared_errors = []
        for model_id in models:
            predictions = self.result_store.get_predictions_by_model(model_id)
            model_variance = np.var(predictions)
            model_mean = np.mean(predictions)
        
            actuals = self.result_store.get_actuals(model_id)
            sample_mean  = np.mean(actuals)
            model_squared_error = (model_mean - sample_mean) ** 2.0
        
            variances.append(model_variance)
            squared_errors.append(model_squared_error)
        
        strategy_variance = np.mean(variances)
        strategy_bias = np.mean(squared_errors)

        return GroupUpdateData(
            group_id,
            strategy_bias,
            strategy_variance
        )

    def _calculate_pointwise(self, group_id: str) -> GroupUpdateData:
        '''
        Calculates the strategy bias and variance with the pointwise method.
        - Bias: For a variation group in a study, each group has a shared set
        of test points from the baseline train-test split. For each point in
        the test points, there is a set of models that made a prediction on the
        point. Across all predictions for a given point, the mean of predictions
        is subtracted by the point, then squared. The bias is the average of all
        test point squared errors.
        - Variance: For a variation group in a study, each group has a shared
        set of test points. For each point in the test points, there is a set of
        models that made a prediction on the point. Across all predictions on a
        given point, the variance is calculated. With all point variances, the
        variance is the average point variance.

        Parameters
        -----------
        run_id: str
            The run identifier used to query results in the cache tables.
        
        Returns
        ------------
        GroupUpdateData
        '''
        tests = self.result_store.get_tests(group_id)
        
        variances = []
        squared_errors = []
        for test_id in tests:
            predictions = self.result_store.get_predictions_by_test(test_id)
            point_variance = np.var(predictions)
            point_mean = np.mean(predictions)
        
            test_point = self.result_store.get_actual(test_id)
            point_squared_error = (point_mean - test_point) ** 2
        
            variances.append(point_variance)
            squared_errors.append(point_squared_error)
        
        strategy_variance = np.mean(variances)
        strategy_bias = np.mean(squared_errors)

        return GroupUpdateData(
            group_id,
            strategy_bias,
            strategy_variance
        )

    def _evaluate_strategy_bias_and_variance(self, group_id: str) -> GroupUpdateData:
        method = self.result_store.get_method(group_id)
        match method:
            case EvaluationMethod.AVERAGING.value:
                results = self._calculate_averaging(group_id)
            case EvaluationMethod.POINTWISE.value:
                results = self._calculate_pointwise(group_id)
            case _:
                raise ValueError(
                    f'Unknown method: {method!r}.'
                )

        return results

    def evaluate(
            self,
            run_id: str,
    ) -> Mapping[str, tuple[GroupUpdateData]]:
        '''
        Evaluates the biases and variances for all variation groups in a study.
        There are two types of bias and variance pairs:
        - averaging
        - pointwise

        The method will insert the results based on the selected evaluation
        methods.

        Parameters
        -----------
        methods: frozenset[EvaluationMethod]
            The set of methods used to decompose the biases and variances.
        run_id: str
            The run identifier used to query results in the cache tables.

        Returns
        ------------
        Mapping[str, tuple[GroupUpdateData]]
        '''
        studies = self.result_store.get_studies(run_id)
        post_evaluation_results = {}
        for study_id in studies:
            groups = self.result_store.get_groups(study_id)
            group_results: list[GroupUpdateData] = []
            for group_id in groups:
                group_update = self._evaluate_strategy_bias_and_variance(group_id)
                group_results.append(group_update)

            post_evaluation_results[study_id] = tuple(group_results)

        return post_evaluation_results


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
