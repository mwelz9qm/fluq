from enum import StrEnum

import numpy as np
import torch
from torch import nn


class MetricName(StrEnum):
    """Supported model evaluation metrics."""

    RMSE = "rmse"
    MSE = "mse"
    MAE = "mae"
    R2 = "r2"

DEFAULT_METRICS: tuple[str, ...] = (
    'rmse',
    'r2',
    'mse',
    'mae'
)


class EvaluationMethod(StrEnum):
    AVERAGING = 'averaging'
    POINTWISE = 'pointwise'


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

    def _calculate_averaging(self, run_id: str) -> None:
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
        None
        '''
        variation_groups = self.result_store.get_variation_groups(run_id)
        for group_id in variation_groups:
            models = self.result_store.get_models(group_id)
            if models is None:
                f'No found models from group_id: {group_id}.' # NOTE: Validation could happen in ResultStore implementation?

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

            self.result_store.insert_averaging_bias_and_variance(
                group_id,
                strategy_bias,
                strategy_variance
            )

    def _calculate_pointwise(self, run_id: str) -> None:
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
        None
        '''
        variation_groups = self.result_store.get_variation_groups(run_id)
        for group_id in variation_groups:
            y_test = self.result_store.get_y_test(group_id)
            if y_test is None:
                raise ValueError(
                    f'No shared y_test from group_id: {group_id}.'
                )

            variances = []
            squared_errors = []
            for actual_id in y_test:
                predictions = self.result_store.get_predictions_by_actual(actual_id)
                point_variance = np.var(predictions)
                point_mean = np.mean(predictions)

                test_point = self.result_store.get_actual(actual_id)
                point_squared_error = (point_mean - test_point) ** 2

                variances.append(point_variance)
                squared_errors.append(point_squared_error)

            strategy_variance = np.mean(variances)
            strategy_bias = np.mean(squared_errors)

            self.result_store.insert_pointwise_bias_and_variance(
                group_id,
                strategy_bias,
                strategy_variance
            )


    def evaluate(
        self,
        methods: frozenset[EvaluationMethod],
        run_id: str,
    ) -> None:
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
        None
        '''
        for method in methods:
            match(method):
                case EvaluationMethod.AVERAGING:
                    self._calculate_averaging(run_id)

                case EvaluationMethod.POINTWISE:
                    self._calculate_pointwise(run_id)

                case _:
                    raise ValueError(
                        f'Invalid evaluation method: {method!r}.'
                    )

def get_model_predictions_and_test_loss(
    model: nn.Module,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    resolved_device: torch.device,
    loss: str = 'mse',
) -> tuple[np.ndarray, float]:
    model.eval()

    with torch.inference_mode():
        predictions = model(
            x_test.to(resolved_device)
        )
        test_loss = _build_loss(loss)(
            predictions,
            y_test.to(resolved_device)
        )

    return predictions.cpu().numpy(), float(test_loss.item())

def get_model_scores(
    predictions:  np.ndarray,
    y_test: np.ndarray,
    metrics: list[MetricName]
) -> dict[str, float]:
    pass

def _build_loss(loss: str) -> nn.Module:
    losses: dict[str, type[nn.Module]] = {
        'mse': nn.MSELoss,
        'mae': nn.L1Loss,
    }

    try:
        loss_type = losses[loss]

    except KeyError:
        raise ValueError(
            f'Unsupported loss: {loss!r}.'
            f'Expected one of {sorted(losses)}.'
        ) from None
    
    return loss_type()
