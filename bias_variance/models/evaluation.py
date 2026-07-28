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
    '''
    def __init__(
        self,
        result_store,
    ):
        self.result_store = result_store

    def _decompose_bias_and_variance(self):
        pass

    def evaluate(
        self,
        method: frozenset[EvaluationMethod],
        run_id: str,
    ) -> tuple[float, float]:
        pass

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
