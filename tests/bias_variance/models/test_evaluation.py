import numpy as np
import pandas as pd
import pytest

from bias_variance.models.evaluation import (
    Evaluator,
    GroupUpdateData,
    MetricName,
    get_model_scores,
)
from bias_variance.persistence.store import StoredTestPointPrediction


class FakeResultStore:
    def __init__(self, pointwise_rows=(), model_ids=(), averaging_data=None):
        self.pointwise_rows = pointwise_rows
        self.model_ids = model_ids
        self.averaging_data = averaging_data

    def get_pointwise_evaluation_data(self, group_id):
        return self.pointwise_rows

    def get_models(self, group_id):
        return self.model_ids

    def get_averaging_evaluation_data(self, group_id):
        return self.averaging_data


def test_averaging_preserves_one_result_per_output() -> None:
    store = FakeResultStore(
        averaging_data=(
            ((1.0, 4.0), (1.0, 1.0)),
            ((1.0, 4.0), (1.0, 4.0)),
        )
    )

    result = Evaluator(store)._evaluate_averaging(7)

    assert result == GroupUpdateData(
        group_id=7,
        bias=(1.0, 2.5),
        variance=(1.0, 4.0),
    )


def test_averaging_rejects_mismatched_model_output_shapes() -> None:
    store = FakeResultStore(
        averaging_data=(
            ((1.0, 4.0),),
            ((1.0,),),
        )
    )

    with pytest.raises(ValueError, match='matching shape'):
        Evaluator(store)._evaluate_averaging(7)


def test_pointwise_preserves_one_result_per_output() -> None:
    store = FakeResultStore(
        model_ids=(1, 2),
        pointwise_rows=(
            StoredTestPointPrediction(1, 0, (0.0,), (1.0, 10.0), (2.0, 12.0)),
            StoredTestPointPrediction(2, 0, (0.0,), (1.0, 10.0), (0.0, 9.0)),
            StoredTestPointPrediction(1, 1, (1.0,), (3.0, 20.0), (3.0, 18.0)),
            StoredTestPointPrediction(2, 1, (1.0,), (3.0, 20.0), (5.0, 22.0)),
        ),
    )

    update, records = Evaluator(store)._evaluate_pointwise(7)

    assert update == GroupUpdateData(
        group_id=7,
        bias=(0.5, 0.125),
        variance=(1.0, 3.125),
    )
    assert tuple(record.y_true for record in records) == (
        (1.0, 10.0),
        (3.0, 20.0),
    )


def test_pointwise_rejects_mismatched_output_shapes() -> None:
    store = FakeResultStore(
        model_ids=(1, 2),
        pointwise_rows=(
            StoredTestPointPrediction(1, 0, (0.0,), (1.0, 10.0), (2.0,)),
            StoredTestPointPrediction(2, 0, (0.0,), (1.0, 10.0), (4.0,)),
        ),
    )

    with pytest.raises(ValueError, match='matching shapes'):
        Evaluator(store)._evaluate_pointwise(7)


def test_pointwise_rejects_empty_position_set() -> None:
    with pytest.raises(ValueError, match='No evaluation data found'):
        Evaluator(FakeResultStore())._evaluate_pointwise(7)


def test_get_model_scores_returns_uniform_metrics() -> None:
    predictions = np.array([[1.0], [3.0], [5.0]])
    y_test = pd.DataFrame({'y': [1.0, 2.0, 7.0]})

    scores = get_model_scores(
        predictions=predictions,
        y_test=y_test,
        metrics=frozenset(
            {
                MetricName.RMSE,
                MetricName.MSE,
                MetricName.MAE,
                MetricName.R2,
            }
        ),
    )

    assert set(scores) == {'rmse', 'mse', 'mae', 'r2'}
    assert scores['mse'] == pytest.approx(5.0 / 3.0)
    assert scores['rmse'] == pytest.approx((5.0 / 3.0) ** 0.5)
    assert scores['mae'] == pytest.approx(1.0)
    assert scores['r2'] == pytest.approx(141.0 / 186.0)


def test_get_model_scores_returns_raw_values_per_output() -> None:
    predictions = np.array([[2.0, 12.0], [4.0, 16.0]])
    y_test = pd.DataFrame([[1.0, 10.0], [3.0, 14.0]])

    scores = get_model_scores(
        predictions=predictions,
        y_test=y_test,
        metrics=frozenset({MetricName.MSE}),
        is_uniform=False,
    )

    assert scores == {'mse': (1.0, 4.0)}
