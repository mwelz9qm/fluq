import pytest

from bias_variance.models.evaluation import Evaluator, GroupUpdateData


class FakeResultStore:
    def __init__(self, method, items, results):
        self.method = method
        self.items = items
        self.results = results

    def get_method(self, group_id):
        return self.method

    def get_models(self, group_id):
        return self.items

    def get_test_set_positions(self, group_id):
        return self.items

    def get_actuals_and_predictions(
        self,
        model_id=None,
        group_id_and_set_pos=None,
    ):
        key = model_id if model_id is not None else group_id_and_set_pos[1]
        return self.results[key]


def test_averaging_preserves_one_result_per_output() -> None:
    store = FakeResultStore(
        method='averaging',
        items=(1, 2),
        results={
            1: (
                ((1.0, 10.0), (3.0, 14.0)),
                ((2.0, 12.0), (4.0, 16.0)),
            ),
            2: (
                ((1.0, 10.0), (3.0, 14.0)),
                ((0.0, 9.0), (2.0, 13.0)),
            ),
        },
    )

    result = Evaluator(store)._evaluate_strategy_bias_and_variance(7)

    assert result == GroupUpdateData(
        group_id=7,
        bias=(1.0, 2.5),
        variance=(1.0, 4.0),
    )


def test_pointwise_preserves_one_result_per_output() -> None:
    store = FakeResultStore(
        method='pointwise',
        items=(0, 1),
        results={
            0: (
                ((1.0, 10.0), (1.0, 10.0)),
                ((2.0, 12.0), (4.0, 14.0)),
            ),
            1: (
                ((3.0, 20.0), (3.0, 20.0)),
                ((3.0, 18.0), (5.0, 22.0)),
            ),
        },
    )

    result = Evaluator(store)._evaluate_strategy_bias_and_variance(7)

    assert result == GroupUpdateData(
        group_id=7,
        bias=(2.5, 4.5),
        variance=(1.0, 2.5),
    )


def test_pointwise_rejects_inconsistent_actual_outputs() -> None:
    store = FakeResultStore(
        method='pointwise',
        items=(0,),
        results={
            0: (
                ((1.0, 10.0), (2.0, 10.0)),
                ((2.0, 12.0), (4.0, 14.0)),
            ),
        },
    )

    with pytest.raises(
        ValueError,
        match='Inconsistent actual outputs for group 7, position 0',
    ):
        Evaluator(store)._evaluate_strategy_bias_and_variance(7)
