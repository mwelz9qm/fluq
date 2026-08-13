import pandas as pd

from bias_variance.config import RunConfigBuilder
from bias_variance.models.evaluation import EvaluationMethod, MetricName


def _builder() -> RunConfigBuilder:
    return (
        RunConfigBuilder()
        .set_X(pd.DataFrame({'x': [0.0, 1.0]}))
        .set_Y(pd.DataFrame({'y': [0.0, 1.0]}))
    )


def test_build_adds_mse_for_averaging_evaluation() -> None:
    config = (
        _builder()
        .set_test_metrics((MetricName.R2,))
        .set_evaluation_methods((EvaluationMethod.AVERAGING,))
        .build()
    )

    assert config.test_metrics == frozenset({MetricName.MSE, MetricName.R2})


def test_build_does_not_add_mse_without_averaging_evaluation() -> None:
    config = (
        _builder()
        .set_evaluation_methods((EvaluationMethod.POINTWISE,))
        .set_test_metrics((MetricName.R2,))
        .build()
    )

    assert config.test_metrics == frozenset({MetricName.R2})
