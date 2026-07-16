"""Constants and closed-choice values used by :mod:`bias_variance.BiasAnalyzer`."""

from enum import StrEnum


class MetricName(StrEnum):
    """Supported model evaluation metrics."""

    RMSE = "rmse"
    MSE = "mse"
    MAE = "mae"
    R2 = "r2"


class StudyName(StrEnum):
    """Supported bias study types."""

    MODEL = "model"
    SAMPLING = "sampling"
    DATA = "data"


class SamplingStrategyName(StrEnum):
    """Supported sampling strategies."""

    BOOTSTRAP = "bootstrap"
    STRATIFIED = "stratified"
    LHS = "lhs"


class PlotType(StrEnum):
    """Supported disagreement-map plot types."""

    VARIANCE_CONTRIBUTION = "variance_contribution"
    PREDICTION_MEANS_BY_R2_SCORES = "prediction_means_by_r2_scores"
    VARIANCE_DISTRIBUTION = "variance_distribution"
    MEAN_DISTRIBUTION = "mean_distribution"


RESULTS_FILENAME = "bias_variance_results.csv"
FIT_ITERATIONS_DIR_NAME = "iterations"

RUN_ID_FIELD_NAME = "run_id"
ITERATION_FIELD_NAME = "iteration"
STUDY_FIELD_NAME = "study"
VARIABLE_FIELD_NAME = "variable"
LOSS_FIELD_NAME = "loss"
VARIANCE_FIELD_NAME = "variance"
MEAN_FIELD_NAME = "mean"
CONF_INTERVAL_LOWER_FIELD_NAME = "conf_interval_lower"
CONF_INTERVAL_UPPER_FIELD_NAME = "conf_interval_upper"
TIMESTAMP_FIELD_NAME = "timestamp"

PREDICTIONS_DATASET_NAME = "predictions"
ACTUALS_DATASET_NAME = "actuals"
PREDICTIONS_LAYER_NAME = "predictions"
MODEL_NAME = "functional_model"
