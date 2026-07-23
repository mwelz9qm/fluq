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
VARIATIONS_FILENAME = "bias_variance_variations.csv"
EVALUATIONS_FILENAME = "bias_variance_evaluations.csv"

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

RUN_TYPE_FIELD_NAME = "run_type"
LOSS_FUNC_FIELD_NAME = "loss_func"

POINTWISE_MEAN_FIELD_NAME = "pointwise_mean"
POINTWISE_VARIANCE_FIELD_NAME = "pointwise_variance"
AVERAGING_MEAN_FIELD_NAME = "averaging_mean"
AVERAGING_VARIANCE_FIELD_NAME = "averaging_variance"

VARIATION_ID_FIELD_NAME = "variation_id"
VARIATION_RANDOM_STATE_FIELD_NAME = "variation_random_state"
VARIATION_LABEL_FIELD_NAME = "variation_label"
STUDY_LABEL_FIELD_NAME = "study_label"

PREDICTION_CI_FIELD_NAME = "prediction_ci"
PREDICTION_CI_INTERVAL_LOWER_FIELD_NAME = "prediction_ci_interval_lower"
PREDICTION_CI_INTERVAL_UPPER_FIELD_NAME = "prediction_ci_interval_upper"
PREDICTION_MEAN_FIELD_NAME = "prediction_mean"
PREDICTION_VARIANCE_FIELD_NAME = "prediction_variance"

EVALUATION_ID_FIELD_NAME = "evaluation_id"
EVALUATION_RANDOM_STATE_FIELD_NAME = "evaluation_random_state"
PREDICTION_FIELD_NAME = "prediction"
Y_TEST_POINT_FIELD_NAME = "y_test_point"
X_TEST_POINT_FIELD_NAME = "x_test_point"

PREDICTIONS_DATASET_NAME = "predictions"
ACTUALS_DATASET_NAME = "actuals"
PREDICTIONS_LAYER_NAME = "predictions"
MODEL_NAME = "functional_model"

RUNS_FILENAME = "bias_variance_runs.csv"
RUN_METADATA_FILENAME = RUNS_FILENAME

CREATED_AT_FIELD_NAME = "created_at"
RANDOM_STATE_FIELD_NAME = "random_state"
TEST_SIZE_FIELD_NAME = "test_size"
N_ITER_FIELD_NAME = "n_iter"

OPTIMIZER_FIELD_NAME = "optimizer"
LEARNING_RATE_FIELD_NAME = "learning_rate"
METRICS_FIELD_NAME = "metrics"
EPOCHS_FIELD_NAME = "epochs"
BATCH_SIZE_FIELD_NAME = "batch_size"
DEVICE_FIELD_NAME = "device"

BASELINE_ARCHITECTURE_FIELD_NAME = "baseline_architecture"
ARCHITECTURE_FIELD_NAME = "architecture"
MODEL_SEED_FIELD_NAME = "model_seed"
