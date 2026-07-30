from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class RunRecord:
    run_id: str
    created_at: datetime
    n_iter: int
    test_size: float
    test_metrics: tuple[str, ...]
    optimizer: str
    learning_rate: float
    loss: str
    epochs: int
    batch_size: int
    device: str
    base_architecture: tuple[int, ...]
    base_x_train: tuple[float]
    base_y_train: tuple[float]
    base_x_test: tuple[float]
    base_y_test: tuple[float]
    input_columns: tuple[str, ...]
    output_columns: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class StudyRecord:
    study_id: str
    run_id: str
    study_name: str
    evaluation_method: str


@dataclass(frozen=True, slots=True)
class GroupRecord:
    group_id: str
    study_id: str
    group_name: str
    averaging_strategy_bias: float | None
    averaging_strategy_variance: float | None
    pointwise_strategy_bias: float | None
    pointwise_strategy_variance: float | None


@dataclass(frozen=True, slots=True)
class ModelRecord:
    model_id: str
    group_id: str
    architecture: tuple[int, ...]
    test_scores: Mapping[str, float]
    x_train: tuple[float, ...]
    y_train: tuple[float, ...]
    x_test: tuple[float, ...]
    y_test: tuple[float, ...]
    predictions: tuple[float, ...]