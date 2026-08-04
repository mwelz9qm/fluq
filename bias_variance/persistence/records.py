from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class RunRecord:
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
    input_columns: tuple[str, ...]
    output_columns: tuple[str, ...]
    base_architecture: tuple[int, ...]

    run_id: str # primary
    base_train_set_id: int | None = None # foreign
    base_test_set_id: int | None = None # foreign


@dataclass(frozen=True, slots=True)
class StudyRecord:
    study_name: str
    evaluation_method: str

    study_id: int | None = None # primary
    run_id: str | None = None # foreign


@dataclass(frozen=True, slots=True)
class GroupRecord:
    group_name: str
    averaging_strategy_bias: float | None
    averaging_strategy_variance: float | None
    pointwise_strategy_bias: float | None
    pointwise_strategy_variance: float | None

    group_id: int | None = None # primary
    study_id: int | None = None # foreign


@dataclass(frozen=True, slots=True)
class ModelRecord:
    architecture: tuple[int, ...]
    test_scores: Mapping[str, float]

    model_id: int | None = None # primary
    group_id: int | None = None # foreign
    train_set_id: int | None = None # foreign
    test_set_id: int | None = None # foreign


@dataclass(frozen=True, slots=True)
class PointRecord:
    inputs: tuple[float, ...]
    outputs: tuple[float, ...]
    predictions: tuple[float, ...] | None

    point_id: int | None = None # primary
    set_id: int | None = None # foreign