from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class RunRecord:
    run_id: str # primary, in uuid format
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


@dataclass(frozen=True, slots=True)
class StudyRecord:
    run_id: str # foreign
    study_name: str
    evaluation_method: str


@dataclass(frozen=True, slots=True)
class GroupRecord:
    study_id: int # foreign
    group_name: str


@dataclass(frozen=True, slots=True)
class EvaluationRecord:
    group_id: int # foreign
    test_set_position: int
    y_true: tuple[float, ...]
    point_mean_prediction: tuple[float, ...]
    bias: tuple[float, ...]
    variance: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class ModelRecord:
    group_id: int # foreign
    architecture: tuple[int, ...]
    model_mean_prediction: tuple[float, ...]
    model_variance_prediction: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class ScoreRecord:
    model_id: int # foreign
    metric: str
    score: float


@dataclass(frozen=True, slots=True)
class TrainPointRecord:
    model_id: int | None # foreign
    run_id: str | None # foreign
    input: tuple[float, ...]
    output: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class TestPointRecord:
    model_id: int | None # foreign
    run_id: str | None # foreign
    set_position: int
    input: tuple[float, ...]
    output: tuple[float, ...]
    prediction: tuple[float, ...]
