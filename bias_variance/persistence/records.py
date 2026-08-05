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


@dataclass(frozen=True, slots=True)
class StudyRecord:
    run_id: str # foreign
    study_name: str
    evaluation_method: str

    study_id: int | None = None # primary


@dataclass(frozen=True, slots=True)
class GroupRecord:
    study_id: int # foreign
    group_name: str
    bias: tuple[float, ...] | None = None
    variance: tuple[float, ...] | None = None

    group_id: int | None = None # primary


@dataclass(frozen=True, slots=True)
class ModelRecord:
    group_id: int # foreign
    architecture: tuple[int, ...]

    model_id: int | None = None # primary


@dataclass(frozen=True, slots=True)
class ScoreRecord:
    model_id: int # foreign
    metric: str
    score: float

    score_id: int | None = None # primary


@dataclass(frozen=True, slots=True)
class TrainPointRecord:
    model_id: int | None # foreign
    run_id: int | None # foreign
    inputs: tuple[float, ...]
    outputs: tuple[float, ...]

    train_point_id: int | None = None # primary


@dataclass(frozen=True, slots=True)
class TestPointRecord:
    model_id: int | None # foreign
    run_id: int | None # foreign
    set_position: int
    inputs: tuple[float, ...]
    outputs: tuple[float, ...]
    predictions: tuple[float, ...]

    test_point_id: int | None = None # primary
