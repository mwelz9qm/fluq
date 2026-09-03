from dataclasses import dataclass
from datetime import datetime
from typing import ClassVar

from bias_variance.persistence.tables import (
    EvaluationTable,
    GroupTable,
    ModelTable,
    RunTable,
    ScoreTable,
    StudyTable,
    Table,
    TestPointTable,
    TrainPointTable,
)


class Record:
    table: ClassVar[type[Table]]

    def __init_subclass__(
        cls,
        *,
        table: type[Table] | None = None,
        **kwargs: object,
    ) -> None:
        super().__init_subclass__(**kwargs)
        if table is not None:
            cls.table = table


@dataclass(frozen=True, slots=True)
class RunRecord(Record, table=RunTable):
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
    input_columns: tuple[str, ...]
    output_columns: tuple[str, ...]
    base_architecture: tuple[int, ...]
    seed_entropy: int | None = None


@dataclass(frozen=True, slots=True)
class StudyRecord(Record, table=StudyTable):
    run_id: str
    study_name: str
    evaluation_method: str


@dataclass(frozen=True, slots=True)
class GroupRecord(Record, table=GroupTable):
    study_id: int
    group_name: str


@dataclass(frozen=True, slots=True)
class EvaluationRecord(Record, table=EvaluationTable):
    group_id: int
    test_set_position: int
    y_true: tuple[float, ...]
    point_mean_prediction: tuple[float, ...]
    bias: tuple[float, ...]
    variance: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class ModelRecord(Record, table=ModelTable):
    group_id: int
    architecture: tuple[int, ...]
    model_mean_prediction: tuple[float, ...]
    model_variance_prediction: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class ScoreRecord(Record, table=ScoreTable):
    model_id: int
    metric: str
    score: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class TrainPointRecord(Record, table=TrainPointTable):
    model_id: int | None
    run_id: str | None
    input: tuple[float, ...]
    output: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class TestPointRecord(Record, table=TestPointTable):
    model_id: int | None
    run_id: str | None
    set_position: int
    input: tuple[float, ...]
    output: tuple[float, ...]
    prediction: tuple[float, ...]
