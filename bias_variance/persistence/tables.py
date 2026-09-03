from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from bias_variance.persistence.serialize import encode_datetime, encode_tuple

type SQLiteT = Literal['INTEGER', 'REAL', 'TEXT']
type EncodingFunction = Callable[[Any], str]


@dataclass(frozen=True, slots=True)
class Column:
    name: str
    data_type: SQLiteT
    nullable: bool = False
    primary_key: bool = False

    def to_sql(self) -> str:
        parts = [self.name, self.data_type]
        if self.primary_key:
            parts.append('PRIMARY KEY')
        if not self.nullable:
            parts.append('NOT NULL')
        return ' '.join(parts)


class Table(ABC):
    TABLE_NAME: str

    @classmethod
    @abstractmethod
    def create_table_sql(cls) -> str:
        raise NotImplementedError

    @classmethod
    def insert_sql(cls, record_dict: Mapping[str, Any]) -> tuple[str, tuple[Any, ...]]:
        return cls._build_insert(record_dict, cls.TABLE_NAME)

    @staticmethod
    def _build_create_table(table_name: str, columns: Sequence[Column], constraints: Sequence[str] = ()) -> str:
        definitions = [column.to_sql() for column in columns]
        definitions.extend(constraints)
        return f"CREATE TABLE IF NOT EXISTS {table_name} (\n\t{',\n\t'.join(definitions)}\n) STRICT;"

    @staticmethod
    def _build_insert(record_dict: Mapping[str, Any], table_name: str) -> tuple[str, tuple[Any, ...]]:
        columns = ', '.join(record_dict)
        placeholders = ', '.join('?' for _ in record_dict)
        return f'INSERT INTO {table_name} ({columns}) VALUES ({placeholders})', tuple(record_dict.values())

    @staticmethod
    def _apply_encoding(serialize_map: Mapping[str, EncodingFunction], record_dict: dict[str, Any]) -> None:
        for key, encode in serialize_map.items():
            if key not in record_dict:
                raise KeyError(f'Key not found: {key!r}.')
            record_dict[key] = encode(record_dict[key])

    @classmethod
    def _encoded_insert(cls, record_dict: Mapping[str, Any], serialize_map: Mapping[str, EncodingFunction]) -> tuple[str, tuple[Any, ...]]:
        encoded_record = dict(record_dict)
        cls._apply_encoding(serialize_map, encoded_record)
        return cls._build_insert(encoded_record, cls.TABLE_NAME)


class RunTable(Table):
    TABLE_NAME = 'runs'
    RUN_ID = Column('run_id', 'TEXT', primary_key=True)
    CREATED_AT = Column('created_at', 'TEXT')
    N_ITER = Column('n_iter', 'INTEGER')
    TEST_SIZE = Column('test_size', 'REAL')
    TEST_METRICS = Column('test_metrics', 'TEXT')
    OPTIMIZER = Column('optimizer', 'TEXT')
    LEARNING_RATE = Column('learning_rate', 'REAL')
    LOSS = Column('loss', 'TEXT')
    EPOCHS = Column('epochs', 'INTEGER')
    BATCH_SIZE = Column('batch_size', 'INTEGER')
    DEVICE = Column('device', 'TEXT')
    INPUT_COLUMNS = Column('input_columns', 'TEXT')
    OUTPUT_COLUMNS = Column('output_columns', 'TEXT')
    BASE_ARCHITECTURE = Column('base_architecture', 'TEXT')
    SEED_ENTROPY = Column('seed_entropy', 'INTEGER', nullable=True)

    @classmethod
    def create_table_sql(cls) -> str:
        return cls._build_create_table(
            cls.TABLE_NAME, (
                cls.RUN_ID,
                cls.CREATED_AT,
                cls.N_ITER,
                cls.TEST_SIZE,
                cls.TEST_METRICS,
                cls.OPTIMIZER,
                cls.LEARNING_RATE,
                cls.LOSS,
                cls.EPOCHS,
                cls.BATCH_SIZE,
                cls.DEVICE,
                cls.INPUT_COLUMNS,
                cls.OUTPUT_COLUMNS,
                cls.BASE_ARCHITECTURE,
                cls.SEED_ENTROPY,
            )
        )

    @classmethod
    def insert_sql(cls, record_dict: Mapping[str, Any]) -> tuple[str, tuple[Any, ...]]:
        return cls._encoded_insert(
            record_dict, {
                cls.CREATED_AT.name: encode_datetime,
                cls.TEST_METRICS.name: encode_tuple,
                cls.INPUT_COLUMNS.name: encode_tuple,
                cls.OUTPUT_COLUMNS.name: encode_tuple,
                cls.BASE_ARCHITECTURE.name: encode_tuple
            }
        )


class StudyTable(Table):
    TABLE_NAME = 'studies'
    STUDY_ID = Column('study_id', 'INTEGER', primary_key=True)
    RUN_ID = Column('run_id', 'TEXT')
    STUDY_NAME = Column('study_name', 'TEXT')
    EVALUATION_METHOD = Column('evaluation_method', 'TEXT')

    @classmethod
    def create_table_sql(cls) -> str:
        return cls._build_create_table(
            cls.TABLE_NAME, (
                cls.STUDY_ID,
                cls.RUN_ID,
                cls.STUDY_NAME,
                cls.EVALUATION_METHOD
            ),
            ('FOREIGN KEY (run_id) REFERENCES runs (run_id)',)
        )


class GroupTable(Table):
    TABLE_NAME = 'groups'
    GROUP_ID = Column('group_id', 'INTEGER', primary_key=True)
    STUDY_ID = Column('study_id', 'INTEGER')
    GROUP_NAME = Column('group_name', 'TEXT')
    STRATEGY_BIAS = Column('strategy_bias', 'TEXT', nullable=True)
    STRATEGY_VARIANCE = Column('strategy_variance', 'TEXT', nullable=True)

    @classmethod
    def create_table_sql(cls) -> str:
        return cls._build_create_table(
            cls.TABLE_NAME, (
                cls.GROUP_ID,
                cls.STUDY_ID,
                cls.GROUP_NAME,
                cls.STRATEGY_BIAS,
                cls.STRATEGY_VARIANCE
            ),
            ('FOREIGN KEY (study_id) REFERENCES studies (study_id)',)
        )


class EvaluationTable(Table):
    TABLE_NAME = 'evaluations'
    EVALUATION_ID = Column('evaluation_id', 'INTEGER', primary_key=True)
    GROUP_ID = Column('group_id', 'INTEGER')
    TEST_SET_POSITION = Column('test_set_position', 'INTEGER')
    Y_TRUE = Column('y_true', 'TEXT')
    POINT_MEAN_PREDICTION = Column('point_mean_prediction', 'TEXT')
    BIAS = Column('bias', 'TEXT')
    VARIANCE = Column('variance', 'TEXT')

    @classmethod
    def create_table_sql(cls) -> str:
        return cls._build_create_table(
            cls.TABLE_NAME, (
                cls.EVALUATION_ID,
                cls.GROUP_ID,
                cls.TEST_SET_POSITION,
                cls.Y_TRUE,
                cls.POINT_MEAN_PREDICTION,
                cls.BIAS,
                cls.VARIANCE
            ),
            (
                'FOREIGN KEY (group_id) REFERENCES groups (group_id)',
                'UNIQUE (group_id, test_set_position)'
            )
        )

    @classmethod
    def insert_sql(cls, record_dict: Mapping[str, Any]) -> tuple[str, tuple[Any, ...]]:
        return cls._encoded_insert(
            record_dict, {
                column.name: encode_tuple
                for column
                in (
                    cls.Y_TRUE,
                    cls.POINT_MEAN_PREDICTION,
                    cls.BIAS,
                    cls.VARIANCE
                )
            }
        )


class ModelTable(Table):
    TABLE_NAME = 'models'
    MODEL_ID = Column('model_id', 'INTEGER', primary_key=True)
    GROUP_ID = Column('group_id', 'INTEGER')
    ARCHITECTURE = Column('architecture', 'TEXT')
    MODEL_MEAN_PREDICTION = Column('model_mean_prediction', 'TEXT')
    MODEL_VARIANCE_PREDICTION = Column('model_variance_prediction', 'TEXT')

    @classmethod
    def create_table_sql(cls) -> str:
        return cls._build_create_table(
            cls.TABLE_NAME, (
                cls.MODEL_ID,
                cls.GROUP_ID,
                cls.ARCHITECTURE,
                cls.MODEL_MEAN_PREDICTION,
                cls.MODEL_VARIANCE_PREDICTION
            ),
            ('FOREIGN KEY (group_id) REFERENCES groups (group_id)',)
        )

    @classmethod
    def insert_sql(cls, record_dict: Mapping[str, Any]) -> tuple[str, tuple[Any, ...]]:
        return cls._encoded_insert(
            record_dict, {
                column.name: encode_tuple
                for column
                in (
                    cls.ARCHITECTURE,
                    cls.MODEL_MEAN_PREDICTION,
                    cls.MODEL_VARIANCE_PREDICTION
                )
            }
        )


class ScoreTable(Table):
    TABLE_NAME = 'scores'
    SCORE_ID = Column('score_id', 'INTEGER', primary_key=True)
    MODEL_ID = Column('model_id', 'INTEGER')
    METRIC = Column('metric', 'TEXT')
    SCORE = Column('score', 'TEXT')

    @classmethod
    def create_table_sql(cls) -> str:
        return cls._build_create_table(
            cls.TABLE_NAME, (
                cls.SCORE_ID,
                cls.MODEL_ID,
                cls.METRIC,
                cls.SCORE
            ),
            (
                'FOREIGN KEY (model_id) REFERENCES models (model_id)',
                'UNIQUE (model_id, metric)'
            )
        )

    @classmethod
    def insert_sql(cls, record_dict: Mapping[str, Any]) -> tuple[str, tuple[Any, ...]]:
        return cls._encoded_insert(
            record_dict, {
                cls.SCORE.name: encode_tuple,
            }
        )

class TrainPointTable(Table):
    TABLE_NAME = 'train_points'
    TRAIN_POINT_ID = Column('train_point_id', 'INTEGER', primary_key=True)
    MODEL_ID = Column('model_id', 'INTEGER', nullable=True)
    RUN_ID = Column('run_id', 'TEXT', nullable=True)
    INPUT = Column('input', 'TEXT')
    OUTPUT = Column('output', 'TEXT')

    @classmethod
    def create_table_sql(cls) -> str:
        return cls._build_create_table(
            cls.TABLE_NAME, (
                cls.TRAIN_POINT_ID,
                cls.MODEL_ID,
                cls.RUN_ID,
                cls.INPUT,
                cls.OUTPUT
            ), (
                'FOREIGN KEY (model_id) REFERENCES models (model_id)',
                'FOREIGN KEY (run_id) REFERENCES runs (run_id)'
            )
        )

    @classmethod
    def insert_sql(cls, record_dict: Mapping[str, Any]) -> tuple[str, tuple[Any, ...]]:
        return cls._encoded_insert(record_dict, {cls.INPUT.name: encode_tuple, cls.OUTPUT.name: encode_tuple})


class TestPointTable(Table):
    TABLE_NAME = 'test_points'
    TEST_POINT_ID = Column('test_point_id', 'INTEGER', primary_key=True)
    MODEL_ID = Column('model_id', 'INTEGER', nullable=True)
    RUN_ID = Column('run_id', 'TEXT', nullable=True)
    SET_POSITION = Column('set_position', 'INTEGER')
    INPUT = Column('input', 'TEXT')
    OUTPUT = Column('output', 'TEXT')
    PREDICTION = Column('prediction', 'TEXT')

    @classmethod
    def create_table_sql(cls) -> str:
        return cls._build_create_table(
            cls.TABLE_NAME, (
                cls.TEST_POINT_ID,
                cls.MODEL_ID,
                cls.RUN_ID,
                cls.SET_POSITION,
                cls.INPUT,
                cls.OUTPUT,
                cls.PREDICTION
            ), (
                'FOREIGN KEY (model_id) REFERENCES models (model_id)',
                'FOREIGN KEY (run_id) REFERENCES runs (run_id)'
            )
        )

    @classmethod
    def insert_sql(cls, record_dict: Mapping[str, Any]) -> tuple[str, tuple[Any, ...]]:
        return cls._encoded_insert(record_dict, {cls.INPUT.name: encode_tuple, cls.OUTPUT.name: encode_tuple, cls.PREDICTION.name: encode_tuple})
