import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime
from os import PathLike
from typing import Self

from bias_variance.persistence.records import Record, RunRecord
from bias_variance.persistence.serialize import (
    decode_datetime_string,
    decode_json_array,
    encode_tuple,
)
from bias_variance.persistence.tables import (
    EvaluationTable,
    GroupTable,
    ModelTable,
    RunTable,
    ScoreTable,
    StudyTable,
    TestPointTable,
    TrainPointTable,
)

type BiasVarianceResult = tuple[
    str,
    str,
    str,
    tuple[float, ...] | None,
    tuple[float, ...] | None,
]


@dataclass(frozen=True, slots=True)
class StoredRun:

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
class StoredTestPointPrediction:
    model_id: int
    set_position: int
    input: tuple[float, ...]
    output: tuple[float, ...]
    prediction: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class ModelResult:
    model_id: int
    mse: tuple[float, ...]
    variance: tuple[float, ...]
    mean: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class TestPointResult:
    test_point_position: int
    actual: tuple[float, ...]
    squared_bias: tuple[float, ...]
    variance: tuple[float, ...]
    mean: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class StoredResultGroup:
    study_id: int
    study_name: str
    evaluation_method: str
    group_id: int
    group_name: str


class ResultStore:

    def __init__(
        self,
        database: str | PathLike[str] = ":memory:",
        *,
        timeout: float = 5.0,
    ) -> None:
        self._connection = sqlite3.connect(database, timeout=timeout)
        self._connection.row_factory = sqlite3.Row

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            if exc_type is None:
                self.commit()

            else:
                self.rollback()

        finally:
            self.close()

    def create_tables(self) -> None:
        cur = self._connection.cursor()
        cur.execute('PRAGMA foreign_keys = ON')
        for table in (
            RunTable,
            StudyTable,
            GroupTable,
            EvaluationTable,
            ModelTable,
            ScoreTable,
            TrainPointTable,
            TestPointTable,
        ):
            cur.execute(table.create_table_sql())

        run_columns = {
            str(row['name'])
            for row in cur.execute(
                f'PRAGMA table_info({RunTable.TABLE_NAME})'
            ).fetchall()
        }
        if RunTable.SEED_ENTROPY.name not in run_columns:
            cur.execute(
                f'ALTER TABLE {RunTable.TABLE_NAME} '
                f'ADD COLUMN {RunTable.SEED_ENTROPY.to_sql()}'
            )

    def add(self, record: Record) -> int | str:
        insert_statement, params = record.table.insert_sql(asdict(record))

        cur = self._connection.cursor()
        cur.execute(insert_statement, params)

        if isinstance(record, RunRecord):
            return record.run_id

        return cur.lastrowid

    def update_group(
        self,
        group_id: int,
        bias: tuple[float, ...],
        variance: tuple[float, ...],
    ) -> None:
        serialized_bias = encode_tuple(bias)
        serialized_variance = encode_tuple(variance)

        cur = self._connection.cursor()
        cur.execute(
            f'''
            UPDATE {GroupTable.TABLE_NAME}
            SET ({GroupTable.STRATEGY_BIAS.name}, {GroupTable.STRATEGY_VARIANCE.name}) = (?, ?)
            WHERE {GroupTable.GROUP_ID.name} = ?
            AND {GroupTable.STRATEGY_BIAS.name} IS NULL
            AND {GroupTable.STRATEGY_VARIANCE.name} IS NULL
            ''',
            (serialized_bias, serialized_variance, group_id)
        )

        if cur.rowcount == 0:
            cur.execute(
                f'''
                SELECT {GroupTable.GROUP_ID.name}
                FROM {GroupTable.TABLE_NAME}
                WHERE {GroupTable.GROUP_ID.name} = ?
                ''',
                (group_id,),
            )
            if cur.fetchone() is None:
                raise KeyError(f'Unknown group_id: {group_id}')
            raise ValueError(f'Group already evaluated: {group_id}')

    def commit(self) -> None:
        self._connection.commit()

    def rollback(self) -> None:
        self._connection.rollback()

    def close(self) -> None:
        self._connection.close()

    def get_recent_run(self) -> str | None:
        cur = self._connection.cursor()

        # Get one run where runs are ordered by created_at descending
        cur.execute(
            f'''
            SELECT {RunTable.RUN_ID.name}
            FROM {RunTable.TABLE_NAME}
            ORDER BY {RunTable.CREATED_AT.name} DESC, {RunTable.RUN_ID.name} DESC
            LIMIT 1
            '''
        )

        row = cur.fetchone()

        return None if row is None else str(row[RunTable.RUN_ID.name])

    def get_runs(self) -> tuple[StoredRun, ...]:
        cur = self._connection.cursor()
        cur.execute(
            f'''
            SELECT
                {RunTable.RUN_ID.name},
                {RunTable.CREATED_AT.name},
                {RunTable.N_ITER.name},
                {RunTable.TEST_SIZE.name},
                {RunTable.TEST_METRICS.name},
                {RunTable.OPTIMIZER.name},
                {RunTable.LEARNING_RATE.name},
                {RunTable.LOSS.name},
                {RunTable.EPOCHS.name},
                {RunTable.BATCH_SIZE.name},
                {RunTable.DEVICE.name},
                {RunTable.INPUT_COLUMNS.name},
                {RunTable.OUTPUT_COLUMNS.name},
                {RunTable.BASE_ARCHITECTURE.name},
                {RunTable.SEED_ENTROPY.name}
            FROM {RunTable.TABLE_NAME}
            ORDER BY {RunTable.CREATED_AT.name} DESC,
                     {RunTable.RUN_ID.name} DESC
            '''
        )

        return tuple(
            StoredRun(
                run_id=str(row[RunTable.RUN_ID.name]),
                created_at=decode_datetime_string(
                    row[RunTable.CREATED_AT.name]
                ),
                n_iter=int(row[RunTable.N_ITER.name]),
                test_size=float(row[RunTable.TEST_SIZE.name]),
                test_metrics=tuple(
                    str(value)
                    for value in decode_json_array(
                        row[RunTable.TEST_METRICS.name]
                    )
                ),
                optimizer=str(row[RunTable.OPTIMIZER.name]),
                learning_rate=float(row[RunTable.LEARNING_RATE.name]),
                loss=str(row[RunTable.LOSS.name]),
                epochs=int(row[RunTable.EPOCHS.name]),
                batch_size=int(row[RunTable.BATCH_SIZE.name]),
                device=str(row[RunTable.DEVICE.name]),
                input_columns=tuple(
                    str(value)
                    for value in decode_json_array(
                        row[RunTable.INPUT_COLUMNS.name]
                    )
                ),
                output_columns=tuple(
                    str(value)
                    for value in decode_json_array(
                        row[RunTable.OUTPUT_COLUMNS.name]
                    )
                ),
                base_architecture=tuple(
                    int(value)
                    for value in decode_json_array(
                        row[RunTable.BASE_ARCHITECTURE.name]
                    )
                ),
                seed_entropy=(
                    None
                    if row[RunTable.SEED_ENTROPY.name] is None
                    else int(row[RunTable.SEED_ENTROPY.name])
                ),
            )
            for row in cur.fetchall()
        )

    def get_run(self, run_id: str) -> StoredRun | None:
        return next(
            (run for run in self.get_runs() if run.run_id == run_id),
            None,
        )

    def get_run_groups(self, run_id: str) -> tuple[StoredResultGroup, ...]:
        cur = self._connection.cursor()
        cur.execute(
            f'''
            SELECT
                s.{StudyTable.STUDY_ID.name},
                s.{StudyTable.STUDY_NAME.name},
                s.{StudyTable.EVALUATION_METHOD.name},
                g.{GroupTable.GROUP_ID.name},
                g.{GroupTable.GROUP_NAME.name}
            FROM {StudyTable.TABLE_NAME} AS s
            INNER JOIN {GroupTable.TABLE_NAME} AS g
                ON g.{GroupTable.STUDY_ID.name} = s.{StudyTable.STUDY_ID.name}
            WHERE s.{StudyTable.RUN_ID.name} = ?
            ORDER BY s.{StudyTable.STUDY_ID.name},
                     g.{GroupTable.GROUP_ID.name}
            ''',
            (run_id,),
        )

        return tuple(
            StoredResultGroup(
                study_id=int(row[StudyTable.STUDY_ID.name]),
                study_name=str(row[StudyTable.STUDY_NAME.name]),
                evaluation_method=str(
                    row[StudyTable.EVALUATION_METHOD.name]
                ),
                group_id=int(row[GroupTable.GROUP_ID.name]),
                group_name=str(row[GroupTable.GROUP_NAME.name]),
            )
            for row in cur.fetchall()
        )

    def get_studies(self, run_id: str) -> tuple[int, ...]:
        cur = self._connection.cursor()

        cur.execute(
            f'''
            SELECT {StudyTable.STUDY_ID.name}
            FROM {StudyTable.TABLE_NAME}
            WHERE {StudyTable.RUN_ID.name} = ?
            ORDER BY {StudyTable.STUDY_ID.name}
            ''',
            (run_id,)
        )
        rows = cur.fetchall()

        return tuple(
            int(row[StudyTable.STUDY_ID.name])
            for row in rows
        )

    def get_groups(self, study_id: int) -> tuple[int, ...]:
        cur = self._connection.cursor()
        
        cur.execute(
            f'''
            SELECT {GroupTable.GROUP_ID.name}
            FROM {GroupTable.TABLE_NAME}
            WHERE {GroupTable.STUDY_ID.name} = ?
            ORDER BY {GroupTable.GROUP_ID.name}
            ''',
            (study_id,)
        )
        rows = cur.fetchall()
        
        return tuple(
            int(row[GroupTable.GROUP_ID.name])
            for row in rows
        )

    def get_models(self, group_id: int) -> tuple[int, ...]:
        cur = self._connection.cursor()
        
        cur.execute(
            f'''
            SELECT {ModelTable.MODEL_ID.name}
            FROM {ModelTable.TABLE_NAME}
            WHERE {ModelTable.GROUP_ID.name} = ?
            ORDER BY {ModelTable.MODEL_ID.name}
            ''',
            (group_id,)
        )
        rows = cur.fetchall()
        
        return tuple(
            int(row[ModelTable.MODEL_ID.name])
            for row in rows
        )

    def get_test_set_positions(
        self,
        group_id: int,
    ) -> tuple[int, ...]:
        cur = self._connection.cursor()
        cur.execute(
            f'''
            SELECT DISTINCT tp.set_position
            FROM {TestPointTable.TABLE_NAME} AS tp
            JOIN {ModelTable.TABLE_NAME} AS m ON m.model_id = tp.model_id
            WHERE m.group_id = ?
            ORDER BY tp.set_position
            ''',
            (group_id,)
        )
        rows = cur.fetchall()

        return tuple(
            int(row['set_position'])
            for row in rows
        )

    def get_actual_and_predictions(
        self,
        group_id: int,
        test_set_pos: int,
    ) -> tuple[
        tuple[float, ...],
        tuple[tuple[float, ...], ...],
    ]:
        cur = self._connection.cursor()

        cur.execute(
            f'''
            SELECT tp.output, tp.prediction
            FROM {ModelTable.TABLE_NAME} AS m
            INNER JOIN {TestPointTable.TABLE_NAME} AS tp
                ON m.model_id = tp.model_id
            WHERE m.group_id = ?
            AND tp.set_position = ?
            ORDER BY m.model_id, tp.test_point_id
            ''',
            (group_id, test_set_pos)
        )

        rows = cur.fetchall()

        if not rows:
            raise ValueError(
                'query returned empty rows.'
            )

        actual = decode_json_array(rows[0]['output'])
        predictions = tuple(
            decode_json_array(row['prediction']) for row in rows
        )

        return actual, predictions

    def get_bias_variance_results(
        self,
        run_id: str,
    ) -> tuple[BiasVarianceResult, ...]:
        cur = self._connection.cursor()
        cur.execute(
            f'''
            SELECT
                s.study_name,
                g.group_name,
                s.evaluation_method,
                g.strategy_bias,
                g.strategy_variance
            FROM {RunTable.TABLE_NAME} AS r
            INNER JOIN {StudyTable.TABLE_NAME} AS s
                ON r.run_id = s.run_id
            INNER JOIN {GroupTable.TABLE_NAME} AS g
                ON s.study_id = g.study_id
            WHERE r.run_id = ?
            ORDER BY s.study_id, g.group_id
            ''',
            (run_id,),
        )

        return tuple(
            (
                str(row['study_name']),
                str(row['group_name']),
                str(row['evaluation_method']),
                None if row['strategy_bias'] is None else tuple(
                    float(value) for value in decode_json_array(row['strategy_bias'])
                ),
                None if row['strategy_variance'] is None else tuple(
                    float(value) for value in decode_json_array(row['strategy_variance'])
                ),
            )
            for row in cur.fetchall()
        )

    def get_method(self, group_id: int) -> str:
        cur = self._connection.cursor()
        
        cur.execute(
            f'''
            SELECT studies.evaluation_method
            FROM {GroupTable.TABLE_NAME}
            JOIN {StudyTable.TABLE_NAME}
                ON {StudyTable.TABLE_NAME}.study_id = {GroupTable.TABLE_NAME}.study_id
            WHERE {GroupTable.TABLE_NAME}.group_id = ? LIMIT 1
            ''',
            (group_id,)
        )
        row = cur.fetchone()

        if row is None:
            raise KeyError(f'Unknown group_id: {group_id}')

        return str(row['evaluation_method'])

    def does_run_exist(self, run_id: str) -> bool:
        cur = self._connection.cursor()

        cur.execute(
            f'SELECT run_id FROM {RunTable.TABLE_NAME} WHERE run_id = ? LIMIT 1',
            (run_id,)
        )

        row = cur.fetchone()

        return row is not None

    def get_averaging_evaluation_data(
        self,
        group_id: int,
    ) -> tuple[
        tuple[
            tuple[float, ...], ...
        ],
        tuple[
            tuple[float, ...], ...
        ]
    ]:
        cur = self._connection.cursor()

        cur.execute(
            f'''
            SELECT
                {ScoreTable.TABLE_NAME}.{ScoreTable.SCORE.name},
                {ModelTable.TABLE_NAME}.{ModelTable.MODEL_VARIANCE_PREDICTION.name}
            FROM {ModelTable.TABLE_NAME}
            INNER JOIN {ScoreTable.TABLE_NAME}
            ON {ScoreTable.TABLE_NAME}.{ScoreTable.MODEL_ID.name} = {ModelTable.TABLE_NAME}.{ModelTable.MODEL_ID.name}
            WHERE {ModelTable.TABLE_NAME}.{ModelTable.GROUP_ID.name} = ?
            AND {ScoreTable.TABLE_NAME}.{ScoreTable.METRIC.name} = ?
            ORDER BY {ModelTable.TABLE_NAME}.{ModelTable.MODEL_ID.name}, {ScoreTable.SCORE_ID.name}
            ''',
            (group_id, 'mse')
        )

        rows = cur.fetchall()

        return (
            tuple(
                tuple(
                    float(value)
                    for value in decode_json_array(row[ScoreTable.SCORE.name])
                )
                for row in rows
            ),
            tuple(
                tuple(
                    float(value)
                    for value in decode_json_array(
                        row[ModelTable.MODEL_VARIANCE_PREDICTION.name]
                    )
                )
                for row in rows
            ),
        )

    def get_pointwise_evaluation_data(
        self,
        group_id: int,
    ) -> tuple[StoredTestPointPrediction, ...]:
        cur = self._connection.cursor()

        cur.execute(
            '''
            SELECT
                m.model_id,
                tp.set_position,
                tp.input,
                tp.output,
                tp.prediction
            FROM models AS m
            JOIN test_points AS tp ON tp.model_id = m.model_id
            WHERE m.group_id = ?
            ORDER BY tp.set_position, m.model_id;
            ''',
            (group_id,)
        )

        rows = cur.fetchall()
        points: list[StoredTestPointPrediction] = []

        for row in rows:
            points.append(
                StoredTestPointPrediction(
                    model_id=int(row[ModelTable.MODEL_ID.name]),
                    set_position=int(row[TestPointTable.SET_POSITION.name]),
                    input=tuple(
                        float(value)
                        for value in decode_json_array(
                            row[TestPointTable.INPUT.name]
                        )
                    ),
                    output=tuple(
                        float(value)
                        for value in decode_json_array(
                            row[TestPointTable.OUTPUT.name]
                        )
                    ),
                    prediction=tuple(
                        float(value)
                        for value in decode_json_array(
                            row[TestPointTable.PREDICTION.name]
                        )
                    ),
                )
            )

        return tuple(points)

    def get_model_results(self, group_id: int) -> tuple[ModelResult, ...]:
        cur = self._connection.cursor()
        cur.execute(
            f'''
            SELECT
                m.{ModelTable.MODEL_ID.name},
                s.{ScoreTable.SCORE.name},
                m.{ModelTable.MODEL_VARIANCE_PREDICTION.name},
                m.{ModelTable.MODEL_MEAN_PREDICTION.name}
            FROM {ModelTable.TABLE_NAME} AS m
            INNER JOIN {ScoreTable.TABLE_NAME} AS s
                ON s.{ScoreTable.MODEL_ID.name} = m.{ModelTable.MODEL_ID.name}
            WHERE m.{ModelTable.GROUP_ID.name} = ?
                AND s.{ScoreTable.METRIC.name} = ?
            ORDER BY m.{ModelTable.MODEL_ID.name}
            ''',
            (group_id, 'mse'),
        )

        return tuple(
            ModelResult(
                model_id=int(row[ModelTable.MODEL_ID.name]),
                mse=tuple(
                    float(value)
                    for value in decode_json_array(row[ScoreTable.SCORE.name])
                ),
                variance=tuple(
                    float(value)
                    for value in decode_json_array(
                        row[ModelTable.MODEL_VARIANCE_PREDICTION.name]
                    )
                ),
                mean=tuple(
                    float(value)
                    for value in decode_json_array(
                        row[ModelTable.MODEL_MEAN_PREDICTION.name]
                    )
                ),
            )
            for row in cur.fetchall()
        )

    def get_test_point_results(
        self,
        group_id: int,
    ) -> tuple[TestPointResult, ...]:
        cur = self._connection.cursor()
        cur.execute(
            f'''
            SELECT
                {EvaluationTable.TEST_SET_POSITION.name},
                {EvaluationTable.Y_TRUE.name},
                {EvaluationTable.BIAS.name},
                {EvaluationTable.VARIANCE.name},
                {EvaluationTable.POINT_MEAN_PREDICTION.name}
            FROM {EvaluationTable.TABLE_NAME}
            WHERE {EvaluationTable.GROUP_ID.name} = ?
            ORDER BY {EvaluationTable.TEST_SET_POSITION.name},
                     {EvaluationTable.EVALUATION_ID.name}
            ''',
            (group_id,),
        )

        return tuple(
            TestPointResult(
                test_point_position=int(
                    row[EvaluationTable.TEST_SET_POSITION.name]
                ),
                actual=tuple(
                    float(value)
                    for value in decode_json_array(
                        row[EvaluationTable.Y_TRUE.name]
                    )
                ),
                squared_bias=tuple(
                    float(value)
                    for value in decode_json_array(
                        row[EvaluationTable.BIAS.name]
                    )
                ),
                variance=tuple(
                    float(value)
                    for value in decode_json_array(
                        row[EvaluationTable.VARIANCE.name]
                    )
                ),
                mean=tuple(
                    float(value)
                    for value in decode_json_array(
                        row[EvaluationTable.POINT_MEAN_PREDICTION.name]
                    )
                ),
            )
            for row in cur.fetchall()
        )
