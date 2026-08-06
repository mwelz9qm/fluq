"""SQLite persistence interface for bias/variance run results.

The methods in :class:`ResultStore` mirror the persistence operations used by
``BiasAnalyzer`` and ``Evaluator``.  The query and mutation methods are
intentionally left as implementation points; this module defines their
signatures, expected SQL behavior, and return contracts without imposing a
particular serialization format for array-valued record fields.
"""

import sqlite3
from dataclasses import asdict
from enum import StrEnum
from os import PathLike
from typing import Self

from bias_variance.persistence.records import (
    GroupRecord,
    ModelRecord,
    RunRecord,
    ScoreRecord,
    StudyRecord,
    TestPointRecord,
    TrainPointRecord,
)
from bias_variance.persistence.serialize import (
    decode_json_array,
    encode_datetime,
    encode_tuple,
)

type Record = RunRecord | StudyRecord | GroupRecord | ModelRecord | ScoreRecord | TrainPointRecord | TestPointRecord


class TableName(StrEnum):
    RUNS = 'runs'
    STUDIES = 'studies'
    GROUPS = 'groups'
    MODELS = 'models'
    SCORES = 'scores'
    TRAIN_POINTS = 'train_points'
    TEST_POINTS = 'test_points'


class ResultStore:
    """Provide the SQLite operations required by an analysis run.

    A store owns one SQLite connection.  Calls to :meth:`add` and
    :meth:`update` participate in the connection's current transaction, and
    :meth:`commit` makes the complete run durable.  Array, tuple, mapping, and
    datetime fields in the record dataclasses must be encoded consistently by
    the eventual implementation (for example as JSON or binary values) because
    SQLite cannot store those Python objects directly.
    """

    def __init__(
        self,
        database: str | PathLike[str] = ":memory:",
        *,
        timeout: float = 5.0,
    ) -> None:
        """Open the SQLite database used to cache analysis results.

        The constructor creates a standard-library ``sqlite3`` connection and
        enables foreign-key enforcement.  A complete implementation should
        also create or migrate the tables represented by ``RunRecord``,
        ``StudyRecord``, ``GroupRecord``, and ``ModelRecord`` before the store
        is used.

        Parameters
        ----------
        database:
            Filesystem path to the SQLite database, or ``":memory:"`` for a
            transient in-memory cache.
        timeout:
            Number of seconds the connection waits for a locked table before
            raising ``sqlite3.OperationalError``.

        Returns
        -------
        None
        """
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

        cur.execute(
            f'''
            CREATE TABLE IF NOT EXISTS {TableName.RUNS.value} (
                run_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                n_iter INTEGER NOT NULL,
                test_size REAL NOT NULL,
                test_metrics TEXT NOT NULL,
                optimizer TEXT NOT NULL,
                learning_rate REAL NOT NULL,
                loss TEXT NOT NULL,
                epochs INTEGER NOT NULL,
                batch_size INTEGER NOT NULL,
                device TEXT NOT NULL,
                input_columns TEXT NOT NULL,
                output_columns TEXT NOT NULL,
                base_architecture TEXT NOT NULL
            ) STRICT
            '''
        )

        cur.execute(
            f'''
            CREATE TABLE IF NOT EXISTS {TableName.STUDIES.value} (
                study_id INTEGER PRIMARY KEY,
                run_id TEXT NOT NULL,
                study_name TEXT NOT NULL,
                evaluation_method TEXT NOT NULL,
                FOREIGN KEY (run_id) REFERENCES runs (run_id)
            ) STRICT
            '''
        )

        cur.execute(
            f'''
            CREATE TABLE IF NOT EXISTS {TableName.GROUPS.value} (
                group_id INTEGER PRIMARY KEY,
                study_id INTEGER NOT NULL,
                group_name TEXT NOT NULL,
                bias TEXT,
                variance TEXT,
                FOREIGN KEY (study_id) REFERENCES studies (study_id)
            ) STRICT
            '''
        )

        cur.execute(
            f'''
            CREATE TABLE IF NOT EXISTS {TableName.MODELS.value} (
                model_id INTEGER PRIMARY KEY,
                group_id INTEGER NOT NULL,
                architecture TEXT NOT NULL,
                FOREIGN KEY (group_id) REFERENCES groups (group_id)
            ) STRICT
            '''
        )

        cur.execute(
            f'''
            CREATE TABLE IF NOT EXISTS {TableName.SCORES.value} (
                score_id INTEGER PRIMARY KEY,
                model_id INTEGER NOT NULL,
                metric TEXT NOT NULL,
                score REAL NOT NULL,
                FOREIGN KEY (model_id) REFERENCES models (model_id)
            ) STRICT
            '''
        )

        cur.execute(
            f'''
            CREATE TABLE IF NOT EXISTS {TableName.TRAIN_POINTS.value} (
                train_point_id INTEGER PRIMARY KEY,
                model_id INTEGER,
                run_id TEXT,
                inputs TEXT NOT NULL,
                outputs TEXT NOT NULL,
                FOREIGN KEY (model_id) REFERENCES models (model_id),
                FOREIGN KEY (run_id) REFERENCES runs (run_id)
            ) STRICT
            '''
        )

        cur.execute(
            f'''
            CREATE TABLE IF NOT EXISTS {TableName.TEST_POINTS.value} (
                test_point_id INTEGER PRIMARY KEY,
                model_id INTEGER,
                run_id TEXT,
                set_position INTEGER NOT NULL,
                inputs TEXT NOT NULL,
                outputs TEXT NOT NULL,
                predictions TEXT NOT NULL,
                FOREIGN KEY (model_id) REFERENCES models (model_id),
                FOREIGN KEY (run_id) REFERENCES runs (run_id)
            ) STRICT
            '''
        )

    @staticmethod
    def _create_insert_statement(record: Record) -> tuple[str, tuple]:
        attributes = asdict(record).copy()
        tuple_keys: tuple[str, ...] = ()
        match record:
            case RunRecord():
                table_name = TableName.RUNS
                dt_value = attributes.get('created_at', None)
                if dt_value is not None:
                    attributes['created_at'] = encode_datetime(dt_value)
                tuple_keys = (
                    'test_metrics',
                    'input_columns',
                    'output_columns',
                    'base_architecture',
                )

            case StudyRecord():
                table_name = TableName.STUDIES

            case GroupRecord():
                table_name = TableName.GROUPS
                tuple_keys = (
                    'bias',
                    'variance'
                )

            case ModelRecord():
                table_name = TableName.MODELS
                tuple_keys = ('architecture',)

            case ScoreRecord():
                table_name = TableName.SCORES

            case TrainPointRecord():
                table_name = TableName.TRAIN_POINTS
                tuple_keys = ('inputs', 'outputs')

            case TestPointRecord():
                table_name = TableName.TEST_POINTS
                tuple_keys = ('inputs', 'outputs', 'predictions')

            case _:
                raise TypeError(
                    f'No matching record class type: {type(record)!r}.'
                )

        for key in tuple_keys:
            value = attributes.get(key, None)
            if value is not None:
                attributes[key] = encode_tuple(value)

        columns = ', '.join(attributes.keys())
        placeholders = ', '.join('?' for _ in attributes)
        statement = f'INSERT INTO {table_name.value} ({columns}) VALUES ({placeholders})'

        return statement, tuple(attributes.values())

    def add(self, record: Record) -> int | str:
        """Stage one record for insertion into its corresponding table.

        The implementation should dispatch on the concrete record type, encode
        non-scalar fields, and execute a parameterized ``INSERT`` without
        committing.  If the record's primary-key field is empty, it should
        generate a unique ID before insertion.  The returned ID must then be
        used by the caller when constructing dependent records.

        Parameters
        ----------
        record:
            A run, study, group, or model record whose fields map to columns in
            the matching SQLite table.

        Returns
        -------
        int
            The inserted row's primary-key ID, whether supplied by ``record``
            or generated by the store.
        """
        insert_statement, params = self._create_insert_statement(record)

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
        """Stage bias and variance values for an evaluated group.

        The implementation should verify that ``group_id`` belongs to
        ``study_id`` and that the study belongs to ``run_id``.  It should read
        the study's evaluation method and update either the averaging or
        pointwise bias/variance columns on the group row with one parameterized
        ``UPDATE`` statement.  The change remains pending until :meth:`commit`
        is called.

        ``run_id`` and ``study_id`` are not strictly required to locate a group
        when IDs are globally unique, but retaining them prevents an update
        from silently crossing run or study boundaries.

        Parameters
        ----------
        group_id:
            ID of the group whose decomposition values are being stored.
        bias:
            Calculated bias value.
        variance:
            Calculated variance value.

        Returns
        -------
        None
        """
        serialized_bias = encode_tuple(bias)
        serialized_variance = encode_tuple(variance)

        cur = self._connection.cursor()
        cur.execute(
            f'''
            UPDATE {TableName.GROUPS.value}
            SET (bias, variance) = (?, ?)
            WHERE group_id = ?
            ''',
            (serialized_bias, serialized_variance, group_id)
        )

        if cur.rowcount == 0:
            raise KeyError(f'Unknown group_id: {group_id}')

    def commit(self) -> None:
        """Commit all staged inserts and updates to SQLite.

        This delegates to the connection transaction so that records added
        during a run become durable together.  SQLite rolls the transaction
        back automatically when the connection closes after an unhandled
        error, but callers may also use :meth:`rollback` explicitly.

        Returns
        -------
        None
        """
        self._connection.commit()

    def rollback(self) -> None:
        """Discard all uncommitted changes in the current transaction.

        This calls the SQLite connection's rollback operation and is useful
        when model training or row serialization fails partway through a run.

        Returns
        -------
        None
        """
        self._connection.rollback()

    def close(self) -> None:
        """Close the underlying SQLite connection.

        Closing releases database resources.  Pending changes are not
        implicitly committed, so callers should call :meth:`commit` first when
        they want to retain them.

        Returns
        -------
        None
        """
        self._connection.close()

    def get_recent_run(self) -> str | None:
        """Return the ID of the most recently created run.

        The implementation should query the run table, order rows by
        ``created_at`` descending with a deterministic primary-key tiebreaker,
        and read at most one row.

        Returns
        -------
        str or None
            The newest run ID, or ``None`` when the run table is empty.
        """
        cur = self._connection.cursor()

        # Get one run where runs are ordered by created_at descending
        cur.execute(
            f'''
            SELECT run_id
            FROM {TableName.RUNS.value}
            ORDER BY created_at DESC, run_id DESC
            LIMIT 1
            '''
        )

        row = cur.fetchone()

        return None if row is None else str(row['run_id'])

    def get_studies(self, run_id: str) -> tuple[int, ...]:
        """Return all study IDs belonging to a run.

        The implementation should select study primary keys whose foreign key
        equals ``run_id`` in a stable order.

        Parameters
        ----------
        run_id:
            ID of the parent run.

        Returns
        -------
        tuple[str, ...]
            Study IDs for the run; an empty tuple when none exist.
        """
        cur = self._connection.cursor()

        cur.execute(
            f'''
            SELECT study_id
            FROM {TableName.STUDIES.value}
            WHERE run_id = ?
            ORDER BY study_id
            ''',
            (run_id,)
        )
        rows = cur.fetchall()

        return tuple(
            int(row['study_id'])
            for row in rows
        )

    def get_groups(self, study_id: int) -> tuple[int, ...]:
        """Return all group IDs belonging to a study.

        The implementation should select group primary keys whose foreign key
        equals ``study_id`` in a stable order.

        Parameters
        ----------
        study_id:
            ID of the parent study.

        Returns
        -------
        tuple[str, ...]
            Group IDs for the study; an empty tuple when none exist.
        """
        cur = self._connection.cursor()
        
        cur.execute(
            f'''
            SELECT group_id
            FROM {TableName.GROUPS.value}
            WHERE study_id = ?
            ORDER BY group_id
            ''',
            (study_id,)
        )
        rows = cur.fetchall()
        
        return tuple(
            int(row['group_id'])
            for row in rows
        )

    def get_models(self, group_id: int) -> tuple[int, ...]:
        """Return all model IDs belonging to a group.

        The implementation should select model primary keys whose foreign key
        equals ``group_id`` in a stable order.  Averaging evaluation uses these
        IDs to calculate one bias/variance contribution per model.

        Parameters
        ----------
        group_id:
            ID of the parent variation group.

        Returns
        -------
        tuple[str, ...]
            Model IDs for the group; an empty tuple when none exist.
        """
        cur = self._connection.cursor()
        
        cur.execute(
            f'''
            SELECT model_id
            FROM {TableName.MODELS.value}
            WHERE group_id = ?
            ORDER BY model_id
            ''',
            (group_id,)
        )
        rows = cur.fetchall()
        
        return tuple(
            int(row['model_id'])
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
            FROM {TableName.TEST_POINTS.value} AS tp
            JOIN {TableName.MODELS.value} AS m ON m.model_id = tp.model_id
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

    def get_actuals_and_predictions(
        self,
        model_id: int | None = None,
        group_id_and_set_pos: tuple[int, int] | None = None,
    ) -> tuple[
        tuple[tuple[float, ...], ...],
        tuple[tuple[float, ...], ...],
    ]:
        """Load every prediction produced by one model.

        The implementation should select the model's serialized prediction
        field, decode it, and preserve its original test-row order and numeric
        precision.

        Parameters
        ----------
        model_id:
            ID of the model row to query.
        group_id_and_set_pos:
            Group ID and testing-set position used to load actual and predicted
            outputs across all models in that group.

        Returns
        -------
        tuple[tuple[tuple[float, ...], ...], tuple[tuple[float, ...], ...]]
            The ordered actual-output rows and prediction rows.
        """
        cur = self._connection.cursor()

        if model_id is not None and group_id_and_set_pos is None:
            cur.execute(
                f'''
                SELECT outputs, predictions
                FROM {TableName.TEST_POINTS.value}
                WHERE model_id = ?
                ORDER BY set_position, test_point_id
                ''',
                (model_id,)
            )
            
        elif model_id is None and group_id_and_set_pos is not None:
            cur.execute(
                f'''
                SELECT tp.outputs, tp.predictions
                FROM {TableName.MODELS.value} AS m
                INNER JOIN {TableName.TEST_POINTS.value} AS tp
                    ON m.model_id = tp.model_id
                WHERE m.group_id = ?
                AND tp.set_position = ?
                ORDER BY m.model_id, tp.test_point_id
                ''',
                group_id_and_set_pos
            )

        else:
            raise ValueError(
                'model_id and group_id_and_set_pos arguments are mutually exclusive.'
            )
        
        rows = cur.fetchall()

        actuals = tuple(decode_json_array(row['outputs']) for row in rows)
        predictions = tuple(
            decode_json_array(row['predictions']) for row in rows
        )

        return actuals, predictions

    def get_method(self, group_id: int) -> str:
        """Return the evaluation method that applies to a group.

        The implementation should join the group row to its parent study and
        select the study's ``evaluation_method`` value.

        Parameters
        ----------
        group_id:
            ID of the group whose evaluation strategy is required.

        Returns
        -------
        str
            Stored evaluation method, currently ``"averaging"`` or
            ``"pointwise"``.
        """
        cur = self._connection.cursor()
        
        cur.execute(
            f'''
            SELECT studies.evaluation_method
            FROM {TableName.GROUPS.value}
            JOIN {TableName.STUDIES.value}
                ON {TableName.STUDIES.value}.study_id = {TableName.GROUPS.value}.study_id
            WHERE {TableName.GROUPS.value}.group_id = ? LIMIT 1
            ''',
            (group_id,)
        )
        row = cur.fetchone()

        if row is None:
            raise KeyError(f'Unknown group_id: {group_id}')

        return str(row['evaluation_method'])
