"""SQLite persistence interface for bias/variance run results.

The methods in :class:`ResultStore` mirror the persistence operations used by
``BiasAnalyzer`` and ``Evaluator``.  The query and mutation methods are
intentionally left as implementation points; this module defines their
signatures, expected SQL behavior, and return contracts without imposing a
particular serialization format for array-valued record fields.
"""

import sqlite3
from collections.abc import Sequence
from dataclasses import asdict
from os import PathLike
from typing import Any

from bias_variance.models.evaluation import EvaluationMethod
from bias_variance.persistence.records import (
    GroupRecord,
    ModelRecord,
    RunRecord,
    ScoreRecord,
    StudyRecord,
    TestPointRecord,
    TrainPointRecord,
)

type Record = RunRecord | StudyRecord | GroupRecord | ModelRecord | ScoreRecord | TrainPointRecord | TestPointRecord
type TestId = tuple[str, int]


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

    def create_tables(self) -> None:
        cur = self._connection.cursor()
        cur.execute('PRAGMA foreign_keys = ON').execute('''
            CREATE TABLE IF NOT EXISTS runs (
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
                base_architecture TEXT NOT NULL,
                base_train_set_id INTEGER NOT NULL,
                base_test_set_id INTEGER NOT NULL,
                input_columns TEXT NOT NULL,
                output_columns TEXT NOT NULL
            ) AS STRICT
        ''').execute('''
            CREATE TABLE IF NOT EXISTS studies (
                study_id INTEGER PRIMARY KEY,
                run_id TEXT NOT NULL,
                study_name TEXT NOT NULL,
                evaluation_method TEXT NOT NULL,
                FOREIGN KEY (run_id) REFERENCES runs (run_id)
            ) AS STRICT
        ''').execute('''
            CREATE TABLE IF NOT EXISTS groups (
                group_id INTEGER PRIMARY KEY,
                study_id INTEGER NOT NULL,
                group_name TEXT NOT NULL,
                averaging_strategy_bias REAL,
                averaging_strategy_variance REAL,
                pointwise_strategy_bias REAL,
                pointwise_strategy_variance REAL,
                FOREIGN KEY (study_id) REFERENCES studies (study_id)
            ) AS STRICT
        ''').execute('''
            CREATE TABLE IF NOT EXISTS models (
                model_id INTEGER PRIMARY KEY,
                group_id INTEGER NOT NULL,
                train_set_id INTEGER NOT NULL,
                test_set_id INTEGER NOT NULL,
                architecture TEXT NOT NULL,
                test_scores TEXT NOT NULL,
                FOREIGN KEY (group_id) REFERENCES groups (group_id)
            ) AS STRICT
        ''').execute('''
            CREATE TABLE IF NOT EXISTS scores (
                scores_id INTEGER PRIMARY KEY,
                model_id INTEGER NOT NULL,
                inputs TEXT NOT NULL,
                outputs TEXT NOT NULL,
                predictions TEXT,
                FOREIGN KEY (model_id) REFERENCES models (model_id)
            ) AS STRICT
        ''').execute('''
            CREATE TABLE IF NOT EXISTS train_points (
                train_point_id INTEGER PRIMARY KEY,
                model_id INTEGER,
                run_id INTEGER,
                train_point_inputs TEXT NOT NULL,
                train_point_outputs TEXT NOT NULL,
                FOREIGN KEY (model_id) REFERENCES models (model_id),
                FOREIGN KEY (run_id) REFERENCES runs (run_id)
            ) AS STRICT
        ''').execute('''
            CREATE TABLE IF NOT EXISTS test_points (
                test_point_id INTEGER PRIMARY KEY,
                model_id INTEGER,
                run_id INTEGER,
                set_position INTEGER NOT NULL,
                test_point_inputs TEXT NOT NULL,
                test_point_outputs TEXT NOT NULL,
                FOREIGN KEY (model_id) REFERENCES models (model_id),
                FOREIGN KEY (run_id) REFERENCES runs (run_id)
            ) AS STRICT
        ''')

    @staticmethod
    def _build_insert_statement(
        table_name: str,
        attributes: dict[str, Any],
        remove_attribute: str | None = None,
    ) -> tuple[str, tuple]:
        if remove_attribute:
            attributes.pop(remove_attribute, None)
        statement = f'INSERT INTO {table_name} ('
        values = []
        for field, value in attributes.items():
            statement += f'{field},'
            values.append(value)

        statement += f') VALUES ({len(values)*'?,'})'

        return statement, tuple(values)

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
        cur = self._connection.cursor()
        match record:
            case cls if cls is RunRecord:
                statement, params = self._build_insert_statement('runs', asdict(record))
                cur.execute(statement, params)
                return record.run_id
            
            case cls if cls is StudyRecord:
                statement, params = self._build_insert_statement(
                    'studies', asdict(record), 'study_id'
                )
                cur.execute(statement, params)
                record.study_id = cur.lastrowid
                return record.study_id
            
            case cls if cls is GroupRecord:
                statement, params = self._build_insert_statement(
                    'groups', asdict(record), 'group_id'
                )
                cur.execute(statement, params)
                record.group_id = cur.lastrowid
                return record.group_id
            
            case cls if cls is ModelRecord:
                statement, params = self._build_insert_statement(
                    'models', asdict(record), 'model_id'
                )
                cur.execute(statement, params)
                record.model_id = cur.lastrowid
                return record.model_id
            
            case cls if cls is ScoreRecord:
                statement, params = self._build_insert_statement(
                    'scores', asdict(record), 'score_id'
                )
                cur.execute(statement, params)
                record.score_id = cur.lastrowid
                return record.score_id
            
            case cls if cls is TrainPointRecord:
                statement, params = self._build_insert_statement(
                    'train_points', asdict(record), 'train_point_id'
                )
                cur.execute(statement, params)
                record.train_point_id = cur.lastrowid
                return record.train_point_id
            
            case cls if cls is TestPointRecord:
                statement, params = self._build_insert_statement(
                    'test_points', asdict(record), 'test_point_id'
                )
                cur.execute(statement, params)
                record.test_point_id = cur.lastrowid
                return record.test_point_id
            
            case _:
                raise TypeError(
                    f'No matching record class type: {type(record)!r}.'
                )

    def update_group(
        self,
        group_id: int,
        bias: float,
        variance: float,
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
        run_id:
            ID of the run that owns the evaluated study.
        study_id:
            ID of the study that owns the evaluated group.
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
        cur = self._connection.cursor()

        # Get study_id
        cur.execute(
            'SELECT study_id FROM groups WHERE group_id = ? LIMIT 1',
            (group_id,)
        )
        row = cur.fetchone()
        study_id = row['study_id']

        # Use fetched study_id for querying evaluation_method
        cur.execute(
            'SELECT evaluation_method FROM studies WHERE study_id = ? LIMIT 1',
            (study_id,)
        )
        row = cur.fetchone()
        method = row['evaluation_method']

        # Use fetched evaluation_method for updating bias and variance
        match method:
            case EvaluationMethod.AVERAGING.value:
                cur.execute(
                    '''
                    UPDATE groups SET (
                        averaging_strategy_bias,
                        averaging_strategy_variance
                    ) = (?,?) WHERE group_id = ?
                    ''',
                    (bias, variance, group_id)
                )
            case EvaluationMethod.POINTWISE.value:
                cur.execute(
                    '''
                    UPDATE groups SET (
                        pointwise_strategy_bias,
                        pointwise_strategy_variance
                    ) = (?,?) WHERE group_id = ?
                    ''',
                    (bias, variance, group_id)
                )
            case _:
                raise ValueError(
                    f'Unknown evaluation_method in studies table: {method!r}.'
                )

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
            'SELECT run_id FROM runs ORDER BY created_at DESC LIMIT 1'
        )

        row = cur.fetchone()

        return row['run_id']

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
            'SELECT study_id FROM studies WHERE run_id = ?',
            (run_id,)
        )
        rows = cur.fetchall()

        return (
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
            'SELECT group_id FROM groups WHERE study_id = ?',
            (study_id,)
        )
        rows = cur.fetchall()
        
        return (
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
            'SELECT model_id FROM models WHERE group_id = ?',
            (group_id,)
        )
        rows = cur.fetchall()
        
        return (
            int(row['model_id'])
            for row in rows
        )

    def get_actuals_and_predictions(
        self,
        model_id: int | None = None,
        group_id_and_tes_pos: tuple[int, int] | None = None,
    ) -> tuple[Sequence[tuple[float, ...]], Sequence[tuple[float, ...]]]:
        """Load every prediction produced by one model.

        The implementation should select the model's serialized prediction
        field, decode it, and preserve its original test-row order and numeric
        precision.

        Parameters
        ----------
        model_id:
            ID of the model row to query.

        Returns
        -------
        Sequence[float]
            Ordered model predictions.  Multi-output predictions require a
            documented flattening convention or a more specific return type.
        """
        cur = self._connection.cursor()

        if model_id and not group_id_and_tes_pos:
            cur.execute(
                'SELECT (outputs, predictions) FROM test_points WHERE model_id = ?',
                (model_id,)
            )
            
        elif not model_id and group_id_and_tes_pos:
            cur.execute(
                '''
                SELECT (test_points.outputs, test_points.predictions)
                FROM models
                INNER JOIN test_points ON models.model_id = test_points.model_id
                WHERE models.group_id = ?
                AND test_points.test_position = ?
                ''',
                group_id_and_tes_pos
            )

        else:
            raise ValueError(
                'model_id and group_id_and_test_pos arguments are mutually exclusive.'
            )
        
        rows = cur.fetchall()
        
        return tuple(rows)

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
            'SELECT evaluation_method FROM groups WHERE group_id = ? LIMIT 1',
            (group_id,)
        )
        row = cur.fetchone()
        
        return row['evaluation_method']
