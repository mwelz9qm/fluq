from collections.abc import Iterator
from datetime import UTC, datetime

import numpy as np
import pytest

from bias_variance.persistence.records import (
    GroupRecord,
    ModelRecord,
    RunRecord,
    ScoreRecord,
    StudyRecord,
    TrainPointRecord,
)
from bias_variance.persistence.records import (
    TestPointRecord as StoredTestPointRecord,
)
from bias_variance.persistence.store import ResultStore, StoredRun


def _add_run(
    store: ResultStore,
    run_id: str = 'run-1',
    created_at: datetime | None = None,
) -> str:
    result = store.add(
        RunRecord(
            run_id=run_id,
            created_at=created_at or datetime.now(UTC),
            n_iter=2,
            test_size=0.2,
            test_metrics=('mse',),
            optimizer='adam',
            learning_rate=0.001,
            loss='mse',
            epochs=5,
            batch_size=16,
            device='cpu',
            input_columns=('x',),
            output_columns=('y',),
            base_architecture=(8,),
        )
    )
    assert isinstance(result, str)
    return result


@pytest.fixture
def store() -> Iterator[ResultStore]:
    with ResultStore() as result:
        result.create_tables()
        yield result


def test_empty_store_has_no_recent_run(store: ResultStore) -> None:
    assert store.get_recent_run() is None
    assert store.get_runs() == ()


def test_get_runs_returns_all_decoded_fields_newest_first(
    store: ResultStore,
) -> None:
    older = datetime(2026, 1, 1, tzinfo=UTC)
    newer = datetime(2026, 1, 2, tzinfo=UTC)
    _add_run(store, 'older-run', older)
    _add_run(store, 'newer-run', newer)

    runs = store.get_runs()

    assert tuple(run.run_id for run in runs) == ('newer-run', 'older-run')
    assert runs[0] == StoredRun(
        run_id='newer-run',
        created_at=newer,
        n_iter=2,
        test_size=0.2,
        test_metrics=('mse',),
        optimizer='adam',
        learning_rate=0.001,
        loss='mse',
        epochs=5,
        batch_size=16,
        device='cpu',
        input_columns=('x',),
        output_columns=('y',),
        base_architecture=(8,),
    )


def test_add_and_query_complete_record_hierarchy(store: ResultStore) -> None:
    run_id = _add_run(store)
    study_id = store.add(StudyRecord(run_id, 'model', 'averaging'))
    group_id = store.add(GroupRecord(study_id, 'small'))
    model_id = store.add(
        ModelRecord(
            group_id,
            (8,),
            (np.float32(2.5),),
            (np.float32(0.25),),
        )
    )

    assert store.add(ScoreRecord(model_id, 'mse', (0.5,))) == 1
    assert store.add(TrainPointRecord(model_id, None, (1.0,), (2.0,))) == 1
    assert store.add(
        StoredTestPointRecord(
            model_id,
            None,
            3,
            (1.0,),
            (2.0,),
            np.array([2.5], dtype=np.float32),
        )
    ) == 1

    assert store.get_recent_run() == run_id
    assert store.get_studies(run_id) == (study_id,)
    assert store.get_groups(study_id) == (group_id,)
    assert store.get_models(group_id) == (model_id,)
    assert store.get_test_set_positions(group_id) == (3,)
    assert store.get_method(group_id) == 'averaging'
    assert store.get_actual_and_predictions(group_id, 3) == (
        (2.0,),
        ((2.5,),),
    )
    assert store.get_averaging_evaluation_data(group_id) == (
        ((0.5,),),
        ((0.25,),),
    )

    store.update_group(group_id, (0.25,), (0.5,))
    assert store.get_bias_variance_results(run_id) == (
        ('model', 'small', 'averaging', (0.25,), (0.5,)),
    )

    with pytest.raises(ValueError, match='Group already evaluated'):
        store.update_group(group_id, (1.0,), (2.0,))


def test_group_position_query_orders_predictions_by_model_id(
    store: ResultStore,
) -> None:
    run_id = _add_run(store)
    study_id = store.add(StudyRecord(run_id, 'model', 'pointwise'))
    group_id = store.add(GroupRecord(study_id, 'small'))
    first_model_id = store.add(ModelRecord(group_id, (8,), (2.5,), (0.25,)))
    second_model_id = store.add(ModelRecord(group_id, (16,), (3.5,), (0.5,)))

    store.add(
        StoredTestPointRecord(
            second_model_id, None, 2, (0.0,), (1.0,), (4.1,)
        )
    )
    store.add(
        StoredTestPointRecord(
            first_model_id, None, 2, (0.0,), (1.0,), (1.1,)
        )
    )

    assert store.get_actual_and_predictions(group_id, 2) == (
        (1.0,),
        ((1.1,), (4.1,)),
    )


def test_missing_group_position_raises_value_error(store: ResultStore) -> None:
    with pytest.raises(ValueError, match='query returned empty rows'):
        store.get_actual_and_predictions(999, 0)


def test_missing_group_operations_raise_key_error(store: ResultStore) -> None:
    with pytest.raises(KeyError, match='Unknown group_id: 999'):
        store.get_method(999)

    with pytest.raises(KeyError, match='Unknown group_id: 999'):
        store.update_group(999, (1.0,), (2.0,))
