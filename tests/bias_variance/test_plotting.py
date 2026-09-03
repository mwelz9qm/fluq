"""Tests for tidy bias/variance plot preparation and rendering."""

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes

from bias_variance.analyzer import BiasAnalyzer
from bias_variance.models.evaluation import EvaluationMethod
from bias_variance.persistence.records import (
    EvaluationRecord,
    GroupRecord,
    ModelRecord,
    RunRecord,
    ScoreRecord,
    StudyRecord,
)
from bias_variance.persistence.store import ResultStore
from bias_variance.plotting import (
    plot_bias_and_variance,
    plot_error_components,
    plot_summary,
)


RUN_ID = 'plot-run'


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close('all')


def _create_plot_database(path: Path) -> tuple[int, int]:
    with ResultStore(path) as store:
        store.create_tables()
        store.add(
            RunRecord(
                run_id=RUN_ID,
                created_at=datetime.now(UTC),
                n_iter=2,
                test_size=0.25,
                test_metrics=('mse',),
                optimizer='adam',
                learning_rate=0.001,
                loss='mse',
                epochs=1,
                batch_size=4,
                device='cpu',
                input_columns=('x',),
                output_columns=('y', 'z'),
                base_architecture=(4,),
            )
        )

        pointwise_study = store.add(
            StudyRecord(RUN_ID, 'model', 'pointwise')
        )
        pointwise_group = store.add(GroupRecord(pointwise_study, 'wide'))
        store.add(
            EvaluationRecord(
                pointwise_group,
                0,
                (1.0, 10.0),
                (1.5, 11.0),
                (0.25, 1.0),
                (0.04, 0.09),
            )
        )
        store.add(
            EvaluationRecord(
                pointwise_group,
                2,
                (3.0, 20.0),
                (2.0, 19.5),
                (1.0, 0.25),
                (0.16, 0.25),
            )
        )
        store.update_group(pointwise_group, (0.625, 0.625), (0.1, 0.17))

        averaging_study = store.add(
            StudyRecord(RUN_ID, 'sampling', 'averaging')
        )
        averaging_group = store.add(GroupRecord(averaging_study, 'small'))
        first_model = store.add(
            ModelRecord(averaging_group, (4,), (2.0, 12.0), (0.25, 1.0))
        )
        second_model = store.add(
            ModelRecord(averaging_group, (8,), (3.0, 18.0), (0.36, 1.44))
        )
        store.add(ScoreRecord(first_model, 'mse', (0.5, 2.0)))
        store.add(ScoreRecord(second_model, 'mse', (0.75, 3.0)))
        store.update_group(
            averaging_group,
            (0.625, 2.5),
            (0.305, 1.22),
        )

    return pointwise_group, averaging_group


def test_low_level_prediction_helper_accepts_array_like_and_actuals() -> None:
    ax = plot_bias_and_variance(
        [0, 1],
        pd.Series([1.5, 2.0]),
        np.array([0.2, 0.4]),
        {'title': 'Predictions', 'color': 'purple'},
        actual_values=(1.0, 3.0),
    )

    assert isinstance(ax, Axes)
    assert ax.get_title() == 'Predictions'
    assert ax.lines[0].get_label() == 'Actual'
    assert [text.get_text() for text in ax.get_legend().get_texts()] == [
        'Actual',
        'Mean prediction ± prediction SD',
    ]


@pytest.mark.parametrize(
    ('args', 'message'),
    [
        (([0], [1, 2], [0.1, 0.2]), 'matching lengths'),
        (([[0]], [1], [0.1]), 'one-dimensional'),
        (([0], [1], [-0.1]), 'non-negative'),
        (([0], [np.nan], [0.1]), 'finite'),
    ],
)
def test_low_level_prediction_helper_rejects_invalid_values(args, message) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        plot_bias_and_variance(*args)


def test_error_component_helper_plots_metrics_and_their_means() -> None:
    ax = plot_error_components(
        (0, 1),
        (0.25, 1.0),
        (0.04, 0.16),
        {'primary_label': 'Squared bias'},
    )

    assert isinstance(ax, Axes)
    assert len(ax.lines) == 4
    assert [text.get_text() for text in ax.get_legend().get_texts()] == [
        'Squared bias',
        'Prediction variance',
    ]


def test_summary_helper_plots_paired_aggregate_bars() -> None:
    ax = plot_summary(
        ('Model', 'Sampling'),
        (0.25, 0.5),
        (0.1, 0.2),
        {'title': 'Summary', 'bar_width': 0.3},
        primary_label='Mean squared bias',
        variance_label='Mean pointwise model variance',
    )

    assert isinstance(ax, Axes)
    assert ax.get_title() == 'Summary'
    assert [patch.get_height() for patch in ax.patches] == pytest.approx(
        [0.25, 0.5, 0.1, 0.2]
    )


def test_prepare_plot_data_expands_records_and_outputs(tmp_path: Path) -> None:
    pointwise_group, averaging_group = _create_plot_database(
        tmp_path / 'results.sqlite3'
    )
    analyzer = BiasAnalyzer(tmp_path / 'results.sqlite3').select_run(RUN_ID)

    results = analyzer.get_bias_variance_plot_data()

    assert tuple(results.columns) == analyzer._PLOT_DATA_COLUMNS
    assert len(results) == 8
    assert set(results['output_name']) == {'y', 'z'}
    assert set(results['result_type']) == {'test_point', 'model'}

    point = results.loc[
        (results['group_id'] == pointwise_group)
        & (results['test_set_position'] == 0)
        & (results['output_name'] == 'z')
    ].iloc[0]
    assert point['actual_value'] == pytest.approx(10.0)
    assert point['mean_prediction'] == pytest.approx(11.0)
    assert point['squared_bias'] == pytest.approx(1.0)
    assert point['prediction_std'] == pytest.approx(0.3)
    assert np.isnan(point['mse'])

    model = results.loc[
        (results['group_id'] == averaging_group)
        & (results['record_index'] == 1)
        & (results['output_name'] == 'z')
    ].iloc[0]
    assert model['mean_prediction'] == pytest.approx(18.0)
    assert model['mse'] == pytest.approx(3.0)
    assert model['prediction_std'] == pytest.approx(1.2)
    assert np.isnan(model['actual_value'])


def test_prepare_plot_data_requires_selection_and_decomposition(
    tmp_path: Path,
) -> None:
    analyzer = BiasAnalyzer(tmp_path / 'results.sqlite3')
    with pytest.raises(RuntimeError, match='No run is selected'):
        analyzer.get_bias_variance_plot_data()

    with ResultStore(analyzer.db_path) as store:
        store.create_tables()
        store.add(
            RunRecord(
                RUN_ID,
                datetime.now(UTC),
                1,
                0.2,
                ('mse',),
                'adam',
                0.001,
                'mse',
                1,
                4,
                'cpu',
                ('x',),
                ('y',),
                (4,),
            )
        )
        study_id = store.add(StudyRecord(RUN_ID, 'model', 'averaging'))
        store.add(GroupRecord(study_id, 'wide'))

    analyzer.select_run(RUN_ID)
    with pytest.raises(RuntimeError, match='not been fully decomposed'):
        analyzer.get_bias_variance_plot_data()


@pytest.mark.parametrize(
    ('column', 'serialized_value', 'message'),
    [
        ('model_mean_prediction', '[1.0]', 'contain 2 output values'),
        ('model_variance_prediction', '[-1.0, 1.0]', 'negative values'),
    ],
)
def test_prepare_plot_data_rejects_invalid_persisted_vectors(
    tmp_path: Path,
    column: str,
    serialized_value: str,
    message: str,
) -> None:
    _create_plot_database(tmp_path / 'results.sqlite3')
    with sqlite3.connect(tmp_path / 'results.sqlite3') as connection:
        connection.execute(
            f'UPDATE models SET {column} = ? WHERE model_id = '
            '(SELECT MIN(model_id) FROM models)',
            (serialized_value,),
        )

    analyzer = BiasAnalyzer(tmp_path / 'results.sqlite3').select_run(RUN_ID)
    with pytest.raises(ValueError, match=message):
        analyzer.get_bias_variance_plot_data()


def test_components_layout_creates_two_panels_per_group(tmp_path: Path) -> None:
    pointwise_group, averaging_group = _create_plot_database(
        tmp_path / 'results.sqlite3'
    )
    analyzer = BiasAnalyzer(tmp_path / 'results.sqlite3').select_run(RUN_ID)
    results = analyzer.get_bias_variance_plot_data()

    plots = analyzer.plot_bias_and_variance(results, output='z')

    assert [plot.group_id for plot in plots] == [
        pointwise_group,
        averaging_group,
    ]
    assert all(plot.metric_axes is not None for plot in plots)
    assert plots[0].figure._suptitle.get_text() == 'POINTWISE: wide'
    assert plots[1].figure._suptitle.get_text() == 'AVERAGING: small'
    assert plots[0].prediction_axes.get_ylabel() == 'z [1]'
    assert plots[0].metric_axes.get_xlabel() == 'Test-set position'
    assert plots[1].metric_axes.get_xlabel() == 'Model number'
    assert [text.get_text() for text in plots[1].metric_axes.get_legend().texts] == [
        'Model MSE',
        'Within-model prediction variance',
    ]


def test_error_relationship_uses_error_metric_on_x_axis(tmp_path: Path) -> None:
    _create_plot_database(tmp_path / 'results.sqlite3')
    analyzer = BiasAnalyzer(tmp_path / 'results.sqlite3').select_run(RUN_ID)
    results = analyzer.get_bias_variance_plot_data()

    plots = analyzer.plot_bias_and_variance(
        results,
        output=0,
        plot_kind='error_relationship',
        max_plots=1,
        plot_settings={'figsize': (8, 5)},
    )

    assert len(plots) == 1
    assert plots[0].metric_axes is None
    assert plots[0].prediction_axes.get_xlabel() == 'Squared bias'
    assert tuple(plots[0].figure.get_size_inches()) == pytest.approx((8, 5))
    np.testing.assert_allclose(
        plots[0].prediction_axes.lines[0].get_xdata(),
        (0.25, 1.0),
    )


def test_summary_aggregates_all_groups_and_outputs(tmp_path: Path) -> None:
    _create_plot_database(tmp_path / 'results.sqlite3')
    analyzer = BiasAnalyzer(tmp_path / 'results.sqlite3').select_run(RUN_ID)
    results = analyzer.get_bias_variance_plot_data()

    axes = analyzer.plot_summary(results)

    pointwise = axes[EvaluationMethod.POINTWISE]
    averaging = axes[EvaluationMethod.AVERAGING]
    assert pointwise.get_title() == 'POINTWISE Summary — All outputs'
    assert averaging.get_title() == 'AVERAGING Summary — All outputs'
    assert [patch.get_height() for patch in pointwise.patches] == pytest.approx(
        [0.625, 0.135]
    )
    assert [patch.get_height() for patch in averaging.patches] == pytest.approx(
        [1.5625, 0.7625]
    )
    assert [text.get_text() for text in averaging.get_legend().texts] == [
        'Mean model MSE (total-error proxy)',
        'Mean within-model prediction variance',
    ]


def test_summary_can_select_one_output_and_override_settings(
    tmp_path: Path,
) -> None:
    _create_plot_database(tmp_path / 'results.sqlite3')
    analyzer = BiasAnalyzer(tmp_path / 'results.sqlite3').select_run(RUN_ID)
    results = analyzer.get_bias_variance_plot_data()

    axes = analyzer.plot_summary(
        results,
        output='z',
        plot_settings={'pointwise': {'title': 'Selected output'}},
    )

    pointwise = axes[EvaluationMethod.POINTWISE]
    averaging = axes[EvaluationMethod.AVERAGING]
    assert pointwise.get_title() == 'Selected output'
    assert averaging.get_title() == 'AVERAGING Summary — z [1]'
    assert [patch.get_height() for patch in pointwise.patches] == pytest.approx(
        [0.625, 0.17]
    )
    assert [patch.get_height() for patch in averaging.patches] == pytest.approx(
        [2.5, 1.22]
    )


def test_plot_output_selection_is_explicit_and_validated(tmp_path: Path) -> None:
    _create_plot_database(tmp_path / 'results.sqlite3')
    analyzer = BiasAnalyzer(tmp_path / 'results.sqlite3').select_run(RUN_ID)
    results = analyzer.get_bias_variance_plot_data()

    with pytest.raises(TypeError, match='integer or string'):
        analyzer.plot_bias_and_variance(results, output=True)
    with pytest.raises(IndexError, match='Unknown output index'):
        analyzer.plot_bias_and_variance(results, output=2)
    with pytest.raises(KeyError, match='Unknown output name'):
        analyzer.plot_bias_and_variance(results, output='missing')
    with pytest.raises(ValueError, match='plot_kind'):
        analyzer.plot_bias_and_variance(results, output=0, plot_kind='unknown')

    ambiguous = results.copy()
    ambiguous['output_name'] = 'duplicate'
    with pytest.raises(ValueError, match='ambiguous'):
        analyzer.plot_bias_and_variance(ambiguous, output='duplicate')

    inconsistent = results.copy()
    inconsistent.loc[inconsistent.index[0], 'output_name'] = 'changed'
    with pytest.raises(ValueError, match='exactly one output_name'):
        analyzer.plot_bias_and_variance(inconsistent, output=0)


def test_plotting_does_not_mutate_explicit_results(tmp_path: Path) -> None:
    _create_plot_database(tmp_path / 'results.sqlite3')
    analyzer = BiasAnalyzer(tmp_path / 'results.sqlite3').select_run(RUN_ID)
    results = analyzer.get_bias_variance_plot_data()
    original = results.copy(deep=True)

    analyzer.plot_bias_and_variance(
        results,
        output='y',
        group_settings={1: {'prediction': {'color': 'purple'}}},
    )

    pd.testing.assert_frame_equal(results, original)
