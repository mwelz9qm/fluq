"""Tests for the current bias/variance plotting workflow."""

from pathlib import Path

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes

import bias_variance.analyzer as analyzer_module
from bias_variance.analyzer import BiasAnalyzer
from bias_variance.plotting import (
    plot_bias_variance,
    plot_prediction_comparison,
)

POINTWISE_GROUP_ID = 11
EXTREME_GROUP_ID = 12
AVERAGING_GROUP_ID = 21
RUN_ID = 'synthetic-run'

def _pointwise_results(n_points: int = 60) -> pd.DataFrame:
    """Build smooth, multi-output pointwise results."""
    x = np.linspace(-3.0, 3.0, n_points)
    second_input = x**2
    actual = 0.45 * x**3 - 1.2 * x + 2.0
    second_actual = 0.5 * actual + 1.0
    mean_error = 0.18 * np.sin(2.0 * x)
    prediction_mean = actual + mean_error
    second_prediction_mean = second_actual - 0.5 * mean_error
    variance = 0.04 + 0.025 * np.cos(x) ** 2
    second_variance = 0.02 + 0.5 * variance

    return pd.DataFrame(
        {
            'group_id': POINTWISE_GROUP_ID,
            'group_name': 'shared test set',
            # Each packed point contains (input vector, actual-output vector).
            'test_points': list(
                zip(
                    zip(x, second_input, strict=True),
                    zip(actual, second_actual, strict=True),
                    strict=True,
                )
            ),
            'prediction_mean': list(
                zip(
                    prediction_mean,
                    second_prediction_mean,
                    strict=True,
                )
            ),
            'bias': list(
                zip(
                    mean_error**2,
                    (-0.5 * mean_error) ** 2,
                    strict=True,
                )
            ),
            'variance': list(
                zip(variance, second_variance, strict=True)
            ),
        }
    )


def _averaging_results(n_models: int = 36) -> pd.DataFrame:
    """Build averaging results spanning a useful range of model R2 scores."""
    r2 = np.linspace(0.55, 0.98, n_models)
    sample_mean = 8.0 + 1.5 * np.sin(np.linspace(0.0, 2.5, n_models))
    mean_error = 0.35 * (1.0 - r2) * np.cos(np.arange(n_models) / 3.0)
    prediction_mean = sample_mean + mean_error
    variance = 0.08 + 0.35 * (1.0 - r2) ** 2

    return pd.DataFrame(
        {
            'group_id': AVERAGING_GROUP_ID,
            'group_name': 'architecture sweep',
            'model_ids': np.arange(1000, 1000 + n_models),
            'r2': r2,
            'sample_mean': sample_mean,
            'prediction_mean': prediction_mean,
            'bias': mean_error**2,
            'variance': variance,
        }
    )


def _extreme_results(n_points: int = 24) -> pd.DataFrame:
    """Build a group that should be excluded by the axis-range guard."""
    values = np.linspace(1.0e9, 1.1e9, n_points)
    return pd.DataFrame(
        {
            'group_id': EXTREME_GROUP_ID,
            'group_name': 'unstable model',
            'test_points': [((value,), (value,)) for value in values],
            'prediction_mean': values + 1.0e7,
            'bias': np.full(n_points, 1.0e14),
            'variance': np.full(n_points, 1.0e16),
        }
    )


class SyntheticResultStore:
    """ResultStore test double exposing deterministic plot-ready rows."""

    entered = 0
    exited = 0
    requested_groups: list[int] = []

    def __init__(self, database, *, timeout=5.0) -> None:
        self.database = database
        self.timeout = timeout

    def __enter__(self):
        type(self).entered += 1
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        type(self).exited += 1

    def create_tables(self) -> None:
        pass

    def get_recent_run(self) -> str:
        return RUN_ID

    def does_run_exist(self, run_id: str) -> bool:
        return run_id == RUN_ID

    def get_studies(self, run_id: str) -> tuple[int, ...]:
        assert run_id == RUN_ID
        return (101, 102)

    def get_groups(self, study_id: int) -> tuple[int, ...]:
        return {
            101: (POINTWISE_GROUP_ID, EXTREME_GROUP_ID),
            102: (AVERAGING_GROUP_ID,),
        }[study_id]

    def get_group_plot_results(self, group_id: int) -> pd.DataFrame:
        type(self).requested_groups.append(group_id)
        return {
            POINTWISE_GROUP_ID: _pointwise_results(),
            EXTREME_GROUP_ID: _extreme_results(),
            AVERAGING_GROUP_ID: _averaging_results(),
        }[group_id].copy()

    def get_method(self, group_id: int) -> str:
        return (
            'averaging'
            if group_id == AVERAGING_GROUP_ID
            else 'pointwise'
        )


@pytest.fixture(autouse=True)
def plotting_environment(monkeypatch):
    SyntheticResultStore.entered = 0
    SyntheticResultStore.exited = 0
    SyntheticResultStore.requested_groups = []
    monkeypatch.setattr(analyzer_module, 'ResultStore', SyntheticResultStore)
    yield
    plt.close('all')


def test_plot_prediction_comparison_draws_all_points_and_deviations():
    results = _pointwise_results()
    prepared = BiasAnalyzer._prepare_group_plot_data(
        results,
        input_index=0,
        output_index=0,
    )

    ax = plot_prediction_comparison(
        prepared['x'],
        prepared['actual'],
        prepared['prediction_mean'],
        np.sqrt(prepared['variance']),
    )

    assert isinstance(ax, Axes)
    assert len(ax.collections[0].get_offsets()) == len(results)
    np.testing.assert_allclose(
        ax.collections[0].get_offsets()[:, 0],
        prepared['x'],
    )
    np.testing.assert_allclose(
        ax.collections[0].get_offsets()[:, 1],
        prepared['actual'],
    )
    np.testing.assert_allclose(ax.lines[0].get_xdata(), prepared['x'])
    np.testing.assert_allclose(
        ax.lines[0].get_ydata(),
        prepared['prediction_mean'],
    )

    error_segments = ax.collections[1].get_segments()
    standard_deviation = np.sqrt(prepared['variance'].to_numpy())
    for index in (0, len(results) // 2, len(results) - 1):
        np.testing.assert_allclose(
            error_segments[index],
            [
                [prepared.loc[index, 'x'],
                 prepared.loc[index, 'prediction_mean']
                 - standard_deviation[index]],
                [prepared.loc[index, 'x'],
                 prepared.loc[index, 'prediction_mean']
                 + standard_deviation[index]],
            ],
        )


def test_plot_bias_variance_draws_paired_bars_for_each_group():
    labels = ('pointwise', 'averaging', 'sampling')
    bias = np.array([0.025, 0.010, 0.040])
    variance = np.array([0.080, 0.120, 0.060])

    ax = plot_bias_variance(labels, bias, variance)

    assert isinstance(ax, Axes)
    assert len(ax.patches) == 2 * len(labels)
    np.testing.assert_allclose(
        [patch.get_height() for patch in ax.patches[:len(labels)]],
        bias,
    )
    np.testing.assert_allclose(
        [patch.get_height() for patch in ax.patches[len(labels):]],
        variance,
    )
    assert tuple(label.get_text() for label in ax.get_xticklabels()) == labels


def test_plot_results_plots_pointwise_and_averaging_groups(
    tmp_path: Path,
):
    analyzer = BiasAnalyzer(tmp_path / 'synthetic.sqlite3')

    axes = analyzer.plot_results(
        RUN_ID,
        max_plots=10,
        max_axis_range=100.0,
    )

    project_root = Path(__file__).resolve().parents[2]
    plot_directory = project_root / 'plot_artifacts'
    plot_directory.mkdir(parents=True, exist_ok=True)

    names = (
        'pointwise_predictions',
        'pointwise_bias_variance',
        'averaging_predictions',
        'averaging_bias_variance',
    )

    for name, ax in zip(names, axes, strict=True):
        ax.figure.savefig(
            plot_directory / f'{name}.png',
            dpi=150,
            bbox_inches='tight',
        )

    # The extreme group is queried but excluded. Each accepted group returns
    # one comparison Axes and one bias/variance Axes.
    assert len(axes) == 4
    assert all(isinstance(ax, Axes) for ax in axes)
    assert SyntheticResultStore.requested_groups == [
        POINTWISE_GROUP_ID,
        EXTREME_GROUP_ID,
        AVERAGING_GROUP_ID,
    ]
    assert SyntheticResultStore.entered == 1
    assert SyntheticResultStore.exited == 1

    pointwise_ax, pointwise_bar_ax, averaging_ax, averaging_bar_ax = axes
    assert pointwise_ax.get_title() == 'shared test set (pointwise)'
    assert pointwise_ax.get_xlabel() == 'Test input'
    assert averaging_ax.get_title() == 'architecture sweep (averaging)'
    assert averaging_ax.get_xlabel() == 'R2 score'

    pointwise = _pointwise_results()
    averaging = _averaging_results()
    np.testing.assert_allclose(
        pointwise_ax.lines[0].get_ydata(),
        np.asarray(pointwise['prediction_mean'].tolist())[:, 0],
    )
    np.testing.assert_allclose(
        averaging_ax.lines[0].get_xdata(),
        averaging['r2'],
    )
    assert [patch.get_height() for patch in pointwise_bar_ax.patches] == (
        pytest.approx(
            [
                np.asarray(pointwise['bias'].tolist())[:, 0].mean(),
                np.asarray(pointwise['variance'].tolist())[:, 0].mean(),
            ]
        )
    )
    assert [patch.get_height() for patch in averaging_bar_ax.patches] == (
        pytest.approx(
            [averaging['bias'].mean(), averaging['variance'].mean()]
        )
    )


def test_plot_results_honors_plot_limit_and_can_save_figure(tmp_path: Path):
    analyzer = BiasAnalyzer(tmp_path / 'synthetic.sqlite3')

    axes = analyzer.plot_results(
        max_plots=1,
        max_axis_range=100.0,
        settings={
            'comparison': {
                'title': 'Synthetic pointwise predictions',
                'figsize': (9, 5),
            },
        },
    )

    assert len(axes) == 2
    assert axes[0].get_title() == 'Synthetic pointwise predictions'
    assert tuple(axes[0].figure.get_size_inches()) == pytest.approx((9.0, 5.0))
    assert SyntheticResultStore.requested_groups == [POINTWISE_GROUP_ID]

    output = tmp_path / 'pointwise-example.png'
    axes[0].figure.savefig(output)
    assert output.stat().st_size > 0


def test_plot_results_selects_requested_input_and_output(tmp_path: Path):
    analyzer = BiasAnalyzer(tmp_path / 'synthetic.sqlite3')

    axes = analyzer.plot_results(
        RUN_ID,
        max_plots=1,
        max_axis_range=100.0,
        input_index=1,
        output_index=1,
    )

    results = _pointwise_results()
    packed_points = results['test_points'].tolist()
    expected_x = np.array([point[0][1] for point in packed_points])
    expected_actual = np.array([point[1][1] for point in packed_points])
    expected_prediction = np.asarray(
        results['prediction_mean'].tolist()
    )[:, 1]

    np.testing.assert_allclose(
        axes[0].collections[0].get_offsets()[:, 0],
        expected_x,
    )
    np.testing.assert_allclose(
        axes[0].collections[0].get_offsets()[:, 1],
        expected_actual,
    )
    np.testing.assert_allclose(axes[0].lines[0].get_ydata(), expected_prediction)


@pytest.mark.parametrize(
    ('kwargs', 'exception', 'message'),
    [
        ({'max_plots': 0}, ValueError, 'max_plots must be positive'),
        (
            {'max_axis_range': 0.0},
            ValueError,
            'max_axis_range must be positive',
        ),
        ({'input_index': -1}, ValueError, 'input_index must be non-negative'),
        ({'output_index': 1.5}, TypeError, 'output_index must be an integer'),
    ],
)
def test_plot_results_rejects_invalid_limits_and_selections(
    tmp_path: Path,
    kwargs,
    exception,
    message,
):
    analyzer = BiasAnalyzer(tmp_path / 'synthetic.sqlite3')

    with pytest.raises(exception, match=message):
        analyzer.plot_results(RUN_ID, **kwargs)
