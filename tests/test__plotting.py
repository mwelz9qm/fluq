import matplotlib
import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from bias_variance._plotting import (
    plot_mean_distribution,
    plot_prediction_means_by_r2_scores,
    plot_variance_contribution,
    plot_variance_distribution,
)


@pytest.fixture
def results_df() -> pd.DataFrame:
    return pd.DataFrame({
        'study': ['sampling', 'sampling', 'model', 'model'],
        'variable': ['bootstrap', 'lhs', 'wide', 'narrow'],
        'r2': [0.70, 0.80, 0.90, 0.95],
        'mean': [10.0, 20.0, 30.0, 40.0],
        'variance': [2.0, 6.0, 3.0, 1.0],
        'conf_interval_lower': [9.0, 18.0, 27.0, 36.0],
        'conf_interval_upper': [11.5, 23.0, 34.0, 45.0],
    })


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close('all')


def test_plot_variance_contribution(results_df):
    ax = plot_variance_contribution(
        results_df,
        settings={'title': 'Variance share', 'figsize': (8, 4)},
    )

    assert isinstance(ax, Axes)
    assert ax.get_title() == 'Variance share'
    assert ax.get_xlabel() == 'Study'
    assert ax.get_ylabel() == 'Proportion of Mean Prediction Variance'
    assert ax.get_ylim() == pytest.approx((0.0, 1.0))
    assert tuple(ax.figure.get_size_inches()) == pytest.approx((8.0, 4.0))

    # Every stacked bar represents one study and must total one.
    totals_by_x = {}
    for patch in ax.patches:
        x = patch.get_x()
        totals_by_x[x] = totals_by_x.get(x, 0.0) + patch.get_height()
    assert list(totals_by_x.values()) == pytest.approx([1.0, 1.0])


def test_plot_prediction_means_uses_training_result_confidence_intervals(
        results_df,
):
    ax = plot_prediction_means_by_r2_scores(results_df)

    assert isinstance(ax, Axes)
    assert ax.get_title() == 'Prediction Means by R2 Score'
    np.testing.assert_allclose(ax.lines[0].get_xdata(), results_df['r2'])
    np.testing.assert_allclose(ax.lines[0].get_ydata(), results_df['mean'])

    error_segments = ax.collections[0].get_segments()
    for index, segment in enumerate(error_segments):
        np.testing.assert_allclose(
            segment,
            [
                [
                    results_df.loc[index, 'r2'],
                    results_df.loc[index, 'conf_interval_lower'],
                ],
                [
                    results_df.loc[index, 'r2'],
                    results_df.loc[index, 'conf_interval_upper'],
                ],
            ],
        )


@pytest.mark.parametrize(
    ('column', 'intervals'),
    [
        ('confidence_interval', [(9.0, 11.0), (18.0, 22.0)]),
        ('confidence_intervals', [
            {'mean': (9.0, 11.0)},
            "{'mean': (18.0, 22.0)}",
        ]),
    ],
)
def test_plot_prediction_means_accepts_paired_interval_columns(column, intervals):
    results = pd.DataFrame({
        'r2': [0.8, 0.9],
        'mean': [10.0, 20.0],
        column: intervals,
    })

    ax = plot_prediction_means_by_r2_scores(results)

    assert len(ax.collections[0].get_segments()) == 2


def test_plot_prediction_means_supports_custom_columns_and_labels():
    results = pd.DataFrame({
        'score': [0.8],
        'prediction_average': [10.0],
        'low': [8.0],
        'high': [13.0],
    })
    settings = {
        'r2_col': 'score',
        'mean_col': 'prediction_average',
        'lower_bound_col': 'low',
        'upper_bound_col': 'high',
        'title': 'Custom title',
        'xlabel': 'Score',
        'ylabel': 'Average',
    }

    ax = plot_prediction_means_by_r2_scores(results, settings=settings)

    assert ax.get_title() == 'Custom title'
    assert ax.get_xlabel() == 'Score'
    assert ax.get_ylabel() == 'Average'


@pytest.mark.parametrize(
    ('plot_function', 'column', 'expected_title', 'expected_xlabel'),
    [
        (
            plot_variance_distribution,
            'variance',
            'Prediction Variance Distribution',
            'Prediction Variance',
        ),
        (
            plot_mean_distribution,
            'mean',
            'Mean Prediction Distribution',
            'Mean Prediction',
        ),
    ],
)
def test_distribution_plots(
        results_df,
        plot_function,
        column,
        expected_title,
        expected_xlabel,
):
    ax = plot_function(results_df, settings={'bins': 2})

    assert isinstance(ax, Axes)
    assert ax.get_title() == expected_title
    assert ax.get_xlabel() == expected_xlabel
    assert ax.get_ylabel() == 'Frequency'
    assert len(ax.patches) == 2
    assert sum(patch.get_height() for patch in ax.patches) == pytest.approx(
        results_df[column].count()
    )


def test_distribution_plot_supports_density_and_custom_column():
    results = pd.DataFrame({'average': [1.0, 2.0, 3.0, 4.0]})

    ax = plot_mean_distribution(
        results,
        settings={'mean_col': 'average', 'density': True, 'bins': 2},
    )

    assert ax.get_ylabel() == 'Density'


@pytest.mark.parametrize(
    ('plot_function', 'results', 'message'),
    [
        (
            plot_mean_distribution,
            pd.DataFrame({'variance': [1.0]}),
            "Results do not contain a 'mean' column.",
        ),
        (
            plot_variance_distribution,
            pd.DataFrame({'variance': ['not numeric']}),
            "Results do not contain any numeric 'variance' values.",
        ),
        (
            plot_prediction_means_by_r2_scores,
            pd.DataFrame({'r2': [0.9], 'mean': [10.0]}),
            'Results do not contain confidence intervals',
        ),
    ],
)
def test_plotting_rejects_missing_or_non_numeric_data(
        plot_function,
        results,
        message,
):
    with pytest.raises(ValueError, match=message):
        plot_function(results)


def test_prediction_mean_must_be_inside_confidence_interval():
    results = pd.DataFrame({
        'r2': [0.9],
        'mean': [10.0],
        'conf_interval_lower': [11.0],
        'conf_interval_upper': [12.0],
    })

    with pytest.raises(ValueError, match='must contain its prediction mean'):
        plot_prediction_means_by_r2_scores(results)
