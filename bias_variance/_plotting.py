import ast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes


def plot_variance_contribution(
        results_df: pd.DataFrame,
        *,
        study_col: str = 'study',
        variable_col: str = 'variable',
        variance_col: str = 'variance',
        settings: dict | None = None,
) -> Axes:
    settings = settings or {}

    contribution_df = (
        results_df
        .groupby([study_col, variable_col])[variance_col]
        .mean()
        .unstack(fill_value=0)
    )

    totals = contribution_df.sum(axis=1)
    contribution_df = contribution_df.div(
        totals.replace(0, float('nan')),
        axis=0,
    ).fillna(0)

    ax = contribution_df.plot(
        kind='bar',
        stacked=True,
        figsize=settings.get('figsize', (10, 6)),
        colormap=settings.get('colormap', 'tab20'),
    )

    ax.set_title(
        settings.get('title', 'Variance Contribution by Study')
    )

    ax.set_xlabel(settings.get('xlabel', 'Study'))
    ax.set_ylabel(
        settings.get('ylabel', 'Proportion of Mean Prediction Variance')
    )
    ax.set_ylim(0, 1)

    return ax


def _numeric_values(results_df: pd.DataFrame, column: str) -> pd.Series:
    """Return a result column as finite numeric values."""
    if column not in results_df.columns:
        raise ValueError(f"Results do not contain a '{column}' column.")

    values = pd.to_numeric(results_df[column], errors='coerce')
    if values.notna().sum() == 0:
        raise ValueError(f"Results do not contain any numeric '{column}' values.")
    return values


def plot_prediction_means_by_r2_scores(
        results_df: pd.DataFrame,
        *,
        settings: dict | None = None,
) -> Axes:
    """Plot prediction means against R2 scores with CI error bars."""
    settings = settings or {}
    r2_col = settings.get('r2_col', 'r2')
    mean_col = settings.get('mean_col', 'mean')
    lower_col = settings.get('conf_interval_lower_col', 'conf_interval_lower')
    upper_col = settings.get('conf_interval_upper_col', 'conf_interval_upper')

    r2_scores = _numeric_values(results_df, r2_col)
    means = _numeric_values(results_df, mean_col)
    lower = _numeric_values(results_df, lower_col)
    upper = _numeric_values(results_df, upper_col)
    valid = r2_scores.notna() & means.notna() & lower.notna() & upper.notna()
    if not valid.any():
        raise ValueError('No rows contain an R2 score, mean, and confidence interval lower and upper bounds.')

    x = r2_scores[valid].to_numpy()
    y = means[valid].to_numpy()
    lower_errors = y - lower[valid].to_numpy()
    upper_errors = upper[valid].to_numpy() - y
    if (lower_errors < 0).any() or (upper_errors < 0).any():
        raise ValueError('Each confidence interval must contain its prediction mean.')

    _, ax = plt.subplots(figsize=settings.get('figsize', (10, 6)))
    ax.errorbar(
        x,
        y,
        yerr=np.vstack((lower_errors, upper_errors)),
        fmt=settings.get('fmt', 'o'),
        color=settings.get('color', 'tab:blue'),
        ecolor=settings.get('error_color', 'tab:gray'),
        capsize=settings.get('capsize', 4),
        alpha=settings.get('alpha', 0.8),
    )
    ax.set_title(settings.get('title', 'Prediction Means by R2 Score'))
    ax.set_xlabel(settings.get('xlabel', 'R2 Score'))
    ax.set_ylabel(settings.get('ylabel', 'Mean Prediction'))
    ax.grid(settings.get('grid', True), alpha=0.25)
    return ax


def _plot_distribution(
        results_df: pd.DataFrame,
        column: str,
        default_title: str,
        default_xlabel: str,
        settings: dict | None,
) -> Axes:
    settings = settings or {} # check plot settings, set as defaults if None
    values = _numeric_values(results_df, column).dropna() # get numeric values from results_df for specified column

    # define the subplot(s) and set values from settings.
    _, ax = plt.subplots(figsize=settings.get('figsize', (10, 6)))
    ax.hist(
        values,
        bins=settings.get('bins', 'auto'),
        color=settings.get('color', 'tab:blue'),
        alpha=settings.get('alpha', 0.75),
        edgecolor=settings.get('edgecolor', 'black'),
        density=settings.get('density', False),
    )
    ax.set_title(settings.get('title', default_title))
    ax.set_xlabel(settings.get('xlabel', default_xlabel))
    ax.set_ylabel(
        settings.get(
            'ylabel',
            'Density' if settings.get('density', False) else 'Frequency',
        )
    )
    ax.grid(settings.get('grid', True), axis='y', alpha=0.25)
    return ax


def plot_variance_distribution(
        results_df: pd.DataFrame,
        *,
        settings: dict | None = None,
) -> Axes:
    """Plot the distribution of prediction variances in the results."""
    column = (settings or {}).get('variance_col', 'variance')
    return _plot_distribution(
        results_df,
        column,
        'Prediction Variance Distribution',
        'Prediction Variance',
        settings,
    )


def plot_mean_distribution(
        results_df: pd.DataFrame,
        *,
        settings: dict | None = None,
) -> Axes:
    """Plot the distribution of mean predictions in the results."""
    column = (settings or {}).get('mean_col', 'mean')
    return _plot_distribution(
        results_df,
        column,
        'Mean Prediction Distribution',
        'Mean Prediction',
        settings,
    )
