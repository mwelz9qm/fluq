"""Generic plotting utilities for bias and variance results."""

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes


def _numeric_1d(values, *, name: str) -> np.ndarray:
    # Return finite numeric values as a non-empty one-dimensional array.
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f'{name} must contain numeric values.') from exc

    if array.ndim != 1:
        raise ValueError(f'{name} must be one-dimensional.')
    if array.size == 0:
        raise ValueError(f'{name} must not be empty.')
    if not np.isfinite(array).all():
        raise ValueError(f'{name} must contain only finite values.')

    return array


def _resolve_axes(
    ax: Axes | None,
    settings: Mapping[str, object],
) -> Axes:
    if ax is not None:
        if not isinstance(ax, Axes):
            raise TypeError('ax must be a matplotlib Axes or None.')
        return ax

    _, resolved_ax = plt.subplots(
        figsize=settings.get('figsize', (10, 6)),
    )
    return resolved_ax


def plot_prediction_comparison(
    x,
    actual,
    prediction_mean,
    prediction_error,
    *,
    ax: Axes | None = None,
    settings: Mapping[str, object] | None = None,
) -> Axes:
    """
    Plot paired actual and mean-prediction values with error bars.

    'prediction_error' contains magnitudes, such as prediction standard
    deviations.  This function deliberately does not calculate them so it can
    visualize either standard deviations or standard errors.
    """
    resolved_settings = settings or {}
    x_values = _numeric_1d(x, name='x')
    actual_values = _numeric_1d(actual, name='actual')
    mean_values = _numeric_1d(prediction_mean, name='prediction_mean')
    error_values = _numeric_1d(prediction_error, name='prediction_error')

    lengths = {
        len(x_values),
        len(actual_values),
        len(mean_values),
        len(error_values),
    }
    if len(lengths) != 1:
        raise ValueError(
            'x, actual, prediction_mean, and prediction_error must have '
            'matching lengths.'
        )
    if (error_values < 0).any():
        raise ValueError('prediction_error must contain non-negative values.')

    resolved_ax = _resolve_axes(ax, resolved_settings)
    resolved_ax.scatter(
        x_values,
        actual_values,
        label=resolved_settings.get('actual_label', 'Actual'),
        marker=resolved_settings.get('actual_marker', 'x'),
        color=resolved_settings.get('actual_color', 'black'),
        alpha=resolved_settings.get('alpha', 0.8),
    )
    resolved_ax.errorbar(
        x_values,
        mean_values,
        yerr=error_values,
        fmt=resolved_settings.get('prediction_marker', 'o'),
        color=resolved_settings.get('prediction_color', 'tab:blue'),
        ecolor=resolved_settings.get('error_color', 'tab:gray'),
        capsize=resolved_settings.get('capsize', 4),
        alpha=resolved_settings.get('alpha', 0.8),
        label=resolved_settings.get('prediction_label', 'Mean prediction'),
    )
    resolved_ax.set_title(
        str(resolved_settings.get('title', 'Actual and Mean Predictions'))
    )
    resolved_ax.set_xlabel(str(resolved_settings.get('xlabel', 'Input')))
    resolved_ax.set_ylabel(str(resolved_settings.get('ylabel', 'Output')))
    resolved_ax.grid(
        bool(resolved_settings.get('grid', True)),
        alpha=0.25,
    )
    resolved_ax.legend()

    return resolved_ax


def plot_bias_variance(
    labels: Sequence[object],
    bias,
    variance,
    *,
    ax: Axes | None = None,
    settings: Mapping[str, object] | None = None,
) -> Axes:
    """Plot paired mean-squared-bias and prediction-variance bars."""
    resolved_settings = settings or {}
    label_values = tuple(str(label) for label in labels)
    if not label_values:
        raise ValueError('labels must not be empty.')

    bias_values = _numeric_1d(bias, name='bias')
    variance_values = _numeric_1d(variance, name='variance')
    if len(label_values) != len(bias_values) or len(label_values) != len(
        variance_values
    ):
        raise ValueError('labels, bias, and variance must have matching lengths.')
    if (bias_values < 0).any() or (variance_values < 0).any():
        raise ValueError('bias and variance must contain non-negative values.')

    resolved_ax = _resolve_axes(ax, resolved_settings)
    positions = np.arange(len(label_values))
    width = float(resolved_settings.get('bar_width', 0.4))
    if width <= 0:
        raise ValueError('bar_width must be positive.')

    resolved_ax.bar(
        positions - width / 2,
        bias_values,
        width,
        label=resolved_settings.get('bias_label', 'Mean squared bias'),
        color=resolved_settings.get('bias_color', 'tab:orange'),
    )
    resolved_ax.bar(
        positions + width / 2,
        variance_values,
        width,
        label=resolved_settings.get(
            'variance_label',
            'Mean prediction variance',
        ),
        color=resolved_settings.get('variance_color', 'tab:blue'),
    )
    resolved_ax.set_xticks(positions, label_values)
    resolved_ax.set_title(
        str(resolved_settings.get('title', 'Bias and Variance'))
    )
    resolved_ax.set_xlabel(str(resolved_settings.get('xlabel', 'Group')))
    resolved_ax.set_ylabel(str(resolved_settings.get('ylabel', 'Value')))
    resolved_ax.set_yscale(str(resolved_settings.get('yscale', 'linear')))
    resolved_ax.grid(
        bool(resolved_settings.get('grid', True)),
        axis='y',
        alpha=0.25,
    )
    resolved_ax.legend()

    return resolved_ax
