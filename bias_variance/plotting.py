"""Matplotlib helpers for prepared bias/variance result data."""

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from numpy.typing import ArrayLike


def _numeric_1d(values: ArrayLike, *, name: str) -> np.ndarray:
    """Return a nonempty, finite, one-dimensional float array."""
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


def _matching_numeric_arrays(
    values: Mapping[str, ArrayLike],
) -> dict[str, np.ndarray]:
    arrays = {
        name: _numeric_1d(value, name=name)
        for name, value in values.items()
    }
    if len({len(array) for array in arrays.values()}) != 1:
        names = ', '.join(values)
        raise ValueError(f'{names} must have matching lengths.')
    return arrays


def _resolve_axes(
    ax: Axes | None,
    plot_settings: Mapping[str, object],
) -> Axes:
    if ax is not None:
        if not isinstance(ax, Axes):
            raise TypeError('ax must be a matplotlib Axes or None.')
        return ax

    _, resolved_ax = plt.subplots(
        figsize=plot_settings.get('figsize', (10, 6)),
    )
    return resolved_ax


def plot_bias_and_variance(
    x_values: ArrayLike,
    y_values: ArrayLike,
    y_error_bar_values: ArrayLike,
    plot_settings: Mapping[str, object] | None = None,
    *,
    actual_values: ArrayLike | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Plot mean predictions with standard-deviation error bars.

    ``actual_values`` is optional because pointwise results have a shared
    actual value at each position, while averaging results summarize models
    evaluated over sets of observations.
    """
    if plot_settings is not None and not isinstance(plot_settings, Mapping):
        raise TypeError('plot_settings must be a mapping or None.')
    settings = plot_settings or {}
    inputs: dict[str, ArrayLike] = {
        'x_values': x_values,
        'y_values': y_values,
        'y_error_bar_values': y_error_bar_values,
    }
    if actual_values is not None:
        inputs['actual_values'] = actual_values
    arrays = _matching_numeric_arrays(inputs)

    errors = arrays['y_error_bar_values']
    if (errors < 0).any():
        raise ValueError(
            'y_error_bar_values must contain non-negative values.'
        )

    resolved_ax = _resolve_axes(ax, settings)
    if actual_values is not None:
        resolved_ax.plot(
            arrays['x_values'],
            arrays['actual_values'],
            marker=settings.get('actual_marker', 'x'),
            linestyle=settings.get('actual_linestyle', '-'),
            color=settings.get('actual_color', 'black'),
            alpha=float(settings.get('actual_alpha', 0.8)),
            label=str(settings.get('actual_label', 'Actual')),
        )

    resolved_ax.errorbar(
        arrays['x_values'],
        arrays['y_values'],
        yerr=errors,
        fmt=str(settings.get('marker', 'o')),
        linestyle=str(settings.get('linestyle', 'none')),
        color=settings.get('color', 'tab:blue'),
        ecolor=settings.get('error_color', 'tab:gray'),
        capsize=float(settings.get('capsize', 4)),
        markersize=float(settings.get('markersize', 5)),
        alpha=float(settings.get('alpha', 0.75)),
        label=str(
            settings.get(
                'label',
                'Mean prediction ± prediction SD',
            )
        ),
    )
    resolved_ax.set_title(str(settings.get('title', 'Prediction Summary')))
    resolved_ax.set_xlabel(str(settings.get('xlabel', 'Record')))
    resolved_ax.set_ylabel(str(settings.get('ylabel', 'Output value')))
    resolved_ax.set_xscale(str(settings.get('xscale', 'linear')))
    resolved_ax.set_yscale(str(settings.get('yscale', 'linear')))
    resolved_ax.grid(
        bool(settings.get('grid', True)),
        alpha=float(settings.get('grid_alpha', 0.25)),
    )
    if bool(settings.get('legend', True)):
        resolved_ax.legend()
    return resolved_ax


def plot_error_components(
    x_values: ArrayLike,
    primary_error_values: ArrayLike,
    variance_values: ArrayLike,
    plot_settings: Mapping[str, object] | None = None,
    *,
    ax: Axes | None = None,
) -> Axes:
    """Plot a primary error metric and variance against record position."""
    if plot_settings is not None and not isinstance(plot_settings, Mapping):
        raise TypeError('plot_settings must be a mapping or None.')
    settings = plot_settings or {}
    arrays = _matching_numeric_arrays(
        {
            'x_values': x_values,
            'primary_error_values': primary_error_values,
            'variance_values': variance_values,
        }
    )
    primary = arrays['primary_error_values']
    variance = arrays['variance_values']
    if (primary < 0).any() or (variance < 0).any():
        raise ValueError(
            'primary_error_values and variance_values must be non-negative.'
        )

    resolved_ax = _resolve_axes(ax, settings)
    resolved_ax.plot(
        arrays['x_values'],
        primary,
        marker=settings.get('primary_marker', 'o'),
        linestyle=settings.get('primary_linestyle', '-'),
        color=settings.get('primary_color', 'tab:orange'),
        alpha=float(settings.get('alpha', 0.8)),
        label=str(settings.get('primary_label', 'Squared bias')),
    )
    resolved_ax.plot(
        arrays['x_values'],
        variance,
        marker=settings.get('variance_marker', 's'),
        linestyle=settings.get('variance_linestyle', '-'),
        color=settings.get('variance_color', 'tab:blue'),
        alpha=float(settings.get('alpha', 0.8)),
        label=str(settings.get('variance_label', 'Prediction variance')),
    )

    if bool(settings.get('show_means', True)):
        resolved_ax.axhline(
            float(primary.mean()),
            color=settings.get('primary_color', 'tab:orange'),
            linestyle='--',
            alpha=0.5,
        )
        resolved_ax.axhline(
            float(variance.mean()),
            color=settings.get('variance_color', 'tab:blue'),
            linestyle='--',
            alpha=0.5,
        )

    resolved_ax.set_title(str(settings.get('title', 'Error Components')))
    resolved_ax.set_xlabel(str(settings.get('xlabel', 'Record')))
    resolved_ax.set_ylabel(str(settings.get('ylabel', 'Squared output value')))
    resolved_ax.set_xscale(str(settings.get('xscale', 'linear')))
    resolved_ax.set_yscale(str(settings.get('yscale', 'linear')))
    resolved_ax.grid(
        bool(settings.get('grid', True)),
        alpha=float(settings.get('grid_alpha', 0.25)),
    )
    if bool(settings.get('legend', True)):
        resolved_ax.legend()
    return resolved_ax


def plot_summary(
    labels: Sequence[object],
    primary_metric_values: ArrayLike,
    variance_values: ArrayLike,
    plot_settings: Mapping[str, object] | None = None,
    *,
    primary_label: str,
    variance_label: str,
    ax: Axes | None = None,
) -> Axes:
    """Plot paired aggregate error and variance bars for named studies."""
    if plot_settings is not None and not isinstance(plot_settings, Mapping):
        raise TypeError('plot_settings must be a mapping or None.')
    settings = plot_settings or {}
    label_values = tuple(str(label) for label in labels)
    if not label_values:
        raise ValueError('labels must not be empty.')

    arrays = _matching_numeric_arrays(
        {
            'primary_metric_values': primary_metric_values,
            'variance_values': variance_values,
        }
    )
    primary = arrays['primary_metric_values']
    variance = arrays['variance_values']
    if len(label_values) != len(primary):
        raise ValueError(
            'labels, primary_metric_values, and variance_values must have '
            'matching lengths.'
        )
    if (primary < 0).any() or (variance < 0).any():
        raise ValueError(
            'primary_metric_values and variance_values must be non-negative.'
        )

    resolved_ax = _resolve_axes(ax, settings)
    positions = np.arange(len(label_values))
    width = float(settings.get('bar_width', 0.4))
    if not 0 < width <= 1:
        raise ValueError('bar_width must be greater than zero and at most one.')

    resolved_ax.bar(
        positions - width / 2,
        primary,
        width,
        color=settings.get('primary_color', 'tab:orange'),
        label=primary_label,
    )
    resolved_ax.bar(
        positions + width / 2,
        variance,
        width,
        color=settings.get('variance_color', 'tab:blue'),
        label=variance_label,
    )
    resolved_ax.set_xticks(positions, label_values)
    resolved_ax.set_title(str(settings.get('title', 'Bias/Variance Summary')))
    resolved_ax.set_xlabel(str(settings.get('xlabel', 'Study')))
    resolved_ax.set_ylabel(
        str(settings.get('ylabel', 'Mean squared output value'))
    )
    resolved_ax.set_yscale(str(settings.get('yscale', 'linear')))
    resolved_ax.grid(
        bool(settings.get('grid', True)),
        axis='y',
        alpha=float(settings.get('grid_alpha', 0.25)),
    )
    if bool(settings.get('legend', True)):
        resolved_ax.legend()
    if 'tick_rotation' in settings:
        resolved_ax.tick_params(
            axis='x',
            labelrotation=float(settings['tick_rotation']),
        )
    return resolved_ax
