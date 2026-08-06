import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

def plot_prediction_comparison(
    x,
    actual,
    prediction_mean,
    prediction_error,
    *,
    ax: Axes | None = None,
    settings: Mapping[str, object] | None = None,
) -> Axes:
    ax.scatter(x, actual, label="Actual", marker="x")

    ax.errorbar(
        x,
        prediction_mean,
        yerr=prediction_error,
        fmt="o",
        capsize=4,
        label="Mean prediction",
    )

def plot_bias_variance(
    labels,
    bias,
    variance,
    *,
    ax: Axes | None = None,
    settings: Mapping[str, object] | None = None,
) -> Axes:
    positions = np.arange(len(labels))
    width = 0.4

    ax.bar(
        positions - width / 2,
        bias,
        width,
        label="Mean squared bias",
    )
    ax.bar(
        positions + width / 2,
        variance,
        width,
        label="Mean prediction variance",
    )

    ax.set_xticks(positions, labels)