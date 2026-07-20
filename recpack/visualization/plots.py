# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE

from typing import Optional, Sequence

import pandas as pd


def _get_axes(ax=None, figsize=None):
    if ax is not None:
        return ax

    import matplotlib.pyplot as plt

    _, ax = plt.subplots(figsize=figsize)
    return ax


def plot_metric_comparison(metrics: pd.DataFrame, metrics_to_plot: Optional[Sequence[str]] = None, ax=None,
                           figsize=None):
    df = metrics.copy()
    if metrics_to_plot is not None:
        df = df.loc[:, list(metrics_to_plot)]

    ax = _get_axes(ax, figsize)
    df.plot(kind="bar", ax=ax)
    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Metric value")
    ax.set_title("Metric comparison")
    ax.legend(title="Metric")
    ax.tick_params(axis="x", labelrotation=30)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    ax.figure.tight_layout()
    return ax


def plot_metric_distribution(metric_results: pd.DataFrame, metric: Optional[str] = None,
                             algorithm: Optional[str] = None, ax=None, figsize=None):
    df = metric_results.copy()
    if metric is not None:
        df = df[df["metric"] == metric]
    elif df["metric"].nunique() > 1:
        raise ValueError("metric must be provided when multiple metrics are present.")

    if algorithm is not None:
        df = df[df["algorithm"] == algorithm]

    if df.empty:
        raise ValueError("No metric results available for the selected filters.")

    ax = _get_axes(ax, figsize)
    df.boxplot(column="score", by="algorithm", ax=ax)
    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Score")
    ax.set_title(f"{df['metric'].iloc[0]} distribution")
    ax.tick_params(axis="x", labelrotation=30)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    ax.figure.suptitle("")
    ax.figure.tight_layout()
    return ax


def plot_recommendation_popularity(recommendations: pd.DataFrame, K: Optional[int] = None,
                                   algorithm: Optional[str] = None, top_n: int = 20, ax=None, figsize=None, ):
    if top_n <= 0:
        raise ValueError("top_n should be a positive integer.")

    df = recommendations.copy()
    if K is not None:
        df = df[df["rank"] <= K]
    if algorithm is not None:
        df = df[df["algorithm"] == algorithm]

    if df.empty:
        raise ValueError("No recommendations available for the selected filters.")

    counts = df["item_id"].value_counts().head(top_n).sort_values()

    ax = _get_axes(ax, figsize)
    counts.plot(kind="barh", ax=ax)
    ax.set_xlabel("Recommendation count")
    ax.set_ylabel("Item")
    ax.set_title("Most recommended items")
    ax.figure.tight_layout()
    return ax
