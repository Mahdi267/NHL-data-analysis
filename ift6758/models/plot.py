# Assisted by Claude to find some specific matplotlib and numpy functions

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import auc, roc_curve


def to_percentile(proba):
    # Convert probabilities to percentiles
    return rankdata(proba) / len(proba) * 100


def roc_plot(y, probas: dict, ax=None):
    # Create a new figure if no axis is provided
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    # Plot the ROC curve for each model
    for name, proba in probas.items():
        fpr, tpr, _ = roc_curve(y, proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc(fpr, tpr):.3f})")

    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    return ax


def goal_rate_plot(y, probas: dict, n_bins: int = 20, ax=None):
    # Create a new figure if no axis is provided
    if ax is None:
        _, ax = plt.subplots(figsize=(7,5))

    y = np.asarray(y)
    edges = np.linspace(0, 100, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2

    # Group shots by their probability percentile
    for name, proba in probas.items():
        percentile = to_percentile(proba)
        bins = np.clip(np.digitize(percentile, edges[1:-1], right=True), 0, n_bins - 1)
        goal_rate = [y[bins==b].mean() * 100 for b in range(n_bins)]
        ax.plot(centers, goal_rate, marker="o", markersize=3, label=name)

    ax.set_xlim(100, 0)  # highest probabilities on the left
    ax.set_xlabel("Shot probability model percentile")
    ax.set_ylabel("Goals / (Shots + Goals) (%)")
    ax.set_title("Goal rate")
    ax.legend()
    ax.grid(alpha=0.3)
    return ax


def cumulative_goals_plot(y, probas: dict, ax=None):
    # Create a new figure if no axis is provided
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    y = np.asarray(y)

    # Sort shots from highest to lowest probability
    for name, proba in probas.items():
        order = np.argsort(proba)[::-1] #highest proba first
        cumulative = np.cumsum(y[order]) / y.sum() * 100
        percentile = 100 - np.arange(1, len(order) + 1) / len(order) * 100
        ax.plot(percentile, cumulative, label=name)

    ax.set_xlim(100, 0)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Shot probability model percentile")
    ax.set_ylabel("Proportion of goals (%)")
    ax.set_title("Cumulative % of goals")
    ax.legend()
    ax.grid(alpha=0.3)
    return ax


def plot_calibration(y, probas: dict, n_bins: int = 10, ax=None):
    # Create a new figure if no axis is provided
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    # Plot the calibration curve for each model
    for name, proba in probas.items():
        CalibrationDisplay.from_predictions(y, proba, n_bins=n_bins, name=name, ax=ax)

    ax.set_title("Reliability diagram")
    ax.grid(alpha=0.3)
    return ax


def all_plot(y, probas: dict, out_dir: Path, prefix: str) -> dict:
    # Create the output directory if it does not exist
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Store all the plotting functions
    plot_functions = {
        "roc": roc_plot,
        "goal_rate": goal_rate_plot,
        "cumulative_goals": cumulative_goals_plot,
        "calibration": plot_calibration,
    }

    figures = {}

    # Generate and save all plots
    for fig_name, plot_function in plot_functions.items():
        ax = plot_function(y, probas)
        fig = ax.get_figure()
        fig.tight_layout()
        fig.savefig(out_dir / f"{prefix}_{fig_name}.png", dpi=150)
        figures[fig_name] = fig
 
    return figures