"""
Visualizations for a single Runner run (RunResults) and for comparing
metrics across an ensemble of runs.
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_single_run(sim, results, savepath=None, show=True, cumulative=False):
    """
    Daily new infections / diagnoses / sequences, GT vs D trigger points,
    the detection day, and intervention start days. Set cumulative=True
    to additionally overlay cumulative curves.
    """
    days = np.arange(sim.npts)
    new_inf = sim.results["new_infections"]
    diag = np.array(results.diagnoses)
    seqs = np.array(results.sequences)

    alpha = 0.6 if cumulative else 1.0

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(days, new_inf, label="New infections", color="tab:blue", alpha=alpha)
    ax.plot(days, diag, label="Diagnoses", color="tab:orange", alpha=alpha)
    ax.plot(days, seqs, label="Sequences", color="tab:green", alpha=alpha)

    if cumulative:
        ax.plot(days, np.cumsum(new_inf), label="Cum infections", color="tab:blue")
        ax.plot(days, np.cumsum(diag), label="Cum diagnoses", color="tab:orange")
        ax.plot(days, np.cumsum(seqs), label="Cum sequences", color="tab:green")

    gt = np.array(results.gt_triggers, dtype=int)
    d = np.array(results.diag_triggers, dtype=int)
    if len(gt):
        ax.scatter(gt, new_inf[gt], color="red", s=40, label="GT triggers", zorder=5)
    if len(d):
        ax.scatter(d, diag[d], color="blue", s=40, label="D triggers", zorder=5)

    if results.detection_day is not None:
        ax.axvline(results.detection_day, color="red", linestyle="--",
                    label=f"Detection day = {results.detection_day}")
    for day in results.intervention_days:
        ax.axvline(day, color="purple", linestyle="--", alpha=0.7)

    ax.set_xlabel("Day")
    ax.set_ylabel("Count")
    ax.set_title("Run timeline: infections, diagnoses, sequences")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if savepath:
        plt.savefig(savepath, dpi=200)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_ensemble_boxplots(runs, keys, savepath=None, show=True):
    """
    runs: list of dicts (e.g. one per seed)
    keys: metric names to plot, one subplot each
    """
    fig, axes = plt.subplots(1, len(keys), figsize=(5 * len(keys), 4))
    if len(keys) == 1:
        axes = [axes]

    for ax, key in zip(axes, keys):
        values = [r[key] for r in runs if r.get(key) is not None]
        if values:
            ax.boxplot(values)
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_ylabel(key)
        ax.set_title(key)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=200)
    if show:
        plt.show()
    else:
        plt.close(fig)


def grouped_boxplot(results_by_policy, metric_key, ylabel, policy_names=None,
                     savepath=None, show=True):
    """
    One boxplot per policy for a single metric -- replaces the ad hoc
    version that used to live inside the W6 notebook.
    results_by_policy: {policy_name: [ {metric_key: value, ...}, ... ]}
    """
    policy_names = policy_names or list(results_by_policy.keys())
    data = [
        [r[metric_key] for r in results_by_policy[p] if r.get(metric_key) is not None]
        for p in policy_names
    ]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.boxplot(data, tick_labels=policy_names)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{metric_key} by policy")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if savepath:
        plt.savefig(savepath, dpi=200)
    if show:
        plt.show()
    else:
        plt.close(fig)