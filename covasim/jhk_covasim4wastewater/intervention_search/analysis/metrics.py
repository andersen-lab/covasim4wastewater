"""
Summary metrics: outbreak impact (from sim + intervention), detection /
intervention timing (from a Runner's RunResults), and a generic ensemble
summarizer.

This replaces four old files: detection_delay.py, intervention_timing.py,
outbreak_impact.py, ensemble_stats.py.
"""

import numpy as np


# --- outbreak impact (from sim + intervention) ------------------------------

def peak_infections(sim):
    return int(np.max(sim.results["new_infections"]))


def total_infections(sim):
    return int(sim.results["new_infections"].values.sum())


def outbreak_duration(sim):
    inf = sim.results["new_infections"]
    for t in range(len(inf) - 1, -1, -1):
        if inf[t] > 0:
            return t
    return 0


def infections_after_intervention(sim, interv):
    if interv.intervention_day is None:
        return None
    return int(np.sum(sim.results["new_infections"][interv.intervention_day:]))


def summarize_outbreak(sim, interv):
    return {
        "peak_infections": peak_infections(sim),
        "total_infections": total_infections(sim),
        "outbreak_duration": outbreak_duration(sim),
        "infections_after_intervention": infections_after_intervention(sim, interv),
    }


# --- detection / intervention timing (from RunResults) ----------------------

def detection_success(results, window=14):
    if results.detection_day is None:
        return None
    return 1.0 if results.detection_day <= window else 0.0


def intervention_success(results, max_delay=7):
    if results.detection_day is None or results.intervention_day is None:
        return None
    return 1.0 if results.intervention_day <= results.detection_day + max_delay else 0.0


def summarize_detection(results, window=14):
    return {
        "first_detection_day": results.detection_day,
        "detection_delay": results.avg_resolution_time,
        "detection_success_window": detection_success(results, window=window),
    }


def summarize_intervention(results, max_delay=7):
    delay = None
    if results.detection_day is not None and results.intervention_day is not None:
        delay = results.intervention_day - results.detection_day

    return {
        "detection_day": results.detection_day,
        "first_intervention_day": results.intervention_day,
        "intervention_delay": delay,
        "intervention_success": intervention_success(results, max_delay=max_delay),
    }


# --- ensemble summary ---------------------------------------------------------

def summarize_ensemble(runs, key):
    """runs: list of dicts; key: metric name, e.g. 'detection_delay'."""
    values = [r[key] for r in runs if r.get(key) is not None]
    if not values:
        return None
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "q25": float(np.percentile(values, 25)),
        "q75": float(np.percentile(values, 75)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "n": len(values),
    }