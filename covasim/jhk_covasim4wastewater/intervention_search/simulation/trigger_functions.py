"""
Diagnosis-based (and true-infection-based) trigger rules used to decide
when a policy's intervention should fire.
"""

from typing import Sequence

import numpy as np


def _to_array(x: Sequence[float]) -> np.ndarray:
    return np.asarray(x, dtype=float)


def relative_threshold(series, t, proportion, pop_size, window=1):
    arr = _to_array(series)
    if t < window - 1:
        return False
    avg = arr[t - window + 1:t + 1].mean()
    return (avg / pop_size) >= proportion


def weekly_growth(series, t, ratio, window=7):
    arr = _to_array(series)
    if t < 2 * window - 1:
        return False
    curr = arr[t - window + 1:t + 1].mean()
    prev = arr[t - 2 * window + 1:t - window + 1].mean()
    if prev <= 0:
        return False
    return (curr / prev) >= ratio


def slope_trigger(series, t, slope_threshold):
    arr = _to_array(series)
    if t < 1:
        return False
    return (arr[t] - arr[t - 1]) >= slope_threshold


def sustained_increase(series, t, days=5):
    arr = _to_array(series)
    if t < days:
        return False
    return all(arr[t - i] > arr[t - i - 1] for i in range(days))


def check_triggers(t, series, trigger_list, sim=None):
    """
    trigger_list = [
        ("weekly_growth", {"ratio": 1.1}),
        ("slope_trigger", {"slope_threshold": 5}),
        ("sustained_increase", {"days": 5}),
        ("relative_threshold", {"proportion": 0.0001, "window": 3}),
    ]
    Returns True as soon as any trigger in the list fires.
    """
    for name, pars in trigger_list:
        if name == "weekly_growth":
            if weekly_growth(series, t, pars["ratio"], pars.get("window", 7)):
                return True
        elif name == "slope_trigger":
            if slope_trigger(series, t, pars["slope_threshold"]):
                return True
        elif name == "sustained_increase":
            if sustained_increase(series, t, pars.get("days", 5)):
                return True
        elif name == "relative_threshold":
            if sim is None:
                raise ValueError("relative_threshold requires sim")
            if relative_threshold(series, t, pars["proportion"], sim.pars["pop_size"], pars.get("window", 1)):
                return True
        else:
            raise ValueError(f"Unknown trigger: {name}")
    return False