"""
Core simulation engine: builds a Covasim sim, layers an ObservationModel
and InterventionController on top of it, applies diagnosis-based triggers
day by day, and records results.

Policies are plain dicts (see configs/intervention_policies.yaml), e.g.:
    {
        "p_test": 0.35, "p_seq": 0.25, "delay_pmf": [...],
        "beta_reduction": 0.05, "beta_min": 0.005, "duration": 7,
        "quiet_period": 0, "max_interventions": 1,
        "trigger_list": [("weekly_growth", {"ratio": 1.1})],
    }
"""

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import covasim as cv
import numpy as np

from .intervention import InterventionController
from .observation_model import ObservationModel
from .trigger_functions import check_triggers


@dataclass
class RunConfig:
    sim_pars: Dict[str, Any]
    evo_pars: Optional[Dict[str, Any]] = None
    policy: Dict[str, Any] = None
    run_id: str = "run_000"
    out_dir: str = "results/raw"


@dataclass
class RunResults:
    run_id: str
    days: List[int] = field(default_factory=list)

    true_infections: List[int] = field(default_factory=list)
    diagnoses: List[int] = field(default_factory=list)
    sequences: List[int] = field(default_factory=list)
    cum_infections: List[int] = field(default_factory=list)
    beta: List[float] = field(default_factory=list)

    detection_day: Optional[int] = None
    intervention_day: Optional[int] = None

    gt_triggers: List[int] = field(default_factory=list)
    diag_triggers: List[int] = field(default_factory=list)
    intervention_days: List[int] = field(default_factory=list)

    # GT -> D resolution metrics
    resolution_times: List[int] = field(default_factory=list)
    resolution_gt_counts: List[int] = field(default_factory=list)
    avg_resolution_time: Optional[float] = None
    avg_cumulative_gt: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "days": self.days,
            "true_infections": self.true_infections,
            "diagnoses": self.diagnoses,
            "sequences": self.sequences,
            "cum_infections": self.cum_infections,
            "beta": self.beta,
            "detection_day": self.detection_day,
            "intervention_day": self.intervention_day,
            "gt_triggers": self.gt_triggers,
            "diag_triggers": self.diag_triggers,
            "intervention_days": self.intervention_days,
            "resolution_times": self.resolution_times,
            "resolution_gt_counts": self.resolution_gt_counts,
            "avg_resolution_time": self.avg_resolution_time,
            "avg_cumulative_gt": self.avg_cumulative_gt,
        }


class Runner:
    """
    Day-by-day orchestrator:
      - steps the Covasim sim
      - applies the ObservationModel
      - checks the GT trigger (on true infections) and the D trigger
        (on observed diagnoses) using the same trigger_list
      - starts the intervention on a D trigger (respecting quiet_period
        and max_interventions)
      - records daily metrics and GT->D resolution timing
    """

    def __init__(self, config: RunConfig):
        self.config = config
        self.sim = self._build_sim(config.sim_pars, config.evo_pars)

        self.obs = ObservationModel(policy=config.policy)
        self.obs.initialize(self.sim)

        self.intervention = InterventionController(policy=config.policy)

        self.results = RunResults(run_id=config.run_id)

        self.gt_triggers: List[int] = []
        self.diag_triggers: List[int] = []
        self.intervention_days: List[int] = []

        # was set but never enforced in the previous version, which let a
        # single run trigger repeated, compounding beta cuts -- exactly
        # the "repeated intervention always wins" problem noted in W5.
        self.max_interventions = config.policy.get("max_interventions", 1)
        self.quiet_period = config.policy.get("quiet_period", 0)
        self.first_diag_trigger_day: Optional[int] = None

    def _build_sim(self, sim_pars, evo_pars) -> cv.Sim:
        pars = sim_pars.copy()
        if evo_pars is not None:
            pars["evo_pars"] = evo_pars
        sim = cv.Sim(pars=pars)
        sim.initialize()
        return sim

    def run(self) -> RunResults:
        n_days = int(self.sim.pars["n_days"])
        trigger_list = self.config.policy.get("trigger_list", [])

        for t in range(n_days):
            self.sim.step()
            self.obs.apply(self.sim, t)

            diagnoses = self.obs.daily_diagnoses
            true_inf_series = self.sim.results["new_infections"]

            if check_triggers(t, true_inf_series, trigger_list, sim=self.sim):
                self.gt_triggers.append(t)

            if check_triggers(t, diagnoses, trigger_list, sim=self.sim):
                self.diag_triggers.append(t)
                if self.first_diag_trigger_day is None:
                    self.first_diag_trigger_day = t

                # NOTE: with every config currently in use, quiet_period is
                # 0, so this is always true (it compares this trigger's
                # own day to itself). If you start using quiet_period > 0,
                # this needs to be redefined as a real cooldown.
                if t >= self.diag_triggers[-1] + self.quiet_period:
                    self._start_intervention(t)

            self.intervention.update(self.sim, t)
            self._record_day(t)

        self._compute_resolution_metrics()

        self.intervention.gt_triggers = self.gt_triggers
        self.intervention.diag_triggers = self.diag_triggers
        self.intervention.intervention_days = self.intervention_days

        self._save_raw()
        return self.results

    def _start_intervention(self, t: int):
        if self.intervention.active:
            return
        if len(self.intervention_days) >= self.max_interventions:
            return
        self.intervention.start(self.sim, t)
        self.intervention_days.append(t)

    def _record_day(self, t: int) -> None:
        r = self.results
        r.days.append(t)
        r.true_infections.append(int(self.sim.results["new_infections"][t]))
        r.cum_infections.append(int(self.sim.results["cum_infections"][t]))
        r.diagnoses.append(int(self.obs.daily_diagnoses[t]))
        r.sequences.append(int(self.obs.daily_sequences[t]))
        r.beta.append(float(self.sim.pars["beta"]))

        if r.detection_day is None and self.first_diag_trigger_day is not None:
            r.detection_day = self.first_diag_trigger_day
        if r.intervention_day is None and self.intervention_days:
            r.intervention_day = self.intervention_days[0]

    def _compute_resolution_metrics(self) -> None:
        gt, diag = self.gt_triggers, self.diag_triggers
        resolution_times: List[int] = []
        resolution_gt_counts: List[int] = []

        i = j = 0
        while i < len(gt) and j < len(diag):
            g_first = gt[i]
            d = diag[j]
            group_count = 0
            while i < len(gt) and gt[i] <= d:
                group_count += 1
                i += 1
            resolution_times.append(d - g_first)
            resolution_gt_counts.append(group_count)
            j += 1

        r = self.results
        r.gt_triggers = gt
        r.diag_triggers = diag
        r.intervention_days = self.intervention_days
        r.resolution_times = resolution_times
        r.resolution_gt_counts = resolution_gt_counts
        if resolution_times:
            r.avg_resolution_time = sum(resolution_times) / len(resolution_times)
        if resolution_gt_counts:
            r.avg_cumulative_gt = sum(resolution_gt_counts) / len(resolution_gt_counts)

    def _save_raw(self) -> None:
        out_dir = self.config.out_dir
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{self.config.run_id}.json")

        def convert(o):
            if isinstance(o, np.integer):
                return int(o)
            if isinstance(o, np.floating):
                return float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            return o

        with open(out_path, "w") as fh:
            json.dump(self.results.to_dict(), fh, indent=2, default=convert)


def run_single(sim_pars: Dict[str, Any], policy: Dict[str, Any], run_id: str) -> RunResults:
    cfg = RunConfig(sim_pars=sim_pars, policy=policy, run_id=run_id)
    return Runner(cfg).run()


def run_evo_simulation(sim_pars: Dict[str, Any],
                        evo_pars: Dict[str, Any],
                        policy: Dict[str, Any],
                        rand_seed: int = 0):
    """
    Convenience wrapper used by the W6-W9 notebooks.
    Returns (sim, obs, intervention, avg_detection_delay, results) --
    same 5-value shape as before, but avg_detection_delay is now just
    results.avg_resolution_time (previously it was recomputed a second
    time, with a different pairing algorithm, by a since-removed
    module-level compute_detection_delay() in this file).
    """
    pars = sim_pars.copy()
    pars["rand_seed"] = rand_seed

    cfg = RunConfig(sim_pars=pars, evo_pars=evo_pars, policy=policy, run_id=f"seed_{rand_seed}")
    runner = Runner(cfg)
    results = runner.run()

    return runner.sim, runner.obs, runner.intervention, results.avg_resolution_time, results