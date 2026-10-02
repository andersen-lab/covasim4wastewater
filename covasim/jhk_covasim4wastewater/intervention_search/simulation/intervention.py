"""
Reduces beta once triggered and keeps it reduced for a fixed duration.
Beta is not restored afterwards -- Covasim's own dynamics continue from
the reduced value.
"""


class InterventionController:
    def __init__(self, policy):
        self.beta_reduction = policy.get("beta_reduction", 0.3)
        self.beta_min = policy.get("beta_min", 0.005)
        # NOTE: configs/intervention_policies.yaml uses the key "duration"
        # (not "intervention_duration", which this used to read -- that
        # mismatch meant every policy silently fell back to the default
        # of 14 days regardless of what the YAML said).
        self.duration = policy.get("duration", 14)

        self.active = False
        self.intervention_day = None
        self.intervention_days = []

        # populated by Runner after the run finishes, for plotting
        self.gt_triggers = []
        self.diag_triggers = []

    def start(self, sim, t):
        if self.active:
            return
        self.intervention_day = t
        self.intervention_days.append(t)
        self._activate(sim)
        self.active = True

    def update(self, sim, t):
        if self.active and t >= self.intervention_day + self.duration:
            self.active = False

    def _activate(self, sim):
        sim.pars["beta"] = max(sim.pars["beta"] * (1 - self.beta_reduction), self.beta_min)