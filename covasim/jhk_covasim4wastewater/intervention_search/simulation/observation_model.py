"""
Observation model: turns true daily infections into observed diagnoses
and observed sequences via a simple aggregate testing/delay/sequencing
process (binomial testing -> multinomial delay -> binomial sequencing).
"""

import numpy as np


class ObservationModel:
    """
    Samples, for each day:
      - how many true infections get diagnosed (Binomial(n, p_test))
      - which future day each diagnosis lands on (Multinomial(delay_pmf))
      - how many of those diagnoses get sequenced (Binomial(count, p_seq))
    and records which specific infection events were sequenced.
    """

    def __init__(self, policy):
        self.p_test = policy.get("p_test", 0.1)
        self.p_seq = policy.get("p_seq", 0.1)
        self.delay_pmf = np.array(policy.get("delay_pmf", [1.0]))

        self.sim_npts = None
        self.daily_diagnoses = None
        self.daily_sequences = None
        self.daily_sequenced_agents = None

    def initialize(self, sim):
        T = sim.npts
        self.sim_npts = T
        self.daily_diagnoses = np.zeros(T, dtype=int)
        self.daily_sequences = np.zeros(T, dtype=int)
        self.daily_sequenced_agents = [[] for _ in range(T)]

    def apply(self, sim, t):
        """Called once per day by the runner."""
        true_inf = int(sim.results["new_infections"][t])

        diagnoses_raw = np.random.binomial(true_inf, self.p_test)
        if diagnoses_raw == 0:
            return

        diag_by_delay = np.random.multinomial(diagnoses_raw, self.delay_pmf)

        for k, count in enumerate(diag_by_delay):
            if count == 0:
                continue

            diag_day = t + k
            if diag_day >= self.sim_npts:
                break

            self.daily_diagnoses[diag_day] += count

            seq_count = np.random.binomial(count, self.p_seq)
            self.daily_sequences[diag_day] += seq_count

            if seq_count > 0:
                infection_log = sim.people.infection_log
                newly_infected_today = [
                    entry["target"] for entry in infection_log if entry["date"] == t
                ]
                if newly_infected_today:
                    chosen = np.random.choice(
                        newly_infected_today,
                        size=min(seq_count, len(newly_infected_today)),
                        replace=False,
                    )
                    event_ids = [f"{agent}_{t}" for agent in chosen]
                    self.daily_sequenced_agents[diag_day].extend(event_ids)