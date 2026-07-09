'''
Clinical sequencing analyzer for Covasim.

ClinicalSequencer is a cv.Analyzer that, on specified days, randomly draws a
fixed number of agents from the current pool of infectious (or symptomatic /
diagnosed) individuals and records each agent's evolved haplotype as a set of
mutations relative to the simulation reference sequence.

This mirrors the real-world practice of routine clinical surveillance sequencing,
where a sample of positive tests is sent for whole-genome sequencing each day.

Requires pars['evo_pars']['enable'] = True.
'''

import dataclasses
import warnings

import numpy as np

from .analysis import Analyzer
from .sequence_evolution import decode_sequence

__all__ = ['ClinicalSequencer', 'ClinicalSample']


def _agent_mutations(tracker, agent_idx, reference):
    '''
    Return the mutation frozenset for agent_idx.

    Prefers tracker.agent_mutations (populated when fitness tracking is active).
    Falls back to diffing _episode_roots against the reference when the key is
    absent — O(L) but only called on sampling days.
    '''
    if agent_idx in tracker.agent_mutations:
        return tracker.agent_mutations[agent_idx]
    root = tracker._episode_roots.get(agent_idx, reference)
    return frozenset(
        (j, int(reference[j]), int(root[j]))
        for j in range(len(reference))
        if root[j] != reference[j]
    )


def _decode_mutations(mutations, reference):
    '''Reconstruct an ACGT string from a mutation frozenset and reference uint8 array.'''
    seq = reference.copy()
    for site, _ref_nt, alt_nt in mutations:
        seq[site] = alt_nt
    return decode_sequence(seq)


@dataclasses.dataclass
class ClinicalSample:
    '''Per-agent clinical sequencing snapshot for a single day.'''
    day:           int
    date:          str
    agent_id:      int        # agent index
    mutations:     frozenset  # frozenset of (site_0idx, ref_nt_int, alt_nt_int) vs reference
    variant_label: str        # variant label for this agent


class ClinicalSequencer(Analyzer):
    '''
    Analyzer that randomly draws a specified number of infectious agents on
    given days and records each agent's full evolved haplotype.

    For each sampled day:
      1. Identifies agents in the sampling pool (infectious, symptomatic, or
         diagnosed — controlled by the ``pool`` parameter).
      2. Draws ``n_samples`` agents uniformly at random without replacement.
         If the pool is smaller than ``n_samples``, all pool members are
         returned and a warning is issued.
      3. Records each agent's haplotype as a frozenset of
         ``(site_0indexed, ref_nt_int, alt_nt_int)`` mutation tuples relative
         to the simulation reference.  Call ``to_fasta(day)`` to reconstruct
         full ACGT strings on demand.
      4. Stores one ``ClinicalSample`` per agent in ``self.samples[day]``.

    Samples are stored in self.samples (dict: int day → list[ClinicalSample]),
    one ``ClinicalSample`` per sequenced agent.
    Use to_fasta(day) to get a multi-FASTA string suitable for downstream
    analysis or comparison with wastewater reads.

    Requires pars['evo_pars']['enable'] = True.

    Args:
        days     (list or dict): Simulation days (int) or calendar date strings
                                 at which to sample.  Pass a plain list together
                                 with a scalar ``n_samples`` for uniform sampling,
                                 or a dict mapping day → n_samples for variable
                                 per-day counts.
        n_samples (int):         Number of agents to sample per day when ``days``
                                 is a list.  Ignored when ``days`` is a dict.
        pool      (str):         Which agents are eligible for sampling.
                                 ``'symptomatic'`` (default) — infectious agents
                                 who are also currently symptomatic.
                                 ``'infectious'`` — all currently infectious agents.
                                 ``'diagnosed'`` — agents with a confirmed diagnosis.
        label     (str):         Optional label for the analyzer.

    Example::

        cs = cv.ClinicalSequencer(days=[56, 120], n_samples=50)
        sim = cv.Sim(pars, analyzers=[cs])
        sim.run()
        cs = sim.get_analyzer('ClinicalSequencer')
        print(cs.to_fasta(56))
    '''

    def __init__(self, days, n_samples=None, pool='symptomatic', label=None):
        super().__init__(label=label)
        self._days_input = days
        self._n_samples  = n_samples
        self.pool        = pool
        self.samples     = {}
        self._reference  = None  # set in initialize()

    def initialize(self, sim):
        super().initialize(sim)
        if not sim['evo_pars'].get('enable', False):
            raise ValueError(
                "ClinicalSequencer requires pars['evo_pars']['enable'] = True"
            )
        valid_pools = ('infectious', 'symptomatic', 'diagnosed')
        if self.pool not in valid_pools:
            raise ValueError(
                f"pool must be one of {valid_pools}, got '{self.pool}'"
            )
        self._reference = sim.people.sequence_tracker.reference

        # Build {day_int: n_samples} lookup
        if isinstance(self._days_input, dict):
            self._day_to_n = {}
            for d, n in self._days_input.items():
                day_int = sim.day(d) if isinstance(d, str) else int(d)
                self._day_to_n[day_int] = int(n)
        else:
            if self._n_samples is None:
                raise ValueError(
                    "n_samples must be provided when days is a list"
                )
            self._day_to_n = {}
            for d in self._days_input:
                day_int = sim.day(d) if isinstance(d, str) else int(d)
                self._day_to_n[day_int] = int(self._n_samples)

    def apply(self, sim):
        if sim.t in self._day_to_n:
            self._take_sample(sim, self._day_to_n[sim.t])

    def _take_sample(self, sim, n):
        people  = sim.people
        tracker = people.sequence_tracker
        ref     = self._reference

        # Build the eligible pool
        if self.pool == 'infectious':
            pool_inds = np.nonzero(people.infectious)[0]
        elif self.pool == 'symptomatic':
            pool_inds = np.nonzero(people.symptomatic)[0]
        else:  # 'diagnosed'
            pool_inds = np.nonzero(people.diagnosed)[0]

        if len(pool_inds) == 0:
            self.samples[sim.t] = []
            return

        if len(pool_inds) < n:
            warnings.warn(
                f"ClinicalSequencer day {sim.t}: requested {n} samples but only "
                f"{len(pool_inds)} agents are available in pool='{self.pool}'. "
                f"Returning all {len(pool_inds)} agents."
            )
            sampled_inds = pool_inds
        else:
            sampled_inds = np.random.choice(pool_inds, size=n, replace=False)

        variant_map = sim.pars.get('variant_map', {})
        date        = sim.date(sim.t)

        day_samples = []
        for idx in sampled_inds:
            i = int(idx)
            v = people.infectious_variant[i]
            label = variant_map.get(int(v), 'wild') if not np.isnan(v) else 'wild'
            day_samples.append(ClinicalSample(
                day           = sim.t,
                date          = date,
                agent_id      = i,
                mutations     = _agent_mutations(tracker, i, ref),
                variant_label = label,
            ))

        self.samples[sim.t] = day_samples

    def to_fasta(self, day):
        '''Return a multi-FASTA string, one record per sampled agent, for the given day.'''
        day_samples = self.samples.get(day, [])
        if not day_samples:
            return ''
        ref   = self._reference
        lines = []
        for s in day_samples:
            lines.append(
                f'>agent_{s.agent_id}  day={day}  date={s.date}  variant={s.variant_label}'
            )
            lines.append(_decode_mutations(s.mutations, ref))
        return '\n'.join(lines)
