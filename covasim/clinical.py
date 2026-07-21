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

import collections
import dataclasses
import enum
import warnings

import numpy as np

from .analysis import Analyzer
from .sequence_evolution import agent_mutations as _agent_mutations
from .sequence_evolution import decode_mutations as _decode_mutations

__all__ = ['ClinicalPool', 'ClinicalSequencer', 'ClinicalSample']


class ClinicalPool(enum.Enum):
    '''Agent pools eligible for clinical sequencing sampling.'''
    infectious  = 'infectious'
    symptomatic = 'symptomatic'
    diagnosed   = 'diagnosed'


def _coerce_pool(pool):
    '''Accept a ClinicalPool or pool name string.'''
    if isinstance(pool, ClinicalPool):
        return pool
    if isinstance(pool, str):
        try:
            return ClinicalPool(pool)
        except ValueError:
            valid = [p.value for p in ClinicalPool]
            raise ValueError(f"pool must be one of {valid}, got '{pool}'")
    raise TypeError(f"pool must be a ClinicalPool or str, got {type(pool)}")


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

    Specimens collected on a sampling day are not necessarily released the
    same day: they first enter a FIFO pending queue, and are only released
    (moved into ``self.samples``) once the queue holds at least
    ``sequencing_batch_size`` specimens.  This models real-world sequencing
    runs, where samples are batched to fill a flow cell before being
    processed.  With the default ``sequencing_batch_size=1``, every
    specimen is released immediately on the day it is collected (the
    original, unbatched behavior).

    Samples are stored in self.samples (dict: int release_day → list[ClinicalSample]),
    one ``ClinicalSample`` per sequenced agent.  Each ``ClinicalSample.day``
    still reflects the day the specimen was *collected*, which may differ
    from the release day it appears under in ``self.samples`` once batching
    is enabled.  Specimens still awaiting release live in ``self._pending``
    (a FIFO queue); ``self.n_pending`` reports its length. Any specimens
    still pending when the sim ends are never released.
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
        pool      (str or ClinicalPool): Which agents are eligible for sampling.
                                 ``ClinicalPool.diagnosed`` (default) — agents with a
                                 confirmed diagnosis.
                                 ``ClinicalPool.infectious`` — all currently infectious
                                 agents.
                                 ``ClinicalPool.symptomatic`` — infectious agents who
                                 are also currently symptomatic.  Equivalent string
                                 values are also accepted.
        sequencing_batch_size (int): Number of specimens that must accumulate in
                                 the pending queue before a batch is released
                                 (default 1 = release immediately on collection).
                                 Must be a positive integer.
        label     (str):         Optional label for the analyzer.

    Example::

        cs = cv.ClinicalSequencer(days=[56, 120], n_samples=50)
        sim = cv.Sim(pars, analyzers=[cs])
        sim.run()
        cs = sim.get_analyzer('ClinicalSequencer')
        print(cs.to_fasta(56))
    '''

    def __init__(self, days, n_samples=None, pool='diagnosed', sequencing_batch_size=1, label=None):
        super().__init__(label=label)
        if not isinstance(sequencing_batch_size, (int, np.integer)) or sequencing_batch_size < 1:
            raise ValueError(
                f'sequencing_batch_size must be a positive integer, got {sequencing_batch_size}'
            )
        self._days_input          = days
        self._n_samples            = n_samples
        self.pool                  = _coerce_pool(pool)
        self.sequencing_batch_size = int(sequencing_batch_size)
        self.samples               = {}
        self._pending              = collections.deque()  # FIFO queue of collected-but-unreleased ClinicalSamples
        self._reference            = None  # set in initialize()
        self._rng                  = None  # set in initialize()

    @property
    def n_pending(self):
        '''Number of collected specimens awaiting release in the batch queue.'''
        return len(self._pending)

    def initialize(self, sim):
        super().initialize(sim)
        if not sim['evo_pars'].get('enable', False):
            raise ValueError(
                "ClinicalSequencer requires pars['evo_pars']['enable'] = True"
            )
        self._reference = sim.people.sequence_tracker.reference
        self._rng       = np.random.default_rng(sim['rand_seed'])

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
        if self.pool == ClinicalPool.infectious:
            pool_inds = np.nonzero(people.infectious)[0]
        elif self.pool == ClinicalPool.symptomatic:
            pool_inds = np.nonzero(people.symptomatic)[0]
        else:  # ClinicalPool.diagnosed
            pool_inds = np.nonzero(people.diagnosed)[0]

        if len(pool_inds) == 0:
            return

        if len(pool_inds) < n:
            warnings.warn(
                f"ClinicalSequencer day {sim.t}: requested {n} samples but only "
                f"{len(pool_inds)} agents are available in pool='{self.pool.value}'. "
                f"Returning all {len(pool_inds)} agents."
            )
            sampled_inds = pool_inds
        else:
            sampled_inds = self._rng.choice(pool_inds, size=n, replace=False)

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

        self._pending.extend(day_samples)
        self._release_batches(sim.t)

    def _release_batches(self, release_day):
        '''Move complete batches from the pending FIFO queue into self.samples.'''
        while len(self._pending) >= self.sequencing_batch_size:
            batch = [self._pending.popleft() for _ in range(self.sequencing_batch_size)]
            self.samples.setdefault(release_day, []).extend(batch)

    def to_fasta(self, day):
        '''Return a multi-FASTA string, one record per sampled agent, for the given day.'''
        day_samples = self.samples.get(day, [])
        if not day_samples:
            return ''
        ref   = self._reference
        lines = []
        for s in day_samples:
            lines.append(
                f'>agent_{s.agent_id}  day={s.day}  date={s.date}  variant={s.variant_label}'
            )
            lines.append(_decode_mutations(s.mutations, ref))
        return '\n'.join(lines)
