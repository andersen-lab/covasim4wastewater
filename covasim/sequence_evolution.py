'''
Sequence evolution module for Covasim.

When enabled via pars['seq_pars']['enable'] = True, annotates each
infection_log entry with nucleotide mutation deltas accumulated along the
transmission edge using a Jukes-Cantor (or pluggable) substitution model.

Sequences are encoded as numpy uint8 arrays: A=0, C=1, G=2, T=3.
Only branch-level mutation deltas are stored on log entries; full haplotypes
are reconstructed by walking the transmission tree and applying episode roots
plus deltas (see reconstruct_haplotype()).
'''

import numpy as np
from abc import ABC, abstractmethod

__all__ = ['SubstitutionModel', 'JukesCantor', 'LineageSequenceTracker',
           'encode_sequence', 'decode_sequence']

_NT_TO_INT = {'A': 0, 'C': 1, 'G': 2, 'T': 3,
              'a': 0, 'c': 1, 'g': 2, 't': 3}
_INT_TO_NT = ['A', 'C', 'G', 'T']


def encode_sequence(seq_str):
    '''Convert a nucleotide string (ACGT) to a uint8 array.'''
    return np.array([_NT_TO_INT[c] for c in seq_str], dtype=np.uint8)


def decode_sequence(seq_arr):
    '''Convert a uint8 array back to a nucleotide string.'''
    return ''.join(_INT_TO_NT[int(b)] for b in seq_arr)


class SubstitutionModel(ABC):
    '''Abstract base for nucleotide substitution models.'''

    @property
    @abstractmethod
    def name(self):  # pragma: no cover
        ...

    @abstractmethod
    def evolve_branch(self, rng, parent_seq, delta_t, rate_per_site):
        '''
        Apply substitutions along a branch of length delta_t.

        Args:
            rng            (Generator): numpy random generator
            parent_seq     (ndarray):   uint8 array of length L (parent haplotype)
            delta_t        (float):     branch length in days
            rate_per_site  (float):     expected substitutions per site per day (μ)

        Returns:
            mutations (list of tuple): (site, from_nt_int, to_nt_int) for each event
            child_seq (ndarray):       uint8 array, haplotype after mutations
        '''
        ...  # pragma: no cover


class JukesCantor(SubstitutionModel):
    '''
    Jukes-Cantor (JC69) substitution model.

    Total mutations over L sites and branch length Δt are drawn from
    Poisson(L * μ * Δt), then allocated to uniformly random sites.
    At each site a uniform random one of the three alternative nucleotides
    replaces the current base.
    '''

    name = 'JC'

    def evolve_branch(self, rng, parent_seq, delta_t, rate_per_site):
        L = len(parent_seq)
        n_events = int(rng.poisson(L * rate_per_site * delta_t)) if delta_t > 0 else 0
        if n_events == 0:
            return [], parent_seq.copy()

        child_seq = parent_seq.copy()
        sites   = rng.integers(0, L, size=n_events)
        offsets = rng.integers(1, 4, size=n_events)  # 1, 2, or 3 → guarantees different nucleotide

        mutations = []
        for site, offset in zip(sites, offsets):
            from_nt = int(child_seq[site])
            to_nt   = int((from_nt + int(offset)) % 4)
            child_seq[site] = to_nt
            mutations.append((int(site), from_nt, to_nt))

        return mutations, child_seq


_MODEL_REGISTRY = {'JC': JukesCantor}


class LineageSequenceTracker:
    '''
    Tracks viral sequence evolution along the transmission tree.

    Maintains a per-agent episode haplotype (reset on each new infection) and
    annotates infection_log entries with branch-specific mutation deltas.

    Attach to a sim via pars['seq_pars']['enable'] = True; Sim.initialize()
    creates the tracker and attaches it to sim.people.sequence_tracker so that
    People.infect() can call annotate_entry() after each log append.
    '''

    def __init__(self, seq_pars, seed=None):
        self.L           = seq_pars.get('L', 1000)
        self.rate        = seq_pars.get('rate_per_site_per_day', 1e-5)

        wt = seq_pars.get('wild_type')
        if wt is None:
            wt = 'A' * self.L
        if isinstance(wt, str):
            if len(wt) != self.L:
                raise ValueError(f"seq_pars['wild_type'] has length {len(wt)} but L={self.L}")
            wt = encode_sequence(wt)
        self.wild_type = wt.astype(np.uint8)

        model_key = seq_pars.get('model', 'JC')
        if model_key not in _MODEL_REGISTRY:
            raise ValueError(f"Unknown substitution model '{model_key}'; choices: {list(_MODEL_REGISTRY)}")
        self.model = _MODEL_REGISTRY[model_key]()

        self.rng = np.random.default_rng(seed)
        self._episode_roots = {}  # int agent_idx → uint8 haplotype at last acquisition

    def annotate_entry(self, entry, people):
        '''
        Annotate one infection_log entry with sequence evolution data.

        Mutates entry in place, adding:
          branch_length      (float): Δt in days
          branch_mutations   (list):  [(site, from_nt_int, to_nt_int), ...]
          n_mutations        (int):   len(branch_mutations)

        Call this after infection_log.append(entry) and before
        date_exposed[target] is written for the current infection.
        '''
        source = entry['source']
        target = int(entry['target'])
        date   = float(entry['date'])

        if source is None:
            # Seed or import: root is wild_type, branch length 0
            parent_seq = self.wild_type
            delta_t    = 0.0
        else:
            source     = int(source)
            parent_seq = self._episode_roots.get(source, self.wild_type)
            src_exposed = float(people.date_exposed[source])
            if np.isnan(src_exposed):
                delta_t = 0.0
            else:
                delta_t = max(0.0, date - src_exposed)

        mutations, child_seq = self.model.evolve_branch(self.rng, parent_seq, delta_t, self.rate)

        self._episode_roots[target] = child_seq

        entry['branch_length']    = delta_t
        entry['branch_mutations'] = mutations
        entry['n_mutations']      = len(mutations)

    def reconstruct_haplotype(self, agent_idx):
        '''
        Return the haplotype string at last acquisition for agent_idx.
        Returns wild_type string if the agent has never been infected.
        '''
        return decode_sequence(self._episode_roots.get(int(agent_idx), self.wild_type))
