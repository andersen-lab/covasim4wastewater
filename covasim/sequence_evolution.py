'''
Sequence evolution module for Covasim.

When enabled via pars['evo_pars']['enable'] = True, annotates each
infection_log entry with nucleotide mutation deltas accumulated along the
transmission edge using a Jukes-Cantor (or pluggable) substitution model.

Sequences are encoded as numpy uint8 arrays: A=0, C=1, G=2, T=3.
Only branch-level mutation deltas are stored on log entries; full haplotypes
are reconstructed by walking the transmission tree and applying episode roots
plus deltas (see reconstruct_haplotype()).
'''

import os
import numpy as np
from abc import ABC, abstractmethod

__all__ = ['SubstitutionModel', 'JukesCantor', 'LineageSequenceTracker',
           'encode_sequence', 'decode_sequence', 'extract_founding_mutations']

_NT_TO_INT = {'A': 0, 'C': 1, 'G': 2, 'T': 3,
              'a': 0, 'c': 1, 'g': 2, 't': 3}
_INT_TO_NT = ['A', 'C', 'G', 'T']


def encode_sequence(seq_str):
    '''Convert a nucleotide string (ACGT) to a uint8 array.'''
    return np.array([_NT_TO_INT[c] for c in seq_str], dtype=np.uint8)


def decode_sequence(seq_arr):
    '''Convert a uint8 array back to a nucleotide string.'''
    return ''.join(_INT_TO_NT[int(b)] for b in seq_arr)


def _align_to_reference(seq_str, ref_str):
    '''
    Global pairwise alignment of seq_str to ref_str using biopython.

    Returns (aligned_ref, aligned_query) strings with '-' for gaps.
    Only called when lengths differ.
    '''
    try:
        from Bio.Align import PairwiseAligner
    except ImportError:
        raise ImportError(
            'biopython is required to align sequences of different lengths. '
            'Install with: pip install biopython'
        )
    aligner = PairwiseAligner()
    aligner.mode = 'global'
    aligner.open_gap_score   = -10
    aligner.extend_gap_score = -0.5
    aligner.match_score      = 2
    aligner.mismatch_score   = -1
    alignment = next(aligner.align(ref_str, seq_str))
    ref_coords, query_coords = alignment.aligned
    aligned_ref, aligned_query = [], []
    r_prev, q_prev = 0, 0
    for (r_start, r_end), (q_start, q_end) in zip(ref_coords, query_coords):
        if r_start > r_prev:       # deletion in query
            aligned_ref.append(ref_str[r_prev:r_start])
            aligned_query.append('-' * (r_start - r_prev))
        if q_start > q_prev:       # insertion in query
            aligned_query.append(seq_str[q_prev:q_start])
            aligned_ref.append('-' * (q_start - q_prev))
        aligned_ref.append(ref_str[r_start:r_end])
        aligned_query.append(seq_str[q_start:q_end])
        r_prev, q_prev = r_end, q_end
    if r_prev < len(ref_str):
        aligned_ref.append(ref_str[r_prev:])
        aligned_query.append('-' * (len(ref_str) - r_prev))
    if q_prev < len(seq_str):
        aligned_query.append(seq_str[q_prev:])
        aligned_ref.append('-' * (len(seq_str) - q_prev))
    return ''.join(aligned_ref), ''.join(aligned_query)


def extract_founding_mutations(fasta_path, reference):
    '''
    Diff a single-record FASTA against the reference and return a frozenset of SNPs.

    When the FASTA length matches the reference, a fast character-by-character diff
    is used. When lengths differ (common for consensus genomes with indels), the
    sequence is globally aligned to the reference with biopython; indel columns are
    skipped since the downstream fitness model only handles SNPs.

    Args:
        fasta_path (str): path to FASTA file (single record)
        reference  (ndarray): uint8 reference sequence from LineageSequenceTracker

    Returns:
        frozenset of (site_0indexed, ref_nt_int, alt_nt_int)
    '''
    import warnings
    seq_lines = []
    with open(fasta_path) as fh:
        for line in fh:
            if not line.startswith('>'):
                seq_lines.append(line.strip())
    seq_str = ''.join(seq_lines)

    if len(seq_str) != len(reference):
        ref_str = ''.join(_INT_TO_NT[int(b)] for b in reference)
        aligned_ref, aligned_query = _align_to_reference(seq_str, ref_str)
        mutations = set()
        ref_site = 0
        for r_char, q_char in zip(aligned_ref, aligned_query):
            if r_char == '-':           # insertion in query → no reference coordinate
                continue
            if q_char == '-':           # deletion in query → not an SNP
                ref_site += 1
                continue
            if q_char not in _NT_TO_INT:    # ambiguous base → treat as reference
                ref_site += 1
                continue
            ref_nt = _NT_TO_INT[r_char]
            alt_nt = _NT_TO_INT[q_char]
            if alt_nt != ref_nt:
                mutations.add((ref_site, ref_nt, alt_nt))
            ref_site += 1
        return frozenset(mutations)

    # Fast path: lengths match — character-by-character diff
    # Replace N/n (ambiguous bases) with the reference nucleotide so they don't
    # register as mutations — consensus FASTAs commonly contain Ns at low-coverage sites.
    seq_str = ''.join(
        _INT_TO_NT[int(reference[i])] if c in ('N', 'n') else c
        for i, c in enumerate(seq_str)
    )
    seq_arr = encode_sequence(seq_str)
    return frozenset(
        (i, int(reference[i]), int(seq_arr[i]))
        for i in range(len(reference))
        if seq_arr[i] != reference[i]
    )


def _apply_branch_mutations(parent_muts, branch_mutations, reference):
    '''
    Compute a child's mutation set by applying branch mutations to a parent's set.

    parent_muts and the returned frozenset are relative to the reference:
    each element is (site_0indexed, ref_nt_int, child_nt_int).
    Reversions back to the reference allele are removed.

    Args:
        parent_muts      (frozenset): parent's mutations vs reference
        branch_mutations (list):      [(site, from_parent_nt, to_child_nt), ...]
        reference        (ndarray):   uint8 reference sequence

    Returns:
        frozenset of (site_0indexed, ref_nt_int, child_nt_int)
    '''
    current = {site: to_nt for site, _, to_nt in parent_muts}
    for site, _from, to_child in branch_mutations:
        ref_nt = int(reference[site])
        if to_child == ref_nt:
            current.pop(site, None)   # reversion to reference
        else:
            current[site] = to_child  # new or updated mutation
    return frozenset((site, int(reference[site]), to_nt) for site, to_nt in current.items())


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

    Attach to a sim via pars['evo_pars']['enable'] = True; Sim.initialize()
    creates the tracker and attaches it to sim.people.sequence_tracker so that
    People.infect() can call annotate_entry() after each log append.

    Fitness support:
        A VariantFitnessModel is always instantiated when evo_pars is enabled
        (done by sim.init_sequence_tracker()).
        Set variant_founding_mutations[label] = frozenset(...) for any variant
        whose founding genotype differs from the reference.
    '''

    def __init__(self, evo_pars, seed=None):
        self.L    = evo_pars.get('L', 1000)
        self.rate = evo_pars.get('mol_clock_rate', 1e-5)

        ref = evo_pars.get('reference')
        if ref is None:
            ref = 'A' * self.L
        elif isinstance(ref, str):
            _looks_like_path = (
                ref.endswith(('.fasta', '.fa', '.fas', '.fna'))
                or os.sep in ref
                or '/' in ref
            )
            if _looks_like_path and not os.path.isfile(ref):
                raise FileNotFoundError(
                    f"evo_pars['reference'] path not found: '{ref}'. "
                    f"Current working directory is '{os.getcwd()}'. "
                    "Use an absolute path or ensure the file exists relative to the CWD."
                )
            if os.path.isfile(ref):
                # FASTA file path: read sequence and override L
                seq_lines = []
                with open(ref) as fh:
                    for line in fh:
                        if not line.startswith('>'):
                            seq_lines.append(line.strip())
                ref = ''.join(seq_lines)
                self.L = len(ref)
        if isinstance(ref, str):
            if len(ref) != self.L:
                raise ValueError(f"evo_pars['reference'] has length {len(ref)} but L={self.L}")
            ref = encode_sequence(ref)
        self.reference = ref.astype(np.uint8)

        model_key = evo_pars.get('sub_model', 'JC')
        if model_key not in _MODEL_REGISTRY:
            raise ValueError(f"Unknown substitution model '{model_key}'; choices: {list(_MODEL_REGISTRY)}")
        self.model = _MODEL_REGISTRY[model_key]()

        self.rng = np.random.default_rng(seed)
        self._episode_roots = {}  # int agent_idx → uint8 haplotype at last acquisition

        # Fitness-related state (populated by sim.init_sequence_tracker() if enabled)
        self.fitness_model = None                # VariantFitnessModel instance or None
        self.variant_founding_mutations = {}     # label (str) → frozenset{(site, ref_nt, alt_nt)}
        self.agent_mutations = {}                # agent_idx (int) → frozenset{(site, ref_nt, alt_nt)}

    def annotate_entry(self, entry, people):
        '''
        Annotate one infection_log entry with sequence evolution data.

        Mutates entry in place, adding:
          branch_length      (float): Δt in days
          branch_mutations   (list):  [(site, from_nt_int, to_nt_int), ...]
          n_mutations        (int):   len(branch_mutations)

        Also maintains agent_mutations (relative-to-reference mutation sets) when
        a fitness model or founding mutations are in use.

        Call this after infection_log.append(entry) and before
        date_exposed[target] is written for the current infection.
        '''
        source        = entry['source']
        target        = int(entry['target'])
        date          = float(entry['date'])
        variant_label = entry.get('variant', 'wild')

        _track = self.fitness_model is not None or bool(self.variant_founding_mutations)

        if source is None:
            # Seed or import: branch length 0. The episode root is the variant's
            # founding haplotype (reference + its defining SNPs), not the bare
            # reference — otherwise reconstruct_haplotype() would strip variant
            # identity and every variant would collapse to the reference sequence.
            delta_t     = 0.0
            founding    = self.variant_founding_mutations.get(variant_label, frozenset())
            if founding:
                parent_seq = self.reference.copy()
                for site, _ref_nt, alt_nt in founding:
                    parent_seq[site] = alt_nt
            else:
                parent_seq = self.reference
            parent_muts = founding if _track else None
        else:
            source     = int(source)
            parent_seq = self._episode_roots.get(source, self.reference)
            src_exposed = float(people.date_exposed[source])
            if np.isnan(src_exposed):
                delta_t = 0.0
            else:
                delta_t = max(0.0, date - src_exposed)
            parent_muts = self.agent_mutations.get(source, frozenset()) if _track else None

        mutations, child_seq = self.model.evolve_branch(self.rng, parent_seq, delta_t, self.rate)

        self._episode_roots[target] = child_seq

        if _track:
            self.agent_mutations[target] = _apply_branch_mutations(parent_muts, mutations, self.reference)

        entry['branch_length']    = delta_t
        entry['branch_mutations'] = mutations
        entry['n_mutations']      = len(mutations)

    def initialize_founder_haplotype(self, agent_idx, variant_label):
        '''
        Set agent_mutations for an import/seed agent to their variant's founding set.
        Called implicitly via annotate_entry; exposed as a public method for testing.
        '''
        self.agent_mutations[int(agent_idx)] = self.variant_founding_mutations.get(variant_label, frozenset())

    def get_fitness_multiplier(self, agent_idx):
        '''Return rel_beta multiplier for agent_idx; 1.0 if no fitness model is set.'''
        if self.fitness_model is None:
            return 1.0
        muts = self.agent_mutations.get(int(agent_idx), frozenset())
        return self.fitness_model.compute_fitness(muts)

    def reconstruct_haplotype(self, agent_idx):
        '''
        Return the haplotype string at last acquisition for agent_idx.
        Returns the reference sequence string if the agent has never been infected.
        '''
        return decode_sequence(self._episode_roots.get(int(agent_idx), self.reference))
