'''
Wastewater sampling analyzer for Covasim.

WastewaterSampler is a cv.Analyzer that, at specified time points, snapshots
the set of circulating viral haplotypes and their relative proportions weighted
by individual viral shedding — i.e. what a wastewater sample would look like.
Haplotypes are stored as frozensets of (site_0indexed, ref_nt_int, alt_nt_int)
mutation tuples relative to the simulation reference sequence.
Snapshots can be exported as FASTA or fed directly to Bygul's
simulate_proportions to generate synthetic sequencing reads.

Requires pars['evo_pars']['enable'] = True.
'''

import dataclasses
from collections import defaultdict

import numpy as np

from .analysis import Analyzer
from .sequence_evolution import decode_sequence

__all__ = ['WastewaterSampler', 'WastewaterSample']


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
class WastewaterSample:
    '''Snapshot of circulating haplotypes and their viral-shedding-weighted proportions.'''
    day:            int
    date:           str
    mutation_sets:  list  # deduplicated frozensets of (site_0idx, ref_nt_int, alt_nt_int), one per distinct haplotype
    variant_labels: list  # variant label string for each distinct haplotype
    proportions:    list  # normalized viral-shedding fractions, sums to 1.0
    raw_loads:      list  # un-normalized total viral shedding per haplotype
    n_infectious:   int   # number of currently infectious agents
    n_genotypes:    int   # number of distinct haplotypes present


class WastewaterSampler(Analyzer):
    '''
    Analyzer that snapshots the viral haplotype mixture present in a wastewater
    sample at one or more time points.

    For each sampled day:
      1. Identifies all currently infectious agents.
      2. Reads each agent's viral shedding from ``sim.people.viral_shedding``
         (``viral_load × rel_trans``), which is updated by ``sim.step()`` before
         analyzers are called.
      3. Retrieves each agent's mutation frozenset from LineageSequenceTracker.
      4. Groups identical haplotypes (by frozenset identity) and sums their
         viral shedding contributions.
      5. Normalizes to proportions.

    Snapshots are stored in self.samples (dict: int day → WastewaterSample).
    Use to_fasta(day) to get a multi-FASTA string, or simulate_sample(day, ...)
    to pass the mixture directly to Bygul.

    Requires pars['evo_pars']['enable'] = True.

    Args:
        days  (list): simulation days (int) or calendar date strings to sample.
        label (str):  optional label for the analyzer.

    Example::

        ww = cv.WastewaterSampler(days=[56, 120])
        sim = cv.Sim(pars, analyzers=[ww])
        sim.run()
        print(ww.to_fasta(56))
    '''

    def __init__(self, days, label=None):
        super().__init__(label=label)
        self._days_input = days  # raw user input; converted in initialize()
        self.samples     = {}
        self._reference  = None  # set in initialize()

    def initialize(self, sim):
        super().initialize(sim)
        if not sim['evo_pars'].get('enable', False):
            raise ValueError(
                "WastewaterSampler requires pars['evo_pars']['enable'] = True"
            )
        self._reference = sim.people.sequence_tracker.reference
        # Convert string dates to integer days
        self._day_set = set()
        for d in self._days_input:
            self._day_set.add(sim.day(d) if isinstance(d, str) else int(d))

    def apply(self, sim):
        if sim.t not in self._day_set:
            return
        self._take_sample(sim)

    def _take_sample(self, sim):
        # Infectious agents only
        inds = np.nonzero(sim.people.infectious)[0]

        if len(inds) == 0:
            self.samples[sim.t] = None
            return

        loads       = sim.people.viral_shedding[inds]
        tracker     = sim.people.sequence_tracker
        ref         = self._reference
        variant_map = sim.pars.get('variant_map', {})

        load_by_muts  = defaultdict(float)
        label_by_muts = {}   # first-seen variant label per distinct haplotype
        for idx, load in zip(inds, loads):
            i    = int(idx)
            muts = _agent_mutations(tracker, i, ref)
            load_by_muts[muts] += float(load)
            if muts not in label_by_muts:
                v = sim.people.infectious_variant[i]
                label_by_muts[muts] = (
                    variant_map.get(int(v), 'wild') if not np.isnan(v) else 'wild'
                )

        total         = sum(load_by_muts.values())
        mutation_sets = list(load_by_muts.keys())
        raw_loads     = [load_by_muts[m] for m in mutation_sets]
        proportions   = [v / total for v in raw_loads]
        proportions[-1] = 1.0 - sum(proportions[:-1])

        self.samples[sim.t] = WastewaterSample(
            day            = sim.t,
            date           = sim.date(sim.t),
            mutation_sets  = mutation_sets,
            variant_labels = [label_by_muts[m] for m in mutation_sets],
            proportions    = proportions,
            raw_loads      = raw_loads,
            n_infectious   = int(len(inds)),
            n_genotypes    = len(mutation_sets),
        )

    def to_fasta(self, day):
        '''Return a multi-FASTA string for the haplotype mixture on the given day.'''
        sample = self.samples[day]
        if sample is None:
            return ''
        ref   = self._reference
        lines = []
        for i, (muts, prop, vlabel) in enumerate(zip(sample.mutation_sets, sample.proportions, sample.variant_labels)):
            lines.append(f'>genotype_{i}  proportion={prop:.8f}  day={day}  date={sample.date}  variant={vlabel}')
            lines.append(_decode_mutations(muts, ref))
        return '\n'.join(lines)

    def simulate_sample(self, day, primers, reference, outdir='./reads/', readcnt=500, redo=False, **bygul_kwargs):
        '''
        Write per-haplotype FASTA files and invoke Bygul's simulate_proportions CLI.

        Bygul's simulate_proportions command is a Click CLI that expects genome
        sequences as on-disk FASTA files. This method writes temporary FASTA files,
        builds the required CLI argument list, and calls the command via Click's
        standalone_mode=False interface.

        Args:
            day       (int):  simulation day to sample (must be in self.samples)
            primers   (str):  path to primer BED file (required by Bygul)
            reference (str):  path to reference FASTA file (required by Bygul)
            outdir    (str):  output directory for Bygul reads (default '.') (NOTE: If redo=True, Bygul will delete all contents of outdir)
            readcnt   (int):  number of reads per amplicon (default 500)
            redo      (bool): re-run even if output files already exist (default False)
            **bygul_kwargs:   additional CLI options forwarded as --key value pairs
                              (e.g. wgsim_read_length=150, wgsim_error_rate=0.0001,
                               simulation_mode='amplicon', seed=42)

        Requires the `bygul` package to be installed.
        '''
        import os
        import tempfile

        try:
            from bygul._cli import simulate_proportions
        except ImportError as e:
            raise ImportError(
                "The 'bygul' package is required for WastewaterSampler.simulate(). "
                "Install it from the Andersen Lab: https://github.com/andersen-lab/Bygul"
            ) from e

        sample = self.samples[day]
        if sample is None:
            raise ValueError(f"No infectious agents were present on day {day}; cannot simulate.")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Write each unique haplotype to its own FASTA file
            ref         = self._reference
            fasta_paths = []
            for i, muts in enumerate(sample.mutation_sets):
                path = os.path.join(tmpdir, f'genotype_{i}.fasta')
                with open(path, 'w') as fh:
                    fh.write(f'>genotype_{i}\n{_decode_mutations(muts, ref)}\n')
                fasta_paths.append(path)

            genomes_str = ','.join(fasta_paths)

            # Build proportions string that sums to exactly 1.0 after float parsing.
            # Adjust the last value to absorb any rounding error.
            props = sample.proportions
            rounded = [round(p, 10) for p in props[:-1]]
            rounded.append(round(1.0 - sum(rounded), 10))
            proportions_str = ','.join(str(p) for p in rounded)

            # Build Click CLI args list (genomes positional first, then options)
            args = [genomes_str,
                    '--primers', primers,
                    '--reference', reference,
                    '--proportions', proportions_str,
                    '--outdir', outdir,
                    '--readcnt', str(readcnt),
                    ]

            for key, val in bygul_kwargs.items():
                args.extend([f'--{key}', str(val)])

            if redo:
                args.extend(['--redo'])

            # standalone_mode=False returns the result instead of calling sys.exit
            simulate_proportions.main(args=args, standalone_mode=False)
