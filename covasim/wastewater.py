'''
Wastewater sampling analyzer for Covasim.

WastewaterSampler is a cv.Analyzer that, at specified time points, snapshots
the set of circulating viral genotypes and their relative proportions weighted
by individual viral load — i.e. what a wastewater sample would look like.
Snapshots can be exported as FASTA or fed directly to Bygul's
simulate_proportions to generate synthetic sequencing reads.

Requires pars['seq_pars']['enable'] = True.
'''

import dataclasses
from collections import defaultdict

import numpy as np

from .analysis import Analyzer

__all__ = ['WastewaterSampler', 'WastewaterSample']


@dataclasses.dataclass
class WastewaterSample:
    '''Snapshot of circulating genotypes/variants and their viral-load-weighted proportions.

    In haplotype mode (group_by='haplotype'):
        sequences holds deduplicated ACGT strings, one per distinct evolved sequence.
        variant_labels is None.

    In variant mode (group_by='variant'):
        variant_labels holds variant name strings (e.g. 'wild', 'delta').
        sequences is None.
    '''
    day:            int
    date:           str
    sequences:      list            # ACGT strings (haplotype mode) or None (variant mode)
    proportions:    list            # normalized viral-load fractions, sums to 1.0
    raw_loads:      list            # un-normalized total viral load per genotype/variant
    n_infectious:   int             # number of currently infectious agents
    n_genotypes:    int             # number of distinct sequences/variants present
    group_by:       str  = 'haplotype'  # 'haplotype' or 'variant'
    variant_labels: list = None     # variant name strings (variant mode) or None (haplotype mode)


class WastewaterSampler(Analyzer):
    '''
    Analyzer that snapshots the viral genotype mixture present in a wastewater
    sample at one or more time points.

    For each sampled day:
      1. Identifies all currently infectious agents.
      2. Reads each agent's viral shedding from ``sim.people.viral_shedding``
         (``viral_load × rel_trans``), which is updated by ``sim.step()`` before
         analyzers are called.
      3. Retrieves each agent's evolved haplotype from LineageSequenceTracker.
      4. Groups identical haplotypes and sums their viral load contributions.
      5. Normalizes to proportions.

    Snapshots are stored in self.samples (dict: int day → WastewaterSample).
    Use to_fasta(day) to get a multi-FASTA string (haplotype mode only), or
    simulate_sample(day, ...) to pass the mixture directly to Bygul.

    Args:
        days     (list): simulation days (int) or calendar date strings to sample.
        label    (str):  optional label for the analyzer.
        group_by (str):  how to aggregate viral loads — either ``'haplotype'``
                         (default; groups by full evolved ACGT sequence, requires
                         seq_pars enabled) or ``'variant'`` (groups by named variant
                         such as 'wild', 'delta', as defined in sim['variant_map']).

    Example — haplotype mode (default)::

        ww = cv.WastewaterSampler(days=[56, 120])
        sim = cv.Sim(pars, analyzers=[ww])
        sim.run()
        print(ww.to_fasta(56))

    Example — variant mode::

        ww = cv.WastewaterSampler(days=[56, 120], group_by='variant')
        sim = cv.Sim(pars, analyzers=[ww])
        sim.run()
        sample = ww.samples[56]
        for name, prop in zip(sample.variant_labels, sample.proportions):
            print(f'{name}: {prop:.1%}')
    '''

    def __init__(self, days, label=None, group_by='haplotype'):
        super().__init__(label=label)
        if group_by not in ('haplotype', 'variant'):
            raise ValueError(f"group_by must be 'haplotype' or 'variant', got '{group_by}'")
        self._days_input = days  # raw user input; converted in initialize()
        self.group_by    = group_by
        self.samples     = {}

    def initialize(self, sim):
        super().initialize(sim)
        if self.group_by == 'haplotype' and not sim['seq_pars'].get('enable', False):
            raise ValueError(
                "WastewaterSampler with group_by='haplotype' requires "
                "pars['seq_pars']['enable'] = True"
            )
        # Convert string dates to integer days
        self._day_set = set()
        for d in self._days_input:
            self._day_set.add(sim.day(d) if isinstance(d, str) else int(d))

    def apply(self, sim):
        if sim.t not in self._day_set:
            return
        self._take_sample(sim)

    def _take_sample(self, sim):
        people = sim.people

        # Infectious agents only
        inds = np.nonzero(people.infectious)[0]

        if len(inds) == 0:
            self.samples[sim.t] = None
            return

        loads = sim.people.viral_shedding[inds]

        if self.group_by == 'variant':
            self.samples[sim.t] = self._sample_by_variant(sim, inds, loads)
        else:
            self.samples[sim.t] = self._sample_by_haplotype(sim, inds, loads)

    def _sample_by_variant(self, sim, inds, loads):
        '''Aggregate viral load by named variant (e.g. wild-type, introduced variants).'''
        variant_map     = sim['variant_map']          # int index → label str
        variant_indices = sim.people.infectious_variant[inds]

        load_by_variant = defaultdict(float)
        for vi, load in zip(variant_indices, loads):
            vname = variant_map.get(int(vi), f'variant_{int(vi)}')
            load_by_variant[vname] += float(load)

        total          = sum(load_by_variant.values())
        variant_labels = list(load_by_variant.keys())
        raw_loads      = [load_by_variant[v] for v in variant_labels]
        proportions    = [v / total for v in raw_loads]
        proportions[-1] = 1.0 - sum(proportions[:-1])

        return WastewaterSample(
            day            = sim.t,
            date           = sim.date(sim.t),
            sequences      = None,
            proportions    = proportions,
            raw_loads      = raw_loads,
            n_infectious   = int(len(inds)),
            n_genotypes    = len(variant_labels),
            group_by       = 'variant',
            variant_labels = variant_labels,
        )

    def _sample_by_haplotype(self, sim, inds, loads):
        '''Aggregate viral load by full evolved haplotype (ACGT sequence).'''
        tracker    = sim.people.sequence_tracker
        haplotypes = [tracker.reconstruct_haplotype(int(i)) for i in inds]

        load_by_seq = defaultdict(float)
        for hap, load in zip(haplotypes, loads):
            load_by_seq[hap] += float(load)

        total       = sum(load_by_seq.values())
        sequences   = list(load_by_seq.keys())
        raw_loads   = [load_by_seq[s] for s in sequences]
        proportions = [v / total for v in raw_loads]
        proportions[-1] = 1.0 - sum(proportions[:-1])

        return WastewaterSample(
            day          = sim.t,
            date         = sim.date(sim.t),
            sequences    = sequences,
            proportions  = proportions,
            raw_loads    = raw_loads,
            n_infectious = int(len(inds)),
            n_genotypes  = len(sequences),
            group_by     = 'haplotype',
        )

    def to_fasta(self, day):
        '''Return a multi-FASTA string for the genotype mixture on the given day.

        Only available when group_by='haplotype'. Raises ValueError otherwise.
        '''
        sample = self.samples[day]
        if sample is None:
            return ''
        if sample.group_by == 'variant':
            raise ValueError(
                "to_fasta() requires group_by='haplotype'. "
                "In variant mode use sample.variant_labels and sample.proportions directly."
            )
        lines = []
        for i, (seq, prop) in enumerate(zip(sample.sequences, sample.proportions)):
            lines.append(f'>genotype_{i}  proportion={prop:.8f}  day={day}  date={sample.date}')
            lines.append(seq)
        return '\n'.join(lines)

    def simulate_sample(self, day, primers, reference, outdir='./reads/', readcnt=500, redo=False, **bygul_kwargs):
        '''
        Write per-genotype FASTA files and invoke Bygul's simulate_proportions CLI.

        Only available when group_by='haplotype'. Raises ValueError otherwise.

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
        if sample.group_by == 'variant':
            raise ValueError(
                "simulate_sample() requires group_by='haplotype'. "
                "In variant mode use sample.variant_labels and sample.proportions directly."
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Write each unique genotype to its own FASTA file
            fasta_paths = []
            for i, seq in enumerate(sample.sequences):
                path = os.path.join(tmpdir, f'genotype_{i}.fasta')
                with open(path, 'w') as fh:
                    fh.write(f'>genotype_{i}\n{seq}\n')
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
