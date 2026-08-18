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
from .sequence_evolution import agent_mutations as _agent_mutations
from .sequence_evolution import decode_mutations as _decode_mutations

__all__ = ['WastewaterSampler', 'WastewaterSample']


@dataclasses.dataclass
class WastewaterSample:
    '''Snapshot of circulating haplotypes and their viral-shedding-weighted proportions.'''
    day:              int
    date:             str
    mutation_sets:    list  # deduplicated frozensets of (site_0idx, ref_nt_int, alt_nt_int), one per distinct haplotype
    variant_labels:   list  # variant label string for each distinct haplotype
    proportions:      list  # normalized viral-shedding fractions, sums to 1.0
    raw_loads:        list  # un-normalized total viral shedding per haplotype
    n_infectious:     int   # number of currently infectious agents
    n_genotypes:      int   # number of distinct haplotypes present
    variant_shedding: dict = None  # {variant_label: total_shedding} aggregated by named variant
    region:           int = None
    total_load:       float = 0.0


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
        regions (list/str): region IDs to sample, or ``'all'``
        capture_rates (dict): optional ``{region: fraction_entering_sample}``

    Example::

        ww = cv.WastewaterSampler(days=[56, 120])
        sim = cv.Sim(pars, analyzers=[ww])
        sim.run()
        print(ww.to_fasta(56))
    '''

    def __init__(self, days, label=None, regions=None, capture_rates=None):
        super().__init__(label=label)
        self._days_input = days  # raw user input; converted in initialize()
        self._regions_input = regions
        self.capture_rates = capture_rates or {}
        self.samples     = {}
        self.region_samples = {}
        self._reference  = None  # set in initialize()
        self._regions = []

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

        available_regions = [int(r) for r in np.unique(sim.people.region)]
        if self._regions_input is None:
            self._regions = []
        elif isinstance(self._regions_input, str) and self._regions_input == 'all':
            self._regions = available_regions
        else:
            self._regions = [int(r) for r in self._regions_input]
            unknown = sorted(set(self._regions) - set(available_regions))
            if unknown:
                raise ValueError(f'Unknown wastewater region IDs: {unknown}')

        normalized_rates = {}
        for region, rate in self.capture_rates.items():
            rate = float(rate)
            if rate < 0:
                raise ValueError('Wastewater capture rates must be non-negative')
            normalized_rates[int(region)] = rate
        self.capture_rates = normalized_rates

    def apply(self, sim):
        if sim.t not in self._day_set:
            return
        self._take_sample(sim)

    def _capture_weights(self, sim, inds):
        '''Return the fraction of each agent's shedding entering the sample.'''
        weights = np.ones(len(inds), dtype=float)
        if self.capture_rates:
            regions = sim.people.region[inds]
            for region, rate in self.capture_rates.items():
                weights[regions == region] = rate
        return weights

    def _take_sample(self, sim):
        # Infectious agents only
        inds = np.nonzero(sim.people.infectious)[0]

        self.samples[sim.t] = self._build_sample(sim=sim, inds=inds, region=None)

        regional = {}
        for region in self._regions:
            region_inds = inds[sim.people.region[inds] == region]
            regional[region] = self._build_sample(
                sim=sim,
                inds=region_inds,
                region=region,
            )
        self.region_samples[sim.t] = regional

    def _build_sample(self, sim, inds, region=None):
        '''Build one pooled or regional sample.'''
        if len(inds) == 0:
            return None

        loads = np.asarray(sim.people.viral_shedding[inds], dtype=float)
        loads = loads * self._capture_weights(sim, inds)
        total = float(loads.sum())
        if total <= 0:
            return None

        tracker     = sim.people.sequence_tracker
        ref         = self._reference
        variant_map = sim.pars.get('variant_map', {})

        load_by_muts  = defaultdict(float)
        label_by_muts = {}   # first-seen variant label per distinct haplotype
        load_by_variant = defaultdict(float)

        for idx, load in zip(inds, loads):
            if load <= 0:
                continue

            i    = int(idx)
            muts = _agent_mutations(tracker, i, ref)
            load_by_muts[muts] += float(load)
            v = sim.people.infectious_variant[i]
            if np.isnan(v):
                variant_label = 'wild'
            else:
                variant_label = variant_map.get(
                    int(v),
                    f'variant_{int(v)}',
                )

            label_by_muts.setdefault(muts, variant_label)
            load_by_variant[variant_label] += float(load)

        mutation_sets = list(load_by_muts)
        raw_loads     = [load_by_muts[m] for m in mutation_sets]
        proportions = [load / total for load in raw_loads]
        if proportions:
            proportions[-1] = 1.0 - sum(proportions[:-1])

        return WastewaterSample(
            day              = sim.t,
            date             = sim.date(sim.t),
            mutation_sets    = mutation_sets,
            variant_labels   = [label_by_muts[m] for m in mutation_sets],
            proportions      = proportions,
            raw_loads        = raw_loads,
            n_infectious     = int(len(inds)),
            n_genotypes      = len(mutation_sets),
            variant_shedding = dict(load_by_variant),
            region=region,
            total_load=total,
        )

    def get_sample(self, day, region=None):
        '''Return a pooled sample or a sample for one region.'''
        day = int(day)
        if region is None:
            return self.samples[day]
        return self.region_samples[day][int(region)]
    

    def to_fasta(self, day, region=None):
        '''Return a multi-FASTA string for a pooled or regional sample.'''
        sample = self.get_sample(day, region=region)
        if sample is None:
            return ''
        ref   = self._reference
        region_text = 'pooled' if sample.region is None else str(sample.region)
        lines = []
        for i, (muts, prop, vlabel) in enumerate(zip(sample.mutation_sets, sample.proportions, sample.variant_labels)):
            lines.append(f'>genotype_{i}  proportion={prop:.8f}  day={day}  '
                f'date={sample.date}  region={region_text}  variant={vlabel}'
            )
            lines.append(_decode_mutations(muts, ref))
        return '\n'.join(lines)

    def simulate_sample(self, day, primers, reference, outdir='./reads/', readcnt=500, redo=False, region=None, **bygul_kwargs):
        ''' 
        Write a multi-FASTA file and a proportions CSV, then invoke Bygul's simulate_proportions CLI.

            Bygul's simulate_proportions command is a Click CLI that expects genome
            sequences inside a multi-FASTA file and their relative abundances in a CSV.
            This method writes temporary files, builds the required CLI argument list, 
            and calls the command via Click's standalone_mode=False interface.

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
        import csv

        try:
            from bygul._cli import simulate_proportions
        except ImportError as e:
            raise ImportError(
                "The 'bygul' package is required for WastewaterSampler.simulate(). "
                    "Install it from the Andersen Lab: https://github.com/andersen-lab/Bygul"
            ) from e

        sample = self.get_sample(day, region=region)
        if sample is None:
            where = 'pooled sample' if region is None else f'region {region}'
            raise ValueError(
                f'No measurable infectious shedding was present in the {where} on day {day}'
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            suffix = 'pooled' if region is None else f'region{int(region)}'
            multifasta_path = os.path.join(tmpdir, f'wastewater_day{day}_{suffix}.fasta')
            csv_path = os.path.join(tmpdir, f'proportions_day{day}_{suffix}.csv')
            
            ref = self._reference

            # Build proportions list that sums to exactly 1.0 after float parsing.
            # Adjust the last value to absorb any rounding error.
            props = sample.proportions
            rounded = [round(p, 10) for p in props[:-1]]
            rounded.append(round(1.0 - sum(rounded), 10))

            # Write both the multi-FASTA and the proportions CSV
            with open(multifasta_path, 'w') as f_fasta, open(csv_path, 'w', newline='') as f_csv:
                csv_writer = csv.writer(f_csv)
                csv_writer.writerow(['sample_name', 'proportion'])

                for i, muts in enumerate(sample.mutation_sets):
                    seq_name = f'genotype_{i}'
                    seq = _decode_mutations(muts, ref)

                    # Append sequence to the multi-FASTA
                    f_fasta.write(f'>{seq_name}\n{seq}\n')

                    # Write row to the CSV mapping name to proportion
                    csv_writer.writerow([seq_name, rounded[i]])

            # Build Click CLI args list using the new --multifasta and --csv parameters
            args = [
                '--multifasta', multifasta_path,
                '--csv', csv_path,
                '--primers', primers,
                '--reference', reference,
                '--outdir', outdir,
                '--readcnt', str(readcnt),
            ]

            for key, val in bygul_kwargs.items():
                args.extend([f'--{key}', str(val)])
                
                if redo:
                    args.extend(['--redo'])

                # standalone_mode=False returns the result instead of calling sys.exit
            return simulate_proportions.main(args=args, standalone_mode=False)
