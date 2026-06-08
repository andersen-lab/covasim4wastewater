'''
Variant fitness models for Covasim.

Each model maps a frozenset of encoded mutations (site_0indexed, from_nt_int, to_nt_int)
to a rel_beta multiplier (float). Add new subclasses of VariantFitnessModel to support
alternative data sources or fitness frameworks.
'''

import csv
import math
from abc import ABC, abstractmethod

__all__ = ['VariantFitnessModel', 'BloomNtFitnessModel']


class VariantFitnessModel(ABC):
    '''Abstract base: load fitness data and compute rel_beta multiplier from mutations.'''

    def __init__(self, fitness_path, scale=0.1):
        self.scale  = scale
        self._cache = {}
        self.load(fitness_path)

    @abstractmethod
    def load(self, fitness_path: str) -> None:
        '''Load fitness data from file into whatever structure compute_fitness needs.'''

    @abstractmethod
    def compute_fitness(self, mutations: frozenset) -> float:
        '''Return rel_beta multiplier for a haplotype.

        Args:
            mutations: frozenset of (site_0indexed, from_nt_int, to_nt_int)

        Returns:
            float: rel_beta multiplier; 1.0 for an empty or unknown haplotype
        '''


class BloomNtFitnessModel(VariantFitnessModel):
    '''
    Fitness from the Bloom lab nt_fitness.csv (results_public_2024-11-06).

    Each row gives a pre-computed log-fitness value for one (site, allele) pair.
    The reference allele at each site has fitness≈0; alternate alleles have
    negative (deleterious) or positive (beneficial) values.

    Fitness multiplier: exp(Σ fitness_i * scale), summed over mutations in haplotype.
    '''
    NT_MAP = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

    def load(self, fitness_path: str) -> None:
        self.lookup = {}  # (nt_site_1indexed, nt_int) → log-fitness float
        with open(fitness_path) as f:
            for row in csv.DictReader(f):
                site = int(row['nt_site'])
                nt   = self.NT_MAP[row['nt']]
                self.lookup[(site, nt)] = float(row['fitness'])

    def compute_fitness(self, mutations: frozenset) -> float:
        if mutations in self._cache:
            return self._cache[mutations]
        log_fit = sum(
            self.lookup.get((site + 1, to_nt), 0.0)  # 0-indexed → 1-indexed
            for site, _, to_nt in mutations
        )
        result = math.exp(log_fit * self.scale)
        self._cache[mutations] = result
        return result
