import subprocess
from collections import defaultdict

import dendropy


class Inference:
    def __init__(self, fasta_path, prefix, model="JC", iqtree_bin="/home/yutianc/miniforge3/envs/covasim4ww/bin/iqtree"):
        """
        Parameters
        ----------
        fasta_path : str    path to the alignment to run IQ-TREE on
        model      : str    substitution model, e.g. "GTR+G", "JC", "HKY+G"
        """
        self.fasta_path = fasta_path
        self.prefix = prefix
        self.model = model
        self.iqtree_bin = iqtree_bin

        self.treefile_path = None
        self.state_path = None

    def run_iqtree_asr(self, extra_args=None):
        cmd = [self.iqtree_bin, "-s", self.fasta_path, "-m", self.model, "-asr", "-redo", "-pre", self.prefix]
        if extra_args:
            cmd.extend(extra_args)

        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

        self.treefile_path = self.prefix + ".treefile"
        self.state_path = self.prefix + ".state"
        print(f"IQ-TREE finished. Tree: {self.treefile_path}, ancestral states: {self.state_path}")

    def _parse_state_file(self):
        """Parse the .state file into {node_name: {site: state}}.
        Only internal nodes appear here (e.g. Node21, Node22, ...)."""
        node_states = defaultdict(dict)
        header_skipped = False
        with open(self.state_path) as f:
            f.readline()  # header
            for line in f:
                if not line.strip():
                    continue
                if line.startswith("#"):
                    continue
                if not header_skipped:
                    header_skipped = True
                    continue
                fields = line.rstrip("\n").split("\t")
                node, site, state = fields[0], int(fields[1]), fields[2]
                node_states[node][site] = state
        return node_states

    def _parse_alignment_states(self):
        """Parse self.fasta_path into {tip_name: {site: state}}, 1-indexed
        to match the .state file's Site column convention."""
        from Bio import SeqIO

        tip_states = {}
        for record in SeqIO.parse(self.fasta_path, "fasta"):
            seq = str(record.seq).upper()
            tip_states[record.id] = {i + 1: nuc for i, nuc in enumerate(seq)}
        return tip_states

    def count_mutations(self):
        """
        Walk every branch in the IQ-TREE tree and count mutations by
        diffing the parent's state (always from .state, since parents are
        always internal nodes) against the child's state (from .state if
        internal, from the alignment if the child is a tip).

        Requires run_iqtree_asr() to have been called first.

        Returns
        -------
        total_mutations : int
        per_branch : dict {(parent_name, child_name): [(site, ref_state, alt_state), ...]}
        """
        if self.treefile_path is None or self.state_path is None:
            raise RuntimeError("Call run_iqtree_asr() before count_mutations().")

        tree = dendropy.Tree.get(
            path=self.treefile_path, 
            schema="newick", 
            suppress_internal_node_taxa=False,
            preserve_underscores=True
        )
        node_states = self._parse_state_file()
        tip_states = self._parse_alignment_states()

        total_mutations = 0
        per_branch = {}

        for edge in tree.preorder_edge_iter():
            child, parent = edge.head_node, edge.tail_node
            if parent is None:
                continue  # root has no incoming branch

            parent_name = parent.taxon.label if parent.taxon else parent.label
            child_name = child.taxon.label if child.taxon else child.label

            parent_seq = node_states.get(parent_name) or tip_states.get(parent_name)
            child_seq = node_states.get(child_name) or tip_states.get(child_name)

            if parent_seq is None or child_seq is None:
                print(f"WARNING: missing state for branch {parent_name} -> {child_name}, skipping")
                continue

            mutations = [
                (site, parent_seq[site], child_seq[site])
                for site in parent_seq
                if site in child_seq and parent_seq[site] != child_seq[site]
            ]

            per_branch[(parent_name, child_name)] = mutations
            total_mutations += len(mutations)

        return total_mutations, per_branch
