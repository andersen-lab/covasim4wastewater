"""
Builds transmission / mutation trees from Covasim's infection_log, prunes
them down to the sequenced lineage, and exports them for phylogenetic
inference (IQ-TREE).
"""

import subprocess

import covasim as cv
import networkx as nx
from ete3 import Tree


def build_mutation_tree(sim, daily_sequenced_agents):
    """
    Event-based tree: each node is one infection event ("{agent}_{day}"),
    so every event has exactly one parent and the result is a tree.
    """
    infection_log = getattr(sim, "infection_log", None) or getattr(sim.people, "infection_log", None)
    if infection_log is None:
        raise ValueError("Simulation has no infection_log.")

    sequenced_event_ids = {eid for day_list in daily_sequenced_agents for eid in day_list}

    G = nx.DiGraph()
    last_event_date = {}

    for entry in infection_log:
        src, tgt, date = entry["source"], entry["target"], entry["date"]

        tgt_event = f"{tgt}_{date}"
        if tgt_event not in G:
            G.add_node(tgt_event)
            G.nodes[tgt_event]["sequenced"] = tgt_event in sequenced_event_ids
            G.nodes[tgt_event]["has_seq_descendant"] = G.nodes[tgt_event]["sequenced"]

        if src is None:
            src_event = "None_0"
            if src_event not in G:
                G.add_node(src_event, sequenced=False, has_seq_descendant=False)
        else:
            src_event = f"{src}_{last_event_date[src]}"

        G.add_edge(src_event, tgt_event)
        last_event_date[tgt] = date

        if G.nodes[tgt_event]["sequenced"]:
            for ancestor in nx.ancestors(G, tgt_event):
                G.nodes[ancestor]["has_seq_descendant"] = True

    return G


def build_infection_seq_tree(sim, daily_sequenced_agents):
    """
    Mutation-collapsed tree: each node is a unique mutation state
    (haplotype); each edge is a mutation event. Tracks, per node, whether
    it (or a descendant) was ever sequenced.
    """
    infection_log = getattr(sim, "infection_log", None) or getattr(sim.people, "infection_log", None)
    if infection_log is None:
        raise ValueError("Simulation has no infection_log.")

    sequenced_event_ids = {eid for day_list in daily_sequenced_agents for eid in day_list}

    mutation_nodes = {}
    node_events = {}
    G = nx.DiGraph()

    root_state = frozenset()
    mutation_nodes[root_state] = "Node_0"
    node_events["Node_0"] = []
    G.add_node("Node_0", mutations=root_state, sequenced=False, has_seq_descendant=False)

    current_state = {}

    def mark_sequenced(node_id, event_id):
        node_events[node_id].append(event_id)
        if event_id in sequenced_event_ids:
            G.nodes[node_id]["sequenced"] = True
            G.nodes[node_id]["has_seq_descendant"] = True
            for ancestor in nx.ancestors(G, node_id):
                G.nodes[ancestor]["has_seq_descendant"] = True

    for entry in infection_log:
        src, tgt, date = entry["source"], entry["target"], entry["date"]
        event_id = f"{tgt}_{date}"
        branch_mut = tuple(entry.get("branch_mutations", []))
        parent_state = root_state if src is None else current_state[src]

        if len(branch_mut) == 0:
            current_state[tgt] = parent_state
            mark_sequenced(mutation_nodes[parent_state], event_id)
            continue

        new_state = frozenset(parent_state | set(branch_mut))
        if new_state not in mutation_nodes:
            node_id = f"Node_{len(mutation_nodes)}"
            mutation_nodes[new_state] = node_id
            node_events[node_id] = []
            G.add_node(node_id, mutations=new_state, sequenced=False, has_seq_descendant=False)
            G.add_edge(mutation_nodes[parent_state], node_id)

        current_state[tgt] = new_state
        mark_sequenced(mutation_nodes[new_state], event_id)

    return G, mutation_nodes, node_events


def extract_sequenced_subtree(G):
    """
    Prune down to nodes whose subtree has at least one sequenced event,
    using the precomputed "has_seq_descendant" flag set by both builders
    above.
    """
    H = G.copy()
    to_remove = [n for n in H.nodes if not H.nodes[n].get("has_seq_descendant", False)]
    H.remove_nodes_from(to_remove)
    return H


def mutation_tree_to_newick(G, root="Node_0_internal"):
    def dfs(node):
        children = list(G.successors(node))
        label = node
        ann = []
        if G.nodes[node].get("sequenced", False):
            ann.append("sequenced=true")
        if G.nodes[node].get("has_seq_descendant", False):
            ann.append("has_seq_descendant=true")
        if ann:
            label = f"{label}[&{','.join(ann)}]"

        if not children:
            return f"{label}:1.0"
        child_str = ",".join(dfs(c) for c in children)
        return f"({child_str}){label}" if node == root else f"({child_str}){label}:1.0"

    return dfs(root) + ";"


def add_internal_leaves(G):
    """Give every internal node a pendant leaf copy, so a Newick export
    keeps ancestral (non-tip) sequences visible in the rendered tree."""
    internal_nodes = [n for n in G.nodes if G.out_degree(n) > 0]
    mapping = {n: f"{n}_internal" for n in internal_nodes}
    G = nx.relabel_nodes(G, mapping)

    for old_name, new_name in mapping.items():
        G.add_edge(new_name, old_name)
        G.nodes[old_name]["sequenced"] = True
        G.nodes[old_name]["has_seq_descendant"] = True
        G.nodes[old_name]["mutations"] = G.nodes[new_name].get("mutations", [])

    return G


def set_all_branch_lengths_to_one(newick_path, output_path=None):
    t = Tree(newick_path, format=1)
    for node in t.traverse():
        node.dist = 1.0
    output_path = output_path or newick_path.replace(".treefile", "_bl1.treefile")
    t.write(format=1, outfile=output_path)
    return output_path


def reconstruct_haplotype_from_mutation_state(sim, mutation_state):
    """mutation_state: frozenset of (site, ref_nt, alt_nt) tuples, relative to the reference (root)."""
    ref = sim.sequence_tracker.reference.copy()
    for site, ref_nt, alt_nt in mutation_state:
        ref[site] = alt_nt
    return cv.decode_sequence(ref)


def export_fasta_from_G(G_prune, sim, filepath):
    with open(filepath, "w") as f:
        for node in G_prune.nodes:
            hap = reconstruct_haplotype_from_mutation_state(sim, G_prune.nodes[node]["mutations"])
            f.write(f">{node}\n{hap}\n")


def infer_tree_jc69(fasta_path):
    """Runs IQ-TREE (JC69 model). Output: <fasta_path>.treefile"""
    subprocess.run(["iqtree", "-s", fasta_path, "-m", "JC69", "-czb"], check=True)
    return fasta_path + ".treefile"