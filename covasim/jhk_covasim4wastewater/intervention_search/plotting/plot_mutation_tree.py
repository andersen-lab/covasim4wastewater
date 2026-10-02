"""
Static tree visualizations: the raw event-based / mutation-collapsed
infection tree, and a rendered phylogenetic (JC69) tree.
"""

import matplotlib.pyplot as plt
from Bio import Phylo
from networkx.drawing.nx_agraph import graphviz_layout

from covasim.jhk_covasim4wastewater.intervention_search.analysis.mutation_tree import extract_sequenced_subtree


def _maybe_restrict(G, sequenced_only, collapse):
    if collapse:
        raise NotImplementedError(
            "collapse_clades was removed: it referenced an edge attribute "
            "('mutations') that the tree builders never set, and no "
            "notebook ever called this with collapse=True."
        )
    if sequenced_only:
        G = extract_sequenced_subtree(G)
    return G


def plot_mutation_tree(G, savepath=None, show=True, detection_day=None,
                        sequenced_only=False, collapse=False):
    """Tidy-layout tree; only sequenced nodes are drawn as points."""
    G = _maybe_restrict(G, sequenced_only, collapse)

    pos = graphviz_layout(G, prog="dot")
    fig, ax = plt.subplots(figsize=(14, 12))

    for src, tgt, data in G.edges(data=True):
        if src not in pos or tgt not in pos:
            continue
        x1, y1 = pos[src]
        x2, y2 = pos[tgt]
        lw = 0.5 + 0.1 * data.get("n_mut", 1)
        ax.plot([x1, x2], [y1, y2], color="black", linewidth=lw, alpha=0.8)

    seq_xy = [pos[n] for n in G.nodes if G.nodes[n].get("sequenced", False) and n in pos]
    if seq_xy:
        ax.scatter(*zip(*seq_xy), color="red", s=20, alpha=0.9)

    ax.set_title("Event-based Infection Tree", fontsize=14)
    ax.axis("off")
    plt.tight_layout()

    if savepath:
        plt.savefig(savepath, dpi=200)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_erase_mutation_tree(G, savepath=None, show=True, detection_day=None,
                              erase=True, sequenced_only=False, collapse=False):
    """
    Same tree, but edges leading to a subtree with no sequenced event are
    either dropped (erase=True) or faded (erase=False).
    """
    G = _maybe_restrict(G, sequenced_only, collapse)

    pos = graphviz_layout(G, prog="dot")
    fig, ax = plt.subplots(figsize=(14, 12))

    for src, tgt, data in G.edges(data=True):
        if src not in pos or tgt not in pos:
            continue
        if not G.nodes[tgt].get("has_seq_descendant", False):
            if erase:
                continue
            alpha, color = 0.4, "black"
        else:
            alpha, color = 1.0, "blue"

        x1, y1 = pos[src]
        x2, y2 = pos[tgt]
        lw = 1 + 0.2 * data.get("n_mut", 1)
        ax.plot([x1, x2], [y1, y2], color=color, linewidth=lw, alpha=alpha)

    seq_xy = [pos[n] for n in G.nodes if G.nodes[n].get("sequenced", False) and n in pos]
    if seq_xy:
        ax.scatter(*zip(*seq_xy), color="blue", s=40, alpha=0.9)

    ax.set_title("Ground Truth Transmission Tree (sequenced = blue)", fontsize=30)
    if detection_day is not None:
        ax.text(0.5, 1.02, f"Detection day: {detection_day}", transform=ax.transAxes,
                ha="center", fontsize=11, color="blue")
    ax.axis("off")
    plt.tight_layout()

    if savepath:
        plt.savefig(savepath, dpi=1000)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_phylo_tree(newick_path, savepath=None):
    tree = Phylo.read(newick_path, "newick")
    fig = plt.figure(figsize=(10, 40))
    ax = fig.add_subplot(1, 1, 1)
    Phylo.draw(tree, do_show=False, axes=ax)

    for text in ax.texts:
        text.set_fontsize(6)
        text.set_color("#2c3e50")

    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.get_yaxis().set_visible(False)

    plt.title("Phylogenetic Tree (JC69)", fontsize=14, pad=20)
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.show()