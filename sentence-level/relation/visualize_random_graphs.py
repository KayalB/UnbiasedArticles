import os
import random
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ============================================================
# CONFIG
# ============================================================
GRAPH_DIR = "erg_graphs"
OUTPUT_DIR = "graph_viz"

# GRAPH_DIR = "toy_erg_graphs"
# OUTPUT_DIR = "toy_graph_viz"

NUM_SAMPLES = 10
SHOW_EDGE_LABELS = False   # toggle to True if you want text on edges

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# LOAD 10 RANDOM GRAPHS
# ============================================================
files = [f for f in os.listdir(GRAPH_DIR) if f.endswith(".graphml")]
sample_files = random.sample(files, min(NUM_SAMPLES, 1))

print("\n🎨 Visualizing graphs:")
for f in sample_files:
    print(" -", f)


# ============================================================
# COLOR MAPS
# ============================================================
NODE_COLORS = {
    "sentence": "#4C7FFF",   # blue
    "event": "#4C7FFF"       # orange
}

RELATION_COLORS = {
    "causal": "red",
    "temporal": "green",
    "coreference": "purple",
    "continuation": "orange",
}

# Legend handles
EDGE_LEGEND = [
    Line2D([0], [0], color="red", lw=2, label="Causal"),
    Line2D([0], [0], color="green", lw=2, label="Temporal"),
    Line2D([0], [0], color="purple", lw=2, label="Coreference"),
    Line2D([0], [0], color="orange", lw=2, label="Continuation"),
]


# ============================================================
# DRAW GRAPH
# ============================================================
def draw_graph(G, out_path):

    plt.figure(figsize=(14, 10))

    pos = nx.spring_layout(G, seed=42, k=0.25)

    # Node colors by type
    node_colors = []
    for n, d in G.nodes(data=True):
        ntype = d.get("node_type", "event")
        node_colors.append(NODE_COLORS.get(ntype, "#AAAAAA"))

    # Edge colors by relation type
    edge_colors = []
    edge_labels = {}

    for u, v, d in G.edges(data=True):
        rel = d.get("relation", "other")

        # sentence-to-sentence edges use "sequential" or discourse
        if d.get("edge_type") == "sentence_to_sentence":
            if rel in ["causal_discourse", "consequence_discourse"]:
                rel = rel  # will fall back to black or orange
            else:
                rel = "sequential"

        # event-to-sentence always belongs_to
        if d.get("edge_type") == "event_to_sentence":
            rel = "belongs_to"

        color = RELATION_COLORS.get(rel, "black")
        if color!="black":
            edge_colors.append(color)

        if SHOW_EDGE_LABELS:
            edge_labels[(u, v)] = rel

    # Draw nodes & edges
    nx.draw(
        G,
        pos,
        node_size=220,
        node_color=node_colors,
        edge_color=edge_colors,
        with_labels=False,
        alpha=0.85
    )

    # Edge labels
    if SHOW_EDGE_LABELS:
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    plt.title(f"Article {G.graph.get('article_id')} — Bias: {G.graph.get('article_bias')}", fontsize=15)

    plt.legend(handles=EDGE_LEGEND, loc="lower left", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# ============================================================
# EXECUTE
# ============================================================
for f in sample_files:
    print(f"📌 Rendering {f}...")
    G = nx.read_graphml(os.path.join(GRAPH_DIR, f))
    out = os.path.join(OUTPUT_DIR, f"{f}.png")
    draw_graph(G, out)

print("\n✅ Done! Visualizations saved to graph_viz/")