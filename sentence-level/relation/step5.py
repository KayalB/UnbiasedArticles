import os
import json
import torch
import numpy as np
import networkx as nx
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# ===============================================================
# CONFIG
# ===============================================================
GRAPH_DIR = "erg_graphs"
# GRAPH_DIR = "toy_erg_graphs"

OUTPUT_DIR = "erg_features"
# OUTPUT_DIR = "toy_erg_features"

os.makedirs(OUTPUT_DIR, exist_ok=True)

ENCODER = SentenceTransformer("all-MiniLM-L6-v2")  # 384-dim

# Final bias mapping across steps
BIAS_MAP = {"left": 0, "center": 1, "right": 2}
DEFAULT_BIAS = "center"

DISCOURSE_WEIGHTS = {
    "Main": 1.0,
    "Main_Consequence": 0.9,
    "Cause_General": 0.8,
    "Cause_Specific": 0.8,
    "Speech": 0.7,
    "Distant_Evaluation": 0.6,
    "Distant_Expectation": 0.6,
    "Distant_Historical": 0.5,
    "Distant_Anecdotal": 0.5,
    "NONE": 0.3
}


# ===============================================================
# HELPERS
# ===============================================================
def norm(x):
    x = np.array(x)
    n = np.linalg.norm(x)
    return x if n == 0 else x / n


def get_bias(G):
    b = G.graph.get("article_bias", DEFAULT_BIAS)
    b = b.lower().strip()
    return b if b in BIAS_MAP else DEFAULT_BIAS


# ===============================================================
# ENCODER
# ===============================================================
def encode_graph(gpath):
    G = nx.read_graphml(gpath)
    aid = os.path.basename(gpath).replace(".graphml", "")

    article_bias = get_bias(G)
    label_id = BIAS_MAP[article_bias]

    sentence_feats = {}
    event_feats = {}
    meta = {}

    for node, data in G.nodes(data=True):

        ntype = data.get("node_type")

        # =======================================================
        # SENTENCE NODE
        # =======================================================
        if ntype == "sentence":
            text = data.get("text", "")
            emb = norm(ENCODER.encode(text))

            dr = data.get("discourse_role", "NONE")
            prev = data.get("prev_discourse_role", "NONE")
            nxt = data.get("next_discourse_role", "NONE")

            dr_w = DISCOURSE_WEIGHTS.get(dr, 0.5)
            pr_w = DISCOURSE_WEIGHTS.get(prev, 0.5)
            nr_w = DISCOURSE_WEIGHTS.get(nxt, 0.5)

            # event children
            children = [
                c for c in G.predecessors(node)
                if G.nodes[c].get("node_type") == "event"
            ]

            if children:
                sentiments = [float(G.nodes[c].get("sentiment", 0)) for c in children]
                bias_flags = [float(G.nodes[c].get("has_bias", False)) for c in children]

                mean_sent = np.mean(sentiments)
                std_sent = np.std(sentiments)
                min_sent = np.min(sentiments)
                max_sent = np.max(sentiments)
                count_events = len(children)
                bias_ratio = np.mean(bias_flags)
            else:
                mean_sent = std_sent = min_sent = max_sent = 0.0
                count_events = 0
                bias_ratio = 0.0

            indeg = G.in_degree(node)
            outdeg = G.out_degree(node)

            sentence_feats[node] = np.concatenate([
                emb,
                [
                    dr_w, pr_w, nr_w,
                    mean_sent, std_sent, min_sent, max_sent,
                    count_events, bias_ratio,
                    indeg, outdeg,
                ]
            ]).tolist()

            meta[node] = {
                "type": "sentence",
                "text": text[:120],
                "discourse_role": dr,
                "events": count_events,
                "bias_ratio": bias_ratio
            }

        # =======================================================
        # EVENT NODE
        # =======================================================
        elif ntype == "event":

            trig = data.get("trigger", "")
            emb = norm(ENCODER.encode(trig))

            actor = data.get("actor", "")
            obj = data.get("object", "")
            sent = float(data.get("sentiment", 0))
            has_b = float(data.get("has_bias", False))

            indeg = G.in_degree(node)
            outdeg = G.out_degree(node)

            # parent sentence
            parents = [
                p for p in G.successors(node)
                if G.nodes[p].get("node_type") == "sentence"
            ]
            if parents:
                p = G.nodes[parents[0]]
                dr = p.get("discourse_role", "NONE")
                dr_w = DISCOURSE_WEIGHTS.get(dr, 0.5)
            else:
                dr_w = 0.5

            # relation degrees
            causal = temporal = coref = cont = 0
            for _, tgt, ed in G.out_edges(node, data=True):
                if ed.get("edge_type") == "event_to_event":
                    if ed.get("relation") == "causal":
                        causal += 1
                    elif ed.get("relation") == "temporal":
                        temporal += 1
                    elif ed.get("relation") == "coreference":
                        coref += 1
                    elif ed.get("relation") == "continuation":
                        cont += 1

            event_feats[node] = np.concatenate([
                emb,
                [
                    sent, has_b,
                    float(bool(actor)),
                    float(bool(obj)),
                    indeg, outdeg,
                    dr_w,
                    causal, temporal, coref, cont
                ]
            ]).tolist()

            meta[node] = {
                "type": "event",
                "trigger": trig,
                "actor": actor[:50],
                "object": obj[:50],
                "sentiment": sent,
                "causal": causal,
                "temporal": temporal,
                "coref": coref,
                "cont": cont
            }

    return aid, article_bias, label_id, sentence_feats, event_feats, meta


# ===============================================================
# MAIN
# ===============================================================
print("🧩 Encoding graph features (Step 5)...")

for file in tqdm(os.listdir(GRAPH_DIR)):
    if not file.endswith(".graphml"):
        continue

    graph_path = os.path.join(GRAPH_DIR, file)

    try:
        (
            aid, bias_str, label_id,
            sent_feats, event_feats, meta
        ) = encode_graph(graph_path)
    except Exception as e:
        print(f"❌ Error on {file}: {e}")
        continue

    base = os.path.join(OUTPUT_DIR, aid)

    # save features
    torch.save(torch.tensor(list(sent_feats.values()), dtype=torch.float32), base + "_sentence.pt")
    torch.save(torch.tensor(list(event_feats.values()), dtype=torch.float32), base + "_event.pt")

    # save metadata
    with open(base + "_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # save article bias
    torch.save(torch.tensor([label_id], dtype=torch.long), base + "_label.pt")
    with open(base + "_label.json", "w") as f:
        json.dump({"article_bias": bias_str, "label_id": label_id}, f, indent=2)

print("✅ Step 5 complete → features saved in erg_features/")