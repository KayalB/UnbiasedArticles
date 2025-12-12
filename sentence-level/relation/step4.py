import json
import os
import networkx as nx
from tqdm import tqdm


# ===============================================================
# CONFIG
# ===============================================================
DISCOURSE_WITH_EVENTS = "discourse_with_events.json"            # Step 2 output

ERG_EDGES = "erg_edges_adjacency.json"
# ERG_EDGES = "erg_edges_transformer.json"                        # Step 3B output
# ERG_EDGES = "toy_erg_edges_transformer.json"                        # Step 3B output

OUTPUT_DIR = "erg_graphs"
# OUTPUT_DIR = "toy_erg_graphs"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ===============================================================
# BUILD GRAPH FOR ONE ARTICLE
# ===============================================================
def build_article_graph(article_id, discourse_data, erg_edges):

    G = nx.DiGraph()

    # Filter article sentences + predicted edges
    sents = [s for s in discourse_data if s["article_id"] == article_id]
    edges = [e for e in erg_edges if e["article_id"] == article_id]

    sents = sorted(sents, key=lambda s: s["sentence_idx"])

    # ---- article bias ----
    if sents:
        article_bias = sents[0].get("article_bias_normalized", "center")
    else:
        article_bias = "center"

    G.graph["article_id"] = article_id
    G.graph["article_bias"] = article_bias

    # ===============================================================
    # 1. SENTENCE NODES
    # ===============================================================
    sent_id_map = {}

    for sent in sents:
        idx = sent["sentence_idx"]
        sid = f"{article_id}_sent_{idx}"
        sent_id_map[idx] = sid

        G.add_node(
            sid,
            node_type="sentence",
            text=str(sent["text"]),
            sentence_idx=int(idx),
            article_id=article_id,
            article_bias=article_bias,
            discourse_role=sent.get("discourse_role", "NONE"),
            prev_discourse_role=sent.get("prev_discourse_role", "NONE"),
            next_discourse_role=sent.get("next_discourse_role", "NONE"),
            has_bias=bool(sent.get("has_bias", False))
        )

    # ===============================================================
    # 2. EVENT NODES
    # ===============================================================
    for sent in sents:
        s_idx = sent["sentence_idx"]
        sid = sent_id_map[s_idx]

        for ev_idx, ev in enumerate(sent.get("events", [])):

            eid = f"{article_id}_{s_idx}_ev{ev_idx}"

            G.add_node(
                eid,
                node_type="event",
                trigger=str(ev.get("trigger", "")).lower(),
                actor=str(ev.get("actor", "")),
                object=str(ev.get("object", "")),
                sentiment=float(ev.get("sentiment", 0)),
                has_bias=bool(ev.get("has_bias", False)),
                article_bias=article_bias
            )

            # event → sentence
            G.add_edge(
                eid, sid,
                relation="belongs_to",
                edge_type="event_to_sentence",
                weight=1.0
            )

    # ===============================================================
    # 3. SENTENCE → SENTENCE EDGES (sequential)
    # ===============================================================
    for i in range(len(sents) - 1):
        s1 = sent_id_map[sents[i]["sentence_idx"]]
        s2 = sent_id_map[sents[i+1]["sentence_idx"]]

        # Basic sequential
        G.add_edge(
            s1, s2,
            relation="sequential",
            edge_type="sentence_to_sentence",
            weight=1.0
        )

        # Optional discourse-based extra edges
        r1 = sents[i].get("discourse_role", "NONE")
        r2 = sents[i+1].get("discourse_role", "NONE")

        if r1.startswith("Cause") and r2 == "Main":
            G.add_edge(
                s1, s2,
                relation="causal_discourse",
                edge_type="sentence_to_sentence",
                weight=1.5
            )

        if r1 == "Main" and r2 == "Main_Consequence":
            G.add_edge(
                s1, s2,
                relation="consequence_discourse",
                edge_type="sentence_to_sentence",
                weight=1.5
            )

    # ===============================================================
    # 4. EVENT → EVENT EDGES (from Step 3B)
    # ===============================================================
    missing_src = missing_tgt = 0

    for e in edges:
        src = e["src_event_id"]
        tgt = e["tgt_event_id"]

        if src not in G:
            missing_src += 1
            continue
        if tgt not in G:
            missing_tgt += 1
            continue

        G.add_edge(
            src, tgt,
            relation=e["relation"],
            edge_type="event_to_event",
            weight=float(e.get("confidence", 1.0)),
            sentence_distance=int(e.get("sentence_distance", 0))
        )

    if missing_src or missing_tgt:
        print(f"⚠️ Article {article_id}: missing_src={missing_src}, missing_tgt={missing_tgt}")

    return G


# ===============================================================
# MAIN
# ===============================================================
def main():
    print("📘 Loading Step 2 + Step 3B outputs...")
    discourse = json.load(open(DISCOURSE_WITH_EVENTS))
    edges = json.load(open(ERG_EDGES))

    article_ids = sorted({s["article_id"] for s in discourse})
    print(f"📰 Found {len(article_ids)} articles.")

    for aid in tqdm(article_ids, desc="Building ERGs"):
        G = build_article_graph(aid, discourse, edges)
        nx.write_graphml(G, os.path.join(OUTPUT_DIR, f"{aid}.graphml"))

    print("✅ Step 4 complete — all graphs saved.")


if __name__ == "__main__":
    main()