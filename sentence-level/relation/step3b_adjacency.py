import json
from tqdm import tqdm

# ===============================================================
# CONFIG (MATCHES STEP 3B)
# ===============================================================
DISCOURSE_WITH_EVENTS = "discourse_with_events.json"   # Step 2 output
OUTPUT_JSON = "erg_edges_adjacency.json"               # NEW baseline output

# ===============================================================
# MAIN PIPELINE
# ===============================================================
def build_adjacency_edges():
    print("📘 Loading discourse_with_events.json ...")
    data = json.load(open(DISCOURSE_WITH_EVENTS))

    # Group sentences by article
    articles = {}
    for s in data:
        articles.setdefault(s["article_id"], []).append(s)

    all_edges = []

    print(f"📰 Found {len(articles)} articles.")

    for aid, sents in tqdm(articles.items(), desc="Building adjacency ERGs"):

        # Sort sentences
        sents = sorted(sents, key=lambda s: s["sentence_idx"])

        # Build list of (event_id, sentence_idx)
        sentence_events = {}

        for sent in sents:
            sid = sent["sentence_idx"]
            sentence_events[sid] = []

            for ev_idx, ev in enumerate(sent.get("events", [])):
                ev_id = f"{aid}_{sid}_ev{ev_idx}"
                sentence_events[sid].append(ev_id)

        # Create adjacency-based event-event edges
        for i in range(len(sents) - 1):
            s1_idx = sents[i]["sentence_idx"]
            s2_idx = sents[i + 1]["sentence_idx"]

            evs_1 = sentence_events.get(s1_idx, [])
            evs_2 = sentence_events.get(s2_idx, [])

            for src in evs_1:
                for tgt in evs_2:
                    all_edges.append({
                        "article_id": aid,
                        "src_event_id": src,
                        "tgt_event_id": tgt,
                        "relation": "continuation",
                        "confidence": 1.0,
                        "sentence_distance": 1
                    })

    print(f"💾 Saving {len(all_edges)} adjacency edges → {OUTPUT_JSON}")
    with open(OUTPUT_JSON, "w") as f:
        json.dump(all_edges, f, indent=2, ensure_ascii=False)

    print("✅ Step 3B (Adjacency Baseline) complete.")


if __name__ == "__main__":
    build_adjacency_edges()
