import json
import torch
import re
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

# ===============================================================
# CONFIG
# ===============================================================
DISCOURSE_WITH_EVENTS = "discourse_with_events.json"     # Step 2 output
# DISCOURSE_WITH_EVENTS = "toy_discourse_with_events.json"     # Step 2 output

MODEL_DIR = "edge_classifier_final"                      # Step 3 trained model
OUTPUT_JSON = "erg_edges_transformer.json"               # Step 3B output
# OUTPUT_JSON = "toy_erg_edges_transformer.json"               # Step 3B output

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LABEL2ID = {
    "causal": 0,
    "temporal": 1,
    "coreference": 2,
    "continuation": 3,
    "none": 4
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


# ===============================================================
# HELPER: REBEL-style fields
# ===============================================================
def get_trigger(ev): return ev.get("trigger", "").strip().lower()
def get_actor(ev):   return ev.get("actor", "").strip().lower()
def get_object(ev):  return ev.get("object", "").strip().lower()


# ===============================================================
# LOAD MODEL
# ===============================================================
print("🔧 Loading edge classifier from:", MODEL_DIR)

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(DEVICE)
model.eval()


# ===============================================================
# CLASSIFIER PREDICTION
# ===============================================================
def classify_pair(text1, text2):
    inputs = tokenizer(
        text1,
        text2,
        truncation=True,
        padding="max_length",
        max_length=256,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()[0]
    pred_id = int(np.argmax(probs))
    pred_label = ID2LABEL[pred_id]
    confidence = float(probs[pred_id])

    return pred_label, confidence


# ===============================================================
# MAIN PIPELINE (per article)
# ===============================================================
def apply_classifier():
    print("📘 Loading discourse_with_events.json ...")
    data = json.load(open(DISCOURSE_WITH_EVENTS))

    # Group by article
    articles = {}
    for s in data:
        articles.setdefault(s["article_id"], []).append(s)

    all_edges = []

    print(f"📰 Found {len(articles)} articles.")
    for aid, sents in tqdm(articles.items(), desc="Scoring edges"):

        # Sort consistently
        sents = sorted(sents, key=lambda s: s["sentence_idx"])

        # Expand into events list with IDs
        events = []
        for s in sents:
            sid = s["sentence_idx"]
            for ev_idx, ev in enumerate(s.get("events", [])):
                ev_id = f"{aid}_{sid}_ev{ev_idx}"
                events.append({
                    "event": ev,
                    "sentence": s,
                    "event_id": ev_id
                })

        # Pairwise scoring (local window)
        for i in range(len(events)):
            e1 = events[i]
            s1 = e1["sentence"]
            ev1 = e1["event"]

            for j in range(i + 1, min(i + 8, len(events))):
                e2 = events[j]
                s2 = e2["sentence"]
                ev2 = e2["event"]

                dist = abs(s1["sentence_idx"] - s2["sentence_idx"])
                if dist > 5:
                    continue

                # Build text inputs (same format as Step 3)
                text1 = (
                    f"Sentence: {s1['text']} | "
                    f"Actor: {get_actor(ev1)} | Trigger: {get_trigger(ev1)} | "
                    f"Object: {get_object(ev1)}"
                )
                text2 = (
                    f"Sentence: {s2['text']} | "
                    f"Actor: {get_actor(ev2)} | Trigger: {get_trigger(ev2)} | "
                    f"Object: {get_object(ev2)}"
                )

                pred_label, conf = classify_pair(text1, text2)

                if pred_label == "none":
                    continue  # only keep meaningful relations

                all_edges.append({
                    "article_id": aid,
                    "src_event_id": e1["event_id"],
                    "tgt_event_id": e2["event_id"],
                    "relation": pred_label,
                    "confidence": conf,
                    "sentence_distance": dist
                })

    print(f"💾 Saving {len(all_edges)} edges → {OUTPUT_JSON}")
    with open(OUTPUT_JSON, "w") as f:
        json.dump(all_edges, f, indent=2, ensure_ascii=False)

    print("✅ Step 3B complete.")


if __name__ == "__main__":
    apply_classifier()