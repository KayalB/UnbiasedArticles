import json
import re
import random
import numpy as np
from tqdm import tqdm
from collections import defaultdict, Counter
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, EvalPrediction
)
import torch
from sklearn.metrics import f1_score


# ============================================================
# CONSTANTS
# ============================================================
LABEL2ID = {
    "causal": 0,
    "temporal": 1,
    "coreference": 2,
    "continuation": 3,
    "none": 4
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


# ============================================================
# Helpers for REBEL-style events
# ============================================================
def get_trigger(ev): return ev.get("trigger", "").strip().lower()
def get_actor(ev):   return ev.get("actor", "").strip().lower()
def get_object(ev):  return ev.get("object", "").strip().lower()


# ============================================================
# Relation Detection (future-proof logic)
# ============================================================
def detect_relation_type(e1, e2, s1, s2, dist):
    text_pair = (s1["text"] + " " + s2["text"]).lower()

    t1, t2 = get_trigger(e1), get_trigger(e2)
    a1, a2 = get_actor(e1), get_actor(e2)
    o1, o2 = get_object(e1), get_object(e2)

    role1 = s1.get("discourse_role", "NONE")
    role2 = s2.get("discourse_role", "NONE")

    # --- CAUSAL -------------------------------------------------
    causal_cues = r"\b(because|due to|caused|led to|therefore|hence|consequently|resulted)\b"
    if dist <= 2 and re.search(causal_cues, text_pair):
        return "causal", 0.9

    if role1.startswith("Cause") and role2 == "Main":
        return "causal", 0.85

    # --- TEMPORAL ----------------------------------------------
    temporal_cues = r"\b(before|after|then|since|while|meanwhile|subsequently|following)\b"
    if dist <= 2 and re.search(temporal_cues, text_pair):
        return "temporal", 0.85

    if role1 == "Main" and role2 == "Main_Consequence":
        return "temporal", 0.9

    # --- COREFERENCE -------------------------------------------
    if a1 and a1 == a2 and t1 == t2 and dist <= 2:
        return "coreference", 0.95

    # --- CONTINUATION ------------------------------------------
    if a1 and a1 == a2 and t1 != t2 and dist <= 2:
        return "continuation", 0.85

    # Entity overlap
    if ({a1, o1} & {a2, o2}) and dist <= 3:
        return "continuation", 0.7

    # --- DEFAULT ------------------------------------------------
    if dist > 4:
        return "none", 0.95

    return "none", 0.6


# ============================================================
# Generate Training Pairs
# ============================================================
def create_training_pairs(data, max_pairs_per_article=60, min_conf=0.6):
    articles = defaultdict(list)

    # Group sentences by article
    for s in data:
        articles[s["article_id"]].append(s)

    pairs = []
    stats = Counter()

    for aid, sents in tqdm(list(articles.items()), desc="Pairing events"):
        sents = sorted(sents, key=lambda x: x["sentence_idx"])

        # Expand into (event, sentence) units
        events = [
            {"event": ev, "sentence": s}
            for s in sents
            for ev in s.get("events", [])
        ]

        sampled = 0
        for i, ctx1 in enumerate(events):
            if sampled >= max_pairs_per_article:
                break

            for j in range(i + 1, min(i + 8, len(events))):
                e1, s1 = ctx1["event"], ctx1["sentence"]
                e2, s2 = events[j]["event"], events[j]["sentence"]

                dist = abs(s1["sentence_idx"] - s2["sentence_idx"])
                if dist > 5:
                    continue

                label, conf = detect_relation_type(e1, e2, s1, s2, dist)
                if conf < min_conf:
                    continue

                # Cap NONE class so dataset stays balanced
                if label == "none" and stats["none"] > 2 * (stats["causal"] + stats["temporal"] + stats["coreference"] + stats["continuation"]):
                    continue

                # Build text example
                text1 = (
                    f"Sentence: {s1['text']} | "
                    f"Actor: {get_actor(e1)} | Trigger: {get_trigger(e1)} | Object: {get_object(e1)}"
                )
                text2 = (
                    f"Sentence: {s2['text']} | "
                    f"Actor: {get_actor(e2)} | Trigger: {get_trigger(e2)} | Object: {get_object(e2)}"
                )

                pairs.append({
                    "text1": text1,
                    "text2": text2,
                    "label": LABEL2ID[label]
                })

                stats[label] += 1
                sampled += 1

                # Random NONE negative examples
                if random.random() < 0.25:
                    pairs.append({
                        "text1": s1["text"],
                        "text2": s2["text"],
                        "label": LABEL2ID["none"]
                    })
                    stats["none"] += 1

    print("\n📊 Relation label distribution:")
    for k, v in stats.items():
        print(f"  {k:15s}: {v}")

    return pairs


# ============================================================
# Metrics
# ============================================================
def compute_metrics(eval_pred: EvalPrediction):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": (preds == labels).mean(),
        "macro_f1": f1_score(labels, preds, average="macro")
    }


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    print("🔧 Loading Step 2 output...")
    data = json.load(open("discourse_with_events.json"))

    print("📦 Creating weak-labeled training pairs...")
    pairs = create_training_pairs(data)

    # Save updated Step 2 with sentence_idx untouched
    original_data = json.load(open("discourse_with_events.json"))

    if original_data != data:
        print("💾 Saving updated Step 2 data → discourse_with_events_with_idx.json")
        with open("discourse_with_events_with_idx.json", "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    else:
        print("ℹ️ No changes to Step 2 data — Skipping save.")

    # Convert to HF dataset
    print("📏 Converting to HF dataset...")
    hf_data = Dataset.from_list(pairs)
    hf_data = hf_data.shuffle(seed=42)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def preprocess(batch):
        tok = tokenizer(
            batch["text1"],
            batch["text2"],
            truncation=True,
            padding="max_length",
            max_length=256
        )
        tok["labels"] = batch["label"]
        return tok

    hf_data = hf_data.map(preprocess, batched=False)

    # Train/Val split
    split = hf_data.train_test_split(test_size=0.1, seed=42)
    train_ds = split["train"]
    val_ds = split["test"]

    print("🚂 Training edge classifier...")
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=len(LABEL2ID)
    )

    args = TrainingArguments(
        output_dir="edge_classifier",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        num_train_epochs=2,
        weight_decay=0.01,
        logging_steps=20,
        save_steps=500
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics
    )

    trainer.train()

    print("✅ Step 3 training complete.")
    print("💾 Saving final classifier to edge_classifier_final/")
    trainer.save_model("edge_classifier_final")
    tokenizer.save_pretrained("edge_classifier_final")

    print("🎉 Step 3 output ready for Step 3b.")