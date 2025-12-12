"""
STEP 9 — Full Test Set Evaluation (Sentence-Level Only)
=======================================================

This script:
  • Loads your Step 7 inference function
  • Runs sentence-level predictions for each test article
  • Aggregates global precision/recall/F1/accuracy
  • Saves to step9_metrics.json
"""

import os
import json
import torch
import numpy as np
from step7 import run_inference
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    confusion_matrix
)

from step7 import run_inference   # <-- Your Step 7 returns ONLY results

TEST_SPLIT_PATH = "test_split.json"
GRAPH_DIR = "erg_graphs"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# Load test IDs
# ============================================================
def load_test_ids():
    if os.path.exists(TEST_SPLIT_PATH):
        print(f"📘 Loading test split from {TEST_SPLIT_PATH}")
        return json.load(open(TEST_SPLIT_PATH))

    from sklearn.model_selection import train_test_split
    ids = [f.replace(".graphml", "") for f in os.listdir(GRAPH_DIR) if f.endswith(".graphml")]

    _, temp = train_test_split(ids, test_size=0.3, random_state=42)
    _, test_ids = train_test_split(temp, test_size=0.5, random_state=42)

    print("⚠ No test_split.json found — using Step 6 default split.")
    return test_ids


# ============================================================
# Evaluation
# ============================================================
def evaluate_full_test_set():
    test_ids = load_test_ids()

    print(f"\n===== STEP 9: Evaluating {len(test_ids)} test articles =====\n")

    all_preds = []
    all_gold  = []

    per_article = {}

    for aid in test_ids:
        try:
            print(f"\n🔍 Evaluating article {aid}...")

            # Step 7 returns ONLY: results (list of dicts)
            results = run_inference(aid)

            preds = []
            golds = []

            # extract per-sentence predictions
            for r in results:
                preds.append(int(r["predicted"]))
                golds.append(int(r["ground_truth"]))

                all_preds.append(int(r["predicted"]))
                all_gold.append(int(r["ground_truth"]))

            # compute article-level metrics
            acc = accuracy_score(golds, preds)
            p   = precision_score(golds, preds, zero_division=0)
            r   = recall_score(golds, preds, zero_division=0)
            f1  = f1_score(golds, preds, zero_division=0)

            per_article[aid] = {
                "accuracy": acc,
                "precision": p,
                "recall": r,
                "f1": f1,
                "num_sentences": len(golds)
            }

            print(f"   ✔ Article F1 = {f1:.3f}")

        except Exception as e:
            print(f"   ✗ ERROR processing {aid}: {e}")
            continue

    # =============================
    # Global metrics
    # =============================
    all_preds = np.array(all_preds)
    all_gold  = np.array(all_gold)

    print("\n===== GLOBAL SENTENCE-LEVEL METRICS =====\n")

    acc = accuracy_score(all_gold, all_preds)
    p   = precision_score(all_gold, all_preds, zero_division=0)
    r   = recall_score(all_gold, all_preds, zero_division=0)
    f1  = f1_score(all_gold, all_preds, zero_division=0)

    print(f"Accuracy : {acc:.3f}")
    print(f"Precision: {p:.3f}")
    print(f"Recall   : {r:.3f}")
    print(f"F1       : {f1:.3f}\n")

    cm = confusion_matrix(all_gold, all_preds)
    print("Confusion Matrix:")
    print(cm)

    # =============================
    # Save output JSON
    # =============================
    out = {
        "global_metrics": {
            "accuracy": acc,
            "precision": p,
            "recall": r,
            "f1": f1,
            "confusion_matrix": cm.tolist()
        },
        "per_article": per_article
    }

    with open("step9_metrics.json", "w") as f:
        json.dump(out, f, indent=2)

    print("\n📁 Saved → step9_metrics.json\n")


# ============================================================
# Entry Point
# ============================================================
if __name__ == "__main__":
    evaluate_full_test_set()