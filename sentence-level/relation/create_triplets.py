import json
from collections import defaultdict
import os

# ==============================
# CONFIG
# ==============================
HERE = os.path.dirname(os.path.abspath(__file__))
INPUT_JSON = os.path.join(HERE, "..", "..", "data-preprocessing", "basil_consolidated_all.json")
OUTPUT_JSON = "basil_triplets_normalized.json"

# Normalize ideology labels
STANCE_MAP = {
    "liberal": "left",
    "conservative": "right",
    "center": "center",
    "left": "left",
    "right": "right"
}

# ==============================
# LOAD DATA
# ==============================
with open(INPUT_JSON, "r") as f:
    articles = json.load(f)

# ==============================
# GROUP BY TRIPLET UUID
# ==============================
groups = defaultdict(dict)

for art in articles:
    triplet_id = art.get("triplet_uuid")
    article_id = art.get("uuid")

    stance_raw = art.get("article_level_annotations", {}).get("relative_stance", "center")
    stance_raw = stance_raw.strip().lower()

    if stance_raw not in STANCE_MAP:
        continue

    stance = STANCE_MAP[stance_raw]

    if triplet_id and article_id:
        groups[triplet_id][stance] = article_id

# ==============================
# BUILD FINAL TRIPLET FILE
# ==============================
triplets = []

for tid, members in groups.items():
    # Keep only if at least 2 ideologies exist
    if len(members) >= 2:
        triplets.append({
            "triplet_id": tid,
            "articles": members
        })

# ==============================
# SAVE
# ==============================
with open(OUTPUT_JSON, "w") as f:
    json.dump(triplets, f, indent=2)

print(f"Saved {len(triplets)} triplets to {OUTPUT_JSON}")

# Optional: stats
from collections import Counter
counts = Counter(len(t["articles"]) for t in triplets)
print("Triplet sizes:", dict(counts))
