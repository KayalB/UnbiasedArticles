#!/usr/bin/env python3
# ===============================================================
# BASIL GLOBAL DATASET STATISTICS
# ===============================================================
# Prints:
#   • total biased vs unbiased sentences
#   • total number of articles
#   • relative_stance distribution
#   • total number of unique triplets
# ===============================================================

import json
from collections import Counter

INPUT_FILE = "basil_consolidated_all.json"

# ---------------------------------------------------------------
# Load data
# ---------------------------------------------------------------
with open(INPUT_FILE) as f:
    data = json.load(f)

# ---------------------------------------------------------------
# Counters
# ---------------------------------------------------------------
total_articles = len(data)

biased_sentences = 0
unbiased_sentences = 0

stance_counter = Counter()
triplet_ids = set()

# ---------------------------------------------------------------
# Iterate
# ---------------------------------------------------------------
for article in data:
    # collect triplet ids
    triplet_id = article.get("triplet_uuid")
    if triplet_id:
        triplet_ids.add(triplet_id)

    # article-level stance
    stance = article.get("article_level_annotations", {}).get("relative_stance")
    if stance:
        stance_counter[stance] += 1

    # sentence-level bias
    for sent in article.get("sentences", []):
        if sent.get("has_bias", False):
            biased_sentences += 1
        else:
            unbiased_sentences += 1

# ---------------------------------------------------------------
# Print results
# ---------------------------------------------------------------
print("\n===== BASIL DATASET GLOBAL STATS =====\n")

print(f"Total articles: {total_articles}")
print(f"Total triplets: {len(triplet_ids)}\n")

print("Sentence-level counts:")
print(f"  Biased sentences:   {biased_sentences}")
print(f"  Unbiased sentences: {unbiased_sentences}")
print(f"  Total sentences:    {biased_sentences + unbiased_sentences}\n")

print("Article relative_stance distribution:")
for stance, count in stance_counter.items():
    print(f"  {stance}: {count}")

print("\n====================================\n")