import json
import csv
import os
from collections import defaultdict

input_path = "../data-preprocessing/basil_consolidated_all.json"
output_file = "baseline_dataset.csv"

with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Count articles per source to create sequential IDs
source_counter = defaultdict(int)

# Create a CSV with the format: article_id, article_has_bias, sentence_text, sentence_has_bias
with open(output_file, "w", newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["article_id", "article_has_bias", "sentence_text", "sentence_has_bias"])

    for article in data:
        # Get source and create simple article_id like "nyt_001"
        source = article["article_metadata"]["source"].lower()
        source_counter[source] += 1
        article_id = f"{source}_{source_counter[source]:03d}"

        sentences = article["sentences"]

        # Compute article-level label (1 if any sentence is biased)
        article_has_bias = 1 if any(s["has_bias"] for s in sentences) else 0

        for s in sentences:
            sentence_text = s["text"].replace("\n", " ").replace("\r", " ").strip()
            sentence_has_bias = 1 if s["has_bias"] else 0

            writer.writerow([
                article_id,
                article_has_bias,
                sentence_text,
                sentence_has_bias
            ])

print(f"✓ Created {output_file}")
print(f"✓ Total articles: {sum(source_counter.values())}")
print(f"  - NYT: {source_counter['nyt']}")
print(f"  - Fox: {source_counter['fox']}")
print(f"  - HPO: {source_counter['hpo']}")
