import os
import json
import pandas as pd

DATASET_PATH = "/Users/nainapanjwani/444/UnbiasedNet/data-preprocessing/basil_consolidated_all.json"
CSV_DIR = "/Users/nainapanjwani/444/UnbiasedNet/sentence-level/news-discourse/Discoure_Profiling_RL_EMNLP21Findings/output_roles"
OUTPUT_PATH = "/Users/nainapanjwani/444/UnbiasedNet/data-preprocessing/dataset_with_discourse.json"

# Load the dataset
with open(DATASET_PATH, "r") as f:
    data = json.load(f)

updated_articles = 0
missing_files = []

for article in data:
    uuid = article["uuid"]

    # Find the corresponding CSV file
    matching_csv = None
    for fname in os.listdir(CSV_DIR):
        if fname.startswith(uuid) and fname.endswith(".csv"):
            matching_csv = os.path.join(CSV_DIR, fname)
            break

    if not matching_csv:
        missing_files.append(uuid)
        continue

    # Load predictions
    df = pd.read_csv(matching_csv)
    predicted_roles = df["predicted_role"].tolist()

    # Safety check: align lengths
    sentences = article["sentences"]
    if len(sentences) != len(predicted_roles):
        print(f"⚠️ Mismatch in sentence count for {uuid}: "
              f"{len(sentences)} vs {len(predicted_roles)} — truncating.")
    n = min(len(sentences), len(predicted_roles))

    # Overwrite discourse_role
    for i in range(n):
        sentences[i]["discourse_role"] = predicted_roles[i]

    updated_articles += 1

# Save updated dataset
with open(OUTPUT_PATH, "w") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"✅ Updated {updated_articles} articles.")
if missing_files:
    print(f"⚠️ Missing CSVs for {len(missing_files)} UUIDs: {missing_files}")