import json
import pandas as pd
from tqdm import tqdm

# -------------------------------
# 1️⃣ Load JSON dataset
# -------------------------------
input_path = "./dataset_with_discourse.json"
output_json = "./dataset_with_discourse_context.json"
output_csv = "./dataset_with_discourse_context.csv"

with open(input_path, "r") as f:
    data = json.load(f)

# -------------------------------
# 2️⃣ Flatten into DataFrame
# -------------------------------
rows = []
for article in tqdm(data, desc="Flattening articles"):
    article_id = article.get("id") or article.get("article_id") or article.get("url") or article.get("source", "unknown")
    for i, sent in enumerate(article["sentences"]):
        rows.append({
            "article_id": article_id,
            "sentence_index": i,
            "sentence": sent["text"],
            "discourse_role": sent.get("discourse_role", "NONE")
        })

df = pd.DataFrame(rows)

# -------------------------------
# 3️⃣ Add prev/next discourse roles
# -------------------------------
context_rows = []
for article_id, group in tqdm(df.groupby("article_id"), desc="Adding context"):
    group = group.sort_values("sentence_index")
    prev_roles = ["NONE"] + group["discourse_role"].tolist()[:-1]
    next_roles = group["discourse_role"].tolist()[1:] + ["NONE"]
    group["prev_discourse_role"] = prev_roles
    group["next_discourse_role"] = next_roles
    context_rows.append(group)

df_context = pd.concat(context_rows, ignore_index=True)

# -------------------------------
# 4️⃣ Save as JSON + CSV
# -------------------------------
# Convert back to article-based JSON
articles_out = []
for article_id, group in df_context.groupby("article_id"):
    sentences = []
    for _, row in group.iterrows():
        sentences.append({
            "text": row["sentence"],
            "discourse_role": row["discourse_role"],
            "prev_discourse_role": row["prev_discourse_role"],
            "next_discourse_role": row["next_discourse_role"]
        })
    articles_out.append({
        "article_id": article_id,
        "sentences": sentences
    })

with open(output_json, "w") as f:
    json.dump(articles_out, f, indent=2)

df_context.to_csv(output_csv, index=False)

print(f"✅ Saved with context columns:")
print(f"   JSON → {output_json}")
print(f"   CSV  → {output_csv}")
print(df_context.head(10)[["article_id", "sentence", "discourse_role", "prev_discourse_role", "next_discourse_role"]])