import json
import pandas as pd
from tqdm import tqdm
import os

# -------------------------------
# 1️⃣ Paths
# -------------------------------
input_path = "./dataset_with_discourse.json"
output_path = "./dataset_with_discourse.json"  # overwrite existing
backup_path = "./dataset_with_discourse_backup.json"
output_csv = "./dataset_with_discourse.csv"

# -------------------------------
# 2️⃣ Load JSON
# -------------------------------
with open(input_path, "r") as f:
    data = json.load(f)

# -------------------------------
# 3️⃣ Make a safe backup
# -------------------------------
if not os.path.exists(backup_path):
    with open(backup_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"🧾 Backup saved → {backup_path}")

# -------------------------------
# 4️⃣ Add prev/next discourse roles
# -------------------------------
for article in tqdm(data, desc="Adding prev/next discourse roles"):
    sentences = article.get("sentences", [])
    for i, sent in enumerate(sentences):
        prev_discourse = (
            sentences[i - 1].get("discourse_role", "NONE") if i > 0 else "NONE"
        )
        next_discourse = (
            sentences[i + 1].get("discourse_role", "NONE") if i < len(sentences) - 1 else "NONE"
        )
        sent["prev_discourse_role"] = prev_discourse
        sent["next_discourse_role"] = next_discourse

# -------------------------------
# 5️⃣ Save JSON (overwrite)
# -------------------------------
with open(output_path, "w") as f:
    json.dump(data, f, indent=2)

# -------------------------------
# 6️⃣ Also save a CSV version for sanity check
# -------------------------------
rows = []
for article in data:
    article_id = (
        article.get("id")
        or article.get("article_id")
        or article.get("url")
        or article.get("title")
        or article.get("source", "unknown")
    )
    for s in article.get("sentences", []):
        rows.append({
            "article_id": article_id,
            "text": s.get("text"),
            "discourse_role": s.get("discourse_role"),
            "prev_discourse_role": s.get("prev_discourse_role"),
            "next_discourse_role": s.get("next_discourse_role"),
            "has_bias": s.get("has_bias") or s.get("bias")
        })

df = pd.DataFrame(rows)
df.to_csv(output_csv, index=False)

print("✅ Added prev/next discourse roles to dataset_with_discourse.json")
print(f"💾 JSON updated: {output_path}")
print(f"📄 CSV snapshot: {output_csv}")
print(df.head(10))