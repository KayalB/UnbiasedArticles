import json
import pandas as pd
from tqdm import tqdm

# -------------------------------
# 1️⃣ Load existing discourse dataset
# -------------------------------
input_path = "/Users/nainapanjwani/444/UnbiasedNet/data-preprocessing/dataset_with_discourse.json"
output_json = "/Users/nainapanjwani/444/UnbiasedNet/sentence-level/news-discourse/discourse_data.json"
output_csv = "/Users/nainapanjwani/444/UnbiasedNet/sentence-level/news-discourse/bias_with_discourse_context.csv"

with open(input_path, "r") as f:
    data = json.load(f)

# -------------------------------
# 2️⃣ Flatten with prev/next discourse roles
# -------------------------------
rows = []
for article in tqdm(data, desc="Processing articles"):
    sentences = article.get("sentences", [])
    # ✅ Prefer UUID for article identity (BASIL)
    article_id = (
        article.get("uuid")
    )
    article_bias = (
        article.get("relative_stance")
        or article.get("article-level-annotations", {}).get("relative_stance")
        or article.get("article_level_annotations", {}).get("relative_stance")
        or "UNKNOWN"
    )

    for i, sent in enumerate(sentences):
        discourse_role = sent.get("discourse_role", "NONE")
        sent_has_bias = sent.get("has_bias") or sent.get("bias") or False

        prev_discourse = (
            sentences[i - 1].get("discourse_role", "NONE") if i > 0 else "NONE"
        )
        next_discourse = (
            sentences[i + 1].get("discourse_role", "NONE") if i < len(sentences) - 1 else "NONE"
        )
        annotations = sent.get("annotations")

        rows.append({
            "article_id": article_id,
            "article_bias": article_bias,
            "text": sent.get("text", ""),
            "discourse_role": discourse_role,
            "prev_discourse_role": prev_discourse,
            "next_discourse_role": next_discourse,
            "has_bias": bool(sent_has_bias),
            "annotations":annotations
        })

# -------------------------------
# 3️⃣ Save to JSON and CSV
# -------------------------------
df = pd.DataFrame(rows)

with open(output_json, "w") as f:
    json.dump(df.to_dict(orient="records"), f, indent=2, ensure_ascii=False)

df.to_csv(output_csv, index=False)

print(f"✅ Saved clean bias + discourse dataset:")
print(f"   JSON → {output_json}")
print(f"   CSV  → {output_csv}")
print(df.head(10))