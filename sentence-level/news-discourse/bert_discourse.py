import os
import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.nn.functional import softmax


# ===============================================================
# CONFIG
# ===============================================================
DATA_PATH = "bias_with_discourse_context.csv"
OUTPUT_SENTENCE = "./sentence_bias_predictions.csv"
OUTPUT_ARTICLE = "./article_bias_predictions.csv"
MODEL_CHECKPOINT = "bert-base-uncased"

# ===============================================================
# LOAD DATA
# ===============================================================
df = pd.read_csv(DATA_PATH)

# Drop missing or empty sentences
df = df.dropna(subset=["text"])
df["label"] = df["has_bias"].astype(int)

# Combine discourse features into input text
df["input_text"] = (
    "Discourse: " + df["discourse_role"].fillna("NONE") +
    " | Prev: " + df["prev_discourse_role"].fillna("NONE") +
    " | Next: " + df["next_discourse_role"].fillna("NONE") +
    " | Article stance: " + df["article_bias"].fillna("UNKNOWN") +
    " | Sentence: " + df["text"].fillna("")
)

print("📊 Dataset preview:")
print(df[["article_id", "article_bias", "discourse_role", "label"]].head())

# Convert to HuggingFace dataset
dataset = Dataset.from_pandas(df[["article_id", "input_text", "label"]])

# ===============================================================
# TOKENIZATION
# ===============================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

def tokenize(batch):
    return tokenizer(batch["input_text"], truncation=True, padding="max_length", max_length=128)

dataset = dataset.map(tokenize, batched=True)

# Split 80/10/10
train_testvalid = dataset.train_test_split(test_size=0.2, seed=42)
test_valid = train_testvalid["test"].train_test_split(test_size=0.5, seed=42)

train_ds = train_testvalid["train"]
val_ds = test_valid["train"]
test_ds = test_valid["test"]

# ===============================================================
# MODEL
# ===============================================================
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_CHECKPOINT,
    num_labels=2
)

# ===============================================================
# METRICS
# ===============================================================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
        "f1": f1_score(labels, preds)
    }

# ===============================================================
# TRAINING SETUP
# ===============================================================
training_args = TrainingArguments(
    output_dir=os.path.expanduser("~/sentence_bias_checkpoints"),
    do_train=True,
    do_eval=True,
    save_total_limit=2,
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    logging_dir="./logs",
    logging_steps=100,
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# ===============================================================
# TRAINING
# ===============================================================
trainer.train()

# ===============================================================
# EVALUATION
# ===============================================================
metrics = trainer.evaluate(test_ds)
print("\n📈 Sentence-level test metrics:")
for k, v in metrics.items():
    print(f"  {k}: {v:.4f}" if isinstance(v, (float, int)) else f"  {k}: {v}")

# ===============================================================
# INFERENCE
# ===============================================================
model.eval()
bias_probs = []

for row in df.itertuples():
    tokens = tokenizer(
        row.input_text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128
    ).to(model.device)   # 👈 ensures tensors go to CUDA or CPU automatically

    with torch.no_grad():
        logits = model(**tokens).logits
        prob = softmax(logits, dim=-1)[0, 1].item()
df["bias_prob"] = bias_probs

# ===============================================================
# ARTICLE-LEVEL AGGREGATION
# ===============================================================
article_scores = df.groupby("article_id")["bias_prob"].mean().reset_index()
article_scores["predicted_article_bias"] = article_scores["bias_prob"].apply(
    lambda x: "biased" if x > 0.55 else "neutral"
)

print("\n🧾 Article-level results:")
print(article_scores.head())

# ===============================================================
# SAVE OUTPUTS
# ===============================================================
df.to_csv(OUTPUT_SENTENCE, index=False)
article_scores.to_csv(OUTPUT_ARTICLE, index=False)

print("\n✅ Sentence-level and article-level predictions saved.")