#!/usr/bin/env python3
"""
bert_baseline.py

Fine-tunes a BERT model for sentence-level bias classification
using the consolidated BASIL dataset.

Author: Kayal Bhatia
"""

import os
import json
import random
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
import torch
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm


# ============================================================
# CONFIGURATION
# ============================================================
INPUT_FILE = "../data-preprocessing/basil_consolidated_all.json"
MODEL_NAME = "bert-base-uncased"
OUTPUT_DIR = "./checkpoints/sentence_bias"
SEED = 42
EPOCHS = 3
LR = 2e-5
BATCH_SIZE = 16


# ============================================================
# SEEDING
# ============================================================
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# ============================================================
# LOAD & PREPARE DATA
# ============================================================
def load_basil_sentences(filepath):
    """
    Load consolidated BASIL dataset and flatten into sentence-level samples.
    Each record contains:
      - text (string)
      - label (int): 1 if has_bias == True, else 0
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    sentences = []
    for article in data:
        for sent in article["sentences"]:
            sentences.append(
                {
                    "text": sent["text"],
                    "label": int(sent["has_bias"]),
                    "uuid": article["uuid"],
                }
            )

    print(f"Loaded {len(sentences)} sentences from BASIL")
    return sentences


# ============================================================
# TOKENIZATION
# ============================================================
def tokenize_function(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=128,
    )


# ============================================================
# EVALUATION
# ============================================================
def compute_metrics(eval_pred):
    preds, labels = eval_pred
    preds = np.argmax(preds, axis=1)
    f1 = f1_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate(model, dataloader, device, verbose=True):
    """Manual evaluation loop (optional)"""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    f1 = f1_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    acc = accuracy_score(all_labels, all_preds)

    if verbose:
        print(f"\n=== Sentence-Level Metrics ===")
        print(f"Accuracy:   {acc:.4f}")
        print(f"Precision:  {precision:.4f}")
        print(f"Recall:     {recall:.4f}")
        print(f"F1 Score:   {f1:.4f}")

    return {"f1": f1, "precision": precision, "recall": recall, "accuracy": acc}


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    # Load data
    samples = load_basil_sentences(INPUT_FILE)
    random.shuffle(samples)

    # Split dataset
    split_1 = int(0.8 * len(samples))
    split_2 = int(0.9 * len(samples))
    train_data = samples[:split_1]
    val_data = samples[split_1:split_2]
    test_data = samples[split_2:]

    # Convert to HF Dataset
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_dataset = Dataset.from_list(train_data).map(tokenize_function, batched=True)
    val_dataset = Dataset.from_list(val_data).map(tokenize_function, batched=True)
    test_dataset = Dataset.from_list(test_data).map(tokenize_function, batched=True)

    # Set format for PyTorch
    train_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "label"]
    )
    val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    test_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "label"]
    )

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training args
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        do_train=True,
        do_eval=True,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LR,
        save_total_limit=1,
        logging_dir="./logs",
        report_to=None,
    )
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # Train
    print("\n==============================")
    print("Training BERT on BASIL Dataset")
    print("==============================\n")
    trainer.train()

    # Evaluate
    print("\n--- Validation Results ---")
    val_metrics = trainer.evaluate()
    print(val_metrics)

    print("\n--- Test Results ---")
    test_metrics = trainer.evaluate(eval_dataset=test_dataset)
    print(test_metrics)

    # Save model
    best_model_path = os.path.join(OUTPUT_DIR, "best_sentence_model.pt")
    torch.save(model.state_dict(), best_model_path)
    print(f"\n✓ Saved best model to {best_model_path}")