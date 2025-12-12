#!/usr/bin/env python3
"""
bert_baseline_article_source.py

Fine-tunes a BERT model for article-level source classification (Left/Center/Right)
using the consolidated BASIL dataset.

Author: Kayal Bhatia
"""

# ============================================================
# ENVIRONMENT SETUP - MUST BE BEFORE IMPORTS
# ============================================================
import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

import json
import random
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, classification_report
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
import torch
from tqdm import tqdm


# ============================================================
# CONFIGURATION
# ============================================================
INPUT_FILE = "../data-preprocessing/basil_consolidated_all.json"
MODEL_NAME = "bert-base-uncased"
OUTPUT_DIR = "./checkpoints/article_source"
SEED = 42
EPOCHS = 3
LR = 2e-5
BATCH_SIZE = 8
MAX_LENGTH = 512

# Label mapping
LABEL_MAP = {
    "fox": 0,      # Right
    "nyt": 1,      # Left
    "hpo": 2,      # Center/Left-leaning
}
LABEL_NAMES = ["Right (Fox)", "Left (NYT)", "Center (HPO)"]


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
def load_basil_articles(filepath):
    """
    Load consolidated BASIL dataset at article level.
    Each record contains:
      - text (string): concatenated sentences
      - label (int): 0=Fox(Right), 1=NYT(Left), 2=HPO(Center)
      - uuid (string): article identifier
      - source (string): original source name
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Raw data loaded: {len(data)} entries")

    articles = []
    skipped = 0
    
    for idx, article in enumerate(data):
        # Check if article_metadata exists and has source
        if "article_metadata" not in article:
            print(f"Skipping article {idx} (uuid: {article.get('uuid', 'unknown')}): missing 'article_metadata' field")
            skipped += 1
            continue
            
        if "source" not in article["article_metadata"]:
            print(f"Skipping article {idx} (uuid: {article.get('uuid', 'unknown')}): missing 'source' in article_metadata")
            skipped += 1
            continue
        
        # Concatenate all sentences in the article
        if "sentences" not in article or len(article["sentences"]) == 0:
            print(f"Skipping article {idx} (uuid: {article.get('uuid', 'unknown')}): no sentences")
            skipped += 1
            continue
            
        full_text = " ".join([sent["text"] for sent in article["sentences"]])
        
        # Get source from article_metadata and map to label
        source = article["article_metadata"]["source"].lower()
        if source not in LABEL_MAP:
            print(f"Warning: Unknown source '{source}' for article {article['uuid']} - skipping")
            skipped += 1
            continue
            
        articles.append(
            {
                "text": full_text,
                "label": LABEL_MAP[source],
                "uuid": article["uuid"],
                "source": source,
            }
        )

    print(f"\nSuccessfully loaded {len(articles)} articles from BASIL")
    if skipped > 0:
        print(f"Skipped {skipped} articles due to missing fields or unknown sources")
    
    # Print label distribution
    for label_idx, label_name in enumerate(LABEL_NAMES):
        count = sum(1 for a in articles if a["label"] == label_idx)
        print(f"  - {label_name}: {count} ({count/len(articles)*100:.1f}%)")
    
    return articles


# ============================================================
# TOKENIZATION
# ============================================================
def tokenize_function(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )


# ============================================================
# EVALUATION
# ============================================================
def compute_metrics(eval_pred):
    preds, labels = eval_pred
    preds = np.argmax(preds, axis=1)
    
    # Macro-averaged metrics (average across classes)
    f1_macro = f1_score(labels, preds, average='macro')
    precision_macro = precision_score(labels, preds, average='macro')
    recall_macro = recall_score(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    
    return {
        "accuracy": acc,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
    }


def evaluate(model, dataloader, device, verbose=True):
    """Manual evaluation loop with detailed per-class metrics"""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Overall metrics
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    precision_macro = precision_score(all_labels, all_preds, average='macro')
    recall_macro = recall_score(all_labels, all_preds, average='macro')
    acc = accuracy_score(all_labels, all_preds)

    if verbose:
        print(f"\n=== Article-Level Source Classification Metrics ===")
        print(f"Accuracy:         {acc:.4f}")
        print(f"Precision (Macro): {precision_macro:.4f}")
        print(f"Recall (Macro):    {recall_macro:.4f}")
        print(f"F1 Score (Macro):  {f1_macro:.4f}")
        
        print("\n=== Per-Class Performance ===")
        print(classification_report(
            all_labels, 
            all_preds, 
            target_names=LABEL_NAMES,
            digits=4
        ))

    return {
        "f1_macro": f1_macro, 
        "precision_macro": precision_macro, 
        "recall_macro": recall_macro, 
        "accuracy": acc
    }


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("="*60)
    print("BERT Baseline: Article-Level Source Classification")
    print("="*60)
    
    # Load data
    samples = load_basil_articles(INPUT_FILE)
    
    if len(samples) == 0:
        print("\nERROR: No valid articles found!")
        print("Please check:")
        print("  1. The input file path is correct")
        print("  2. The JSON structure matches expected format")
        print("  3. Articles have 'article_metadata.source' and 'sentences' fields")
        exit(1)
    
    random.shuffle(samples)

    # Split dataset by articles (80/10/10)
    split_1 = int(0.8 * len(samples))
    split_2 = int(0.9 * len(samples))
    train_data = samples[:split_1]
    val_data = samples[split_1:split_2]
    test_data = samples[split_2:]
    
    print(f"\nDataset splits:")
    print(f"  - Train: {len(train_data)} articles")
    print(f"  - Val:   {len(val_data)} articles")
    print(f"  - Test:  {len(test_data)} articles")

    # Convert to HF Dataset
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print("\nTokenizing datasets...")
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

    # Load model (3 classes now)
    print("\nLoading BERT model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=3
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")

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
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=100,
        report_to=None,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        dataloader_num_workers=0,  # Avoid multiprocessing issues
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
    print("\n==========================================")
    print("Training BERT on BASIL Dataset")
    print("(Article-Level Source Classification)")
    print("Classes: Left (NYT) | Center (HPO) | Right (Fox)")
    print("==========================================\n")
    trainer.train()

    # Evaluate
    print("\n--- Validation Results ---")
    val_metrics = trainer.evaluate()
    print(val_metrics)

    print("\n--- Test Results ---")
    test_metrics = trainer.evaluate(eval_dataset=test_dataset)
    print(test_metrics)

    # Save model
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    best_model_path = os.path.join(OUTPUT_DIR, "best_article_source_model.pt")
    torch.save(model.state_dict(), best_model_path)
    print(f"\n✓ Saved best model to {best_model_path}")
    
    # Save label mapping
    label_map_path = os.path.join(OUTPUT_DIR, "label_map.json")
    with open(label_map_path, "w") as f:
        json.dump({
            "label_map": LABEL_MAP,
            "label_names": LABEL_NAMES,
            "id2label": {str(i): name for i, name in enumerate(LABEL_NAMES)},
        }, f, indent=2)
    print(f"✓ Saved label mapping to {label_map_path}")
    
    print("\n" + "="*60)
    print("Training completed successfully!")
    print("="*60)