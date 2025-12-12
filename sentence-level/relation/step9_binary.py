csci 444 left

- [ ] making it center, left, right
- [ ] adding baseline to article level

#!/usr/bin/env python3
# ===============================================================
# STEP 9 — IMPROVED ARTICLE-LEVEL BINARY BIAS CLASSIFICATION
# ---------------------------------------------------------------
# New Features:
#   • Downscaled bias_prob to prevent attention domination
#   • Bias ratio (sentence-level bias density)
#   • Sentence count feature (helps center vs biased)
#   • Temperature-scaled attention (more stable)
#   • Higher dropout (reduce overfitting)
#   • Class weighting to improve unbiased recall
#
# Output:
#        unbiased (0) vs biased (1)
# ===============================================================

import os
import json
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

# Step 7 utilities
from step7 import (
    load_model,
    load_article,
    prepare_data,
    encode_sentences,
    DEVICE,
    GRAPH_DIR,
    FEATURES_DIR
)

# ===============================================================
# CONFIG
# ===============================================================
OUTPUT_DIR = "article_level_step9_binary_improved"
os.makedirs(OUTPUT_DIR, exist_ok=True)

HIDDEN_DIM = 64
ATTN_DIM   = 32
DROPOUT    = 0.45   # HIGHER DROPOUT

LR           = 1e-3
WEIGHT_DECAY = 0.01
NUM_EPOCHS   = 40
PATIENCE     = 8

NUM_CLASSES = 2

# Label files
BINARY_MAP = {
    "center": 0,
    "left": 1,
    "right": 1
}

# Splits
SPLIT_DIR = "splits_binary_improved"
os.makedirs(SPLIT_DIR, exist_ok=True)
TRAIN_SPLIT = os.path.join(SPLIT_DIR, "train.json")
VAL_SPLIT   = os.path.join(SPLIT_DIR, "val.json")
TEST_SPLIT  = os.path.join(SPLIT_DIR, "test.json")

# ===============================================================
# Label loader
# ===============================================================
def load_binary_label(aid):
    json_path = os.path.join(FEATURES_DIR, f"{aid}_label.json")

    if os.path.exists(json_path):
        with open(json_path) as f:
            obj = json.load(f)
        raw = obj["article_bias"].strip().lower()
        return BINARY_MAP[raw]

    raise FileNotFoundError(f"No label for article {aid}")

# ===============================================================
# Dataset
# ===============================================================
class ArticleDataset(Dataset):
    def __init__(self, ids):
        self.ids = []
        self.labels = {}
        for aid in ids:
            try:
                label = load_binary_label(aid)
                self.ids.append(aid)
                self.labels[aid] = label
            except:
                continue

        print(f"Loaded {len(self.ids)} articles.")
        dist = Counter(self.labels.values())
        print("Label dist:", dist)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        aid = self.ids[idx]
        return {"article_id": aid, "label": self.labels[aid]}

# ===============================================================
# Model: Hierarchical Attention + Article Features
# ===============================================================
class ArticleLevelImproved(nn.Module):
    """
    Input per sentence:
      - sentence embedding (H)
      - scaled bias_prob (1)

    Global article-level features:
      - bias_ratio (1)
      - sentence_count (1)

    Output:
      - logits for {unbiased, biased}
    """
    def __init__(self, hidden_dim, attn_dim, dropout):
        super().__init__()

        # attention on [H + 1]
        self.attn_proj = nn.Sequential(
            nn.Linear(hidden_dim + 1, attn_dim),
            nn.Tanh(),
            nn.Dropout(dropout)
        )
        self.attn_vec = nn.Linear(attn_dim, 1)

        # classifier input = hidden_dim + 2 metadata features
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)
        )

    def forward(self, sent_emb, bias_prob):
        # -------- 1. Scale bias probability -----------
        bias_prob = bias_prob * 0.35   # reduces domination

        # -------- 2. Compute attention scores ----------
        att_in = torch.cat([sent_emb, bias_prob.unsqueeze(-1)], dim=-1)
        u = self.attn_proj(att_in)

        # stabilize with temperature
        scores = self.attn_vec(u).squeeze(-1)
        scores = scores / 1.5

        attn = F.softmax(scores, dim=0)

        # -------- 3. Weighted pooling -----------------
        article_emb = (attn.unsqueeze(-1) * sent_emb).sum(dim=0, keepdim=True)

        # -------- 4. Add global article stats ----------
        bias_ratio = torch.mean(bias_prob).unsqueeze(0).unsqueeze(0)   # [1,1]
        sent_count = torch.tensor([[sent_emb.size(0)]], dtype=torch.float, device=DEVICE)

        enhanced = torch.cat([article_emb, bias_ratio, sent_count], dim=-1)

        # -------- 5. Classify --------------------------
        logits = self.classifier(enhanced)
        return logits, attn, bias_ratio.item(), sent_emb.size(0)

# ===============================================================
# Encoding
# ===============================================================
def encode_article(aid, step6, lm, tokenizer, static_dim, lm_dim):
    G, s_feats, e_feats, meta = load_article(aid, static_dim)
    data = prepare_data(G, s_feats, e_feats, meta, static_dim)

    feats      = data["node_features"].to(DEVICE)
    edge_index = data["edge_index"].to(DEVICE)
    edge_types = data["edge_types"].to(DEVICE)
    sent_mask  = data["sentence_mask"].to(DEVICE)
    sent_idx   = data["sentence_indices"].to(DEVICE)
    texts      = data["sentence_texts"]

    with torch.no_grad():
        lm_emb = encode_sentences(tokenizer, lm, texts)

        N = feats.size(0)
        lm_full = torch.zeros(N, lm_dim, device=DEVICE)
        lm_full[sent_idx] = lm_emb

        combined = torch.cat([feats, lm_full], dim=-1)

        node_emb = step6.encode_graph(combined, edge_index, edge_types)
        sent_emb = node_emb[sent_mask]

        logits = step6.classifier(sent_emb)
        probs = F.softmax(logits, dim=-1)
        bias_prob = probs[:, 1]

    return sent_emb, bias_prob

# ===============================================================
# Training / Eval
# ===============================================================
def train_epoch(model, step6, lm, tok, static, lm_dim, loader, opt, crit):
    model.train()
    losses = []
    preds, labels = [], []

    for batch in loader:
        aid = batch["article_id"]
        y = torch.tensor([batch["label"]], dtype=torch.long, device=DEVICE)

        try:
            sent_emb, bias_prob = encode_article(aid, step6, lm, tok, static, lm_dim)
            logits, attn, _, _ = model(sent_emb, bias_prob)

            opt.zero_grad()
            loss = crit(logits, y)
            loss.backward()
            opt.step()

            losses.append(loss.item())
            preds.append(torch.argmax(logits).item())
            labels.append(y.item())
        except Exception as e:
            print(f"⚠ Error encoding {aid}: {e}")

    if not preds:
        return 0,0,0,0

    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    return np.mean(losses), p, r, f1

def eval_epoch(model, step6, lm, tok, static, lm_dim, loader, crit, save=False):
    model.eval()
    losses = []
    preds, labels = [], []
    saves = []

    with torch.no_grad():
        for batch in loader:
            aid = batch["article_id"]
            y = torch.tensor([batch["label"]], dtype=torch.long, device=DEVICE)

            try:
                sent_emb, bias_prob = encode_article(aid, step6, lm, tok, static, lm_dim)
                logits, attn, bias_ratio, count = model(sent_emb, bias_prob)

                losses.append(crit(logits, y).item())
                pred = torch.argmax(logits).item()

                preds.append(pred)
                labels.append(y.item())

                if save:
                    saves.append({
                        "article_id": aid,
                        "attention": attn.cpu().tolist(),
                        "prediction": pred,
                        "label": y.item(),
                        "bias_ratio": float(bias_ratio),
                        "sentence_count": int(count)
                    })
            except Exception as e:
                print(f"⚠ Error encoding {aid}: {e}")

    if not preds:
        return 0,0,0,0,[]

    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    return np.mean(losses), p, r, f1, saves

# ===============================================================
# MAIN
# ===============================================================
def main():
    print(f"Using device: {DEVICE}")

    # -------- Load splits --------
    article_ids = [
        f.replace(".graphml", "")
        for f in os.listdir(GRAPH_DIR)
        if f.endswith(".graphml")
    ]

    if all(os.path.exists(p) for p in [TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT]):
        train_ids = json.load(open(TRAIN_SPLIT))
        val_ids   = json.load(open(VAL_SPLIT))
        test_ids  = json.load(open(TEST_SPLIT))
    else:
        train_ids, tmp = train_test_split(article_ids, test_size=0.3, random_state=42)
        val_ids, test_ids = train_test_split(tmp, test_size=0.5, random_state=42)
        json.dump(train_ids, open(TRAIN_SPLIT,"w"), indent=2)
        json.dump(val_ids,   open(VAL_SPLIT,"w"), indent=2)
        json.dump(test_ids,  open(TEST_SPLIT,"w"), indent=2)

    # -------- Datasets --------
    train_ds = ArticleDataset(train_ids)
    val_ds   = ArticleDataset(val_ids)
    test_ds  = ArticleDataset(test_ids)

    collate = lambda b: b[0]
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=collate)
    val_loader   = DataLoader(val_ds,  batch_size=1, shuffle=False, collate_fn=collate)
    test_loader  = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate)

    # -------- Step 6 model --------
    step6, lm, tok, static_dim, lm_dim = load_model()
    for p in step6.parameters(): p.requires_grad = False
    for p in lm.parameters():   p.requires_grad = False

    # -------- Model --------
    model = ArticleLevelImproved(HIDDEN_DIM, ATTN_DIM, DROPOUT).to(DEVICE)

    # -------- Class weights (improve unbiased recall) --------
    class_weights = torch.tensor([1.4, 1.0], device=DEVICE)
    crit = nn.CrossEntropyLoss(weight=class_weights)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # -------- Train --------
    best_f1 = -1
    patience = 0

    print("===== TRAINING (Improved Binary Bias) =====")
    for epoch in range(1, NUM_EPOCHS + 1):
        tr_loss, tr_p, tr_r, tr_f1 = train_epoch(
            model, step6, lm, tok, static_dim, lm_dim, train_loader, opt, crit
        )
        val_loss, val_p, val_r, val_f1, _ = eval_epoch(
            model, step6, lm, tok, static_dim, lm_dim, val_loader, crit
        )

        print(
            f"E{epoch:02d} | TL={tr_loss:.4f} TP={tr_p:.3f} TR={tr_r:.3f} TF1={tr_f1:.3f} | "
            f"VL={val_loss:.4f} VP={val_p:.3f} VR={val_r:.3f} VF1={val_f1:.3f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            patience = 0
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best.pt"))
        else:
            patience += 1
            if patience >= PATIENCE:
                print("⏹ Early stopping")
                break

    # -------- Test --------
    print("\n===== TESTING =====")
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "best.pt"), map_location=DEVICE))

    test_loss, test_p, test_r, test_f1, saves = eval_epoch(
        model, step6, lm, tok, static_dim, lm_dim, test_loader, crit, save=True
    )

    print(f"Loss:      {test_loss:.4f}")
    print(f"Precision: {test_p:.3f}")
    print(f"Recall:    {test_r:.3f}")
    print(f"F1:        {test_f1:.3f}")

    preds = [x["prediction"] for x in saves]
    labels = [x["label"] for x in saves]
    cm = confusion_matrix(labels, preds)
    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(
        classification_report(
            labels,
            preds,
            target_names=["unbiased", "biased"],
            digits=3
        )
    )

    json.dump(saves, open(os.path.join(OUTPUT_DIR,"attention.json"), "w"), indent=2)
    print("\nSaved attention.json")


if __name__ == "__main__":
    main()