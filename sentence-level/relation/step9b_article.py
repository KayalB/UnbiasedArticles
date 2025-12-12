#!/usr/bin/env python3
# ===============================================================
# STEP 9b — ARTICLE-LEVEL LEFT vs RIGHT CLASSIFICATION
# ---------------------------------------------------------------
# Assumes article is already biased (from Step 9a).
#
# Input:
#   • Sentence embeddings from Step 6
#   • Sentence-level bias probability (intensity)
#
# Output:
#   • left (0) vs right (1)
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
OUTPUT_DIR = "article_level_step9b_left_right"
os.makedirs(OUTPUT_DIR, exist_ok=True)

HIDDEN_DIM = 64
ATTN_DIM   = 32
DROPOUT    = 0.45

LR           = 1e-3
WEIGHT_DECAY = 0.01
NUM_EPOCHS   = 40
PATIENCE     = 8

NUM_CLASSES = 2  # left / right

LEFT_RIGHT_MAP = {
    "left": 0,
    "liberal": 0,
    "right": 1,
    "conservative": 1
}

# Splits
SPLIT_DIR = "splits_left_right"
os.makedirs(SPLIT_DIR, exist_ok=True)
TRAIN_SPLIT = os.path.join(SPLIT_DIR, "train.json")
VAL_SPLIT   = os.path.join(SPLIT_DIR, "val.json")
TEST_SPLIT  = os.path.join(SPLIT_DIR, "test.json")

# ===============================================================
# Label loader (FILTERS CENTER)
# ===============================================================
def load_left_right_label(aid):
    json_path = os.path.join(FEATURES_DIR, f"{aid}_label.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError

    with open(json_path) as f:
        obj = json.load(f)

    raw = obj["article_bias"].strip().lower()

    if raw not in LEFT_RIGHT_MAP:
        raise ValueError("Not left/right")

    return LEFT_RIGHT_MAP[raw]

# ===============================================================
# Dataset
# ===============================================================
class ArticleDataset(Dataset):
    def __init__(self, ids):
        self.ids = []
        self.labels = {}

        for aid in ids:
            try:
                label = load_left_right_label(aid)
                self.ids.append(aid)
                self.labels[aid] = label
            except:
                continue

        print(f"Loaded {len(self.ids)} LEFT/RIGHT articles.")
        print("Label distribution:", Counter(self.labels.values()))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        aid = self.ids[idx]
        return {"article_id": aid, "label": self.labels[aid]}

# ===============================================================
# Model (IDENTICAL TO STEP 9a, just 2-way output)
# ===============================================================
class ArticleLevelLeftRight(nn.Module):
    def __init__(self, hidden_dim, attn_dim, dropout):
        super().__init__()

        self.attn_proj = nn.Sequential(
            nn.Linear(hidden_dim + 1, attn_dim),
            nn.Tanh(),
            nn.Dropout(dropout)
        )
        self.attn_vec = nn.Linear(attn_dim, 1)

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
        # Scale bias prob to avoid domination
        bias_prob_scaled = bias_prob * 0.35

        att_in = torch.cat([sent_emb, bias_prob_scaled.unsqueeze(-1)], dim=-1)
        u = self.attn_proj(att_in)

        scores = self.attn_vec(u).squeeze(-1) / 1.5
        attn = F.softmax(scores, dim=0)

        article_emb = (attn.unsqueeze(-1) * sent_emb).sum(dim=0, keepdim=True)

        bias_ratio = bias_prob.mean().view(1, 1)
        sent_count = torch.tensor([[sent_emb.size(0)]], device=sent_emb.device).float()

        enhanced = torch.cat([article_emb, bias_ratio, sent_count], dim=-1)
        logits = self.classifier(enhanced)

        return logits, attn

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
# Train / Eval
# ===============================================================
def run_epoch(model, step6, lm, tok, static, lm_dim, loader, opt=None, crit=None):
    train = opt is not None
    model.train() if train else model.eval()

    losses, preds, labels = [], [], []

    for batch in loader:
        aid = batch["article_id"]
        y = torch.tensor([batch["label"]], dtype=torch.long, device=DEVICE)

        try:
            sent_emb, bias_prob = encode_article(aid, step6, lm, tok, static, lm_dim)
            logits, _ = model(sent_emb, bias_prob)
            loss = crit(logits, y)

            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()

            losses.append(loss.item())
            preds.append(torch.argmax(logits).item())
            labels.append(y.item())
        except:
            continue

    if not preds:
        return 0, 0, 0, 0

    p, r, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )

    return np.mean(losses), p, r, f1

# ===============================================================
# MAIN
# ===============================================================
def main():
    print(f"Using device: {DEVICE}")

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
        # We need a new split because 9a had "center" included, but 9b excludes it
        # Step 1: Filter raw IDs to only those that are Left or Right
        valid_ids = []
        for aid in article_ids:
            try:
                load_left_right_label(aid)
                valid_ids.append(aid)
            except:
                continue
        
        train_ids, tmp = train_test_split(valid_ids, test_size=0.3, random_state=42)
        val_ids, test_ids = train_test_split(tmp, test_size=0.5, random_state=42)
        
        json.dump(train_ids, open(TRAIN_SPLIT, "w"), indent=2)
        json.dump(val_ids,   open(VAL_SPLIT, "w"), indent=2)
        json.dump(test_ids,  open(TEST_SPLIT, "w"), indent=2)

    train_loader = DataLoader(ArticleDataset(train_ids), batch_size=1, shuffle=True)
    val_loader   = DataLoader(ArticleDataset(val_ids), batch_size=1)
    test_loader  = DataLoader(ArticleDataset(test_ids), batch_size=1)

    step6, lm, tok, static_dim, lm_dim = load_model()
    for p in step6.parameters(): p.requires_grad = False
    for p in lm.parameters():   p.requires_grad = False

    model = ArticleLevelLeftRight(HIDDEN_DIM, ATTN_DIM, DROPOUT).to(DEVICE)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_f1, patience = -1, 0
    print("===== TRAINING (LEFT vs RIGHT) =====")

    for epoch in range(1, NUM_EPOCHS + 1):
        tr = run_epoch(model, step6, lm, tok, static_dim, lm_dim, train_loader, opt, crit)
        va = run_epoch(model, step6, lm, tok, static_dim, lm_dim, val_loader, None, crit)

        print(f"E{epoch:02d} | Train F1={tr[3]:.3f} | Val F1={va[3]:.3f}")

        if va[3] > best_f1:
            best_f1 = va[3]
            patience = 0
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best.pt"))
        else:
            patience += 1
            if patience >= PATIENCE:
                break

    print("\n===== TESTING =====")
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "best.pt"), map_location=DEVICE))
    _, p, r, f1 = run_epoch(model, step6, lm, tok, static_dim, lm_dim, test_loader, None, crit)

    print(f"Precision: {p:.3f}")
    print(f"Recall:    {r:.3f}")
    print(f"F1:        {f1:.3f}")

if __name__ == "__main__":
    main()