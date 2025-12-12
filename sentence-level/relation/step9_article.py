#!/usr/bin/env python3
# ===============================================================
# STEP 9 (BASIL) — TRIPLET-AWARE IDEOLOGY TRAINING
# ---------------------------------------------------------------
# Uses BASIL triplets (Left, Center, Right) to:
#   • classify article ideology
#   • control topic confounds via triplet loss
#
# Step 6 is FROZEN.
# ===============================================================

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from step7 import (
    load_model,
    load_article,
    prepare_data,
    encode_sentences,
    DEVICE,
    GRAPH_DIR
)

# ===============================================================
# CONFIG
# ===============================================================
TRIPLET_JSON = "basil_triplets_normalized.json"
OUTPUT_DIR   = "basil_triplet_model"
os.makedirs(OUTPUT_DIR, exist_ok=True)

HIDDEN_DIM = 128
ATTN_DIM   = 64
DROPOUT    = 0.5

LR         = 1e-3
EPOCHS     = 30
TRIPLET_LAMBDA = 1.5

LABEL_MAP = {"center": 0, "left": 1, "right": 2}
ID2LABEL  = {0: "Center", 1: "Left", 2: "Right"}

# ===============================================================
# DATASET
# ===============================================================
class BasilTripletDataset(Dataset):
    """
    Expects format:
    {
      "triplet_id": "...",
      "articles": {
        "left":   <uuid>,
        "center": <uuid>,
        "right":  <uuid>
      }
    }
    """
    def __init__(self, triplet_json, graph_dir):
        with open(triplet_json) as f:
            raw = json.load(f)

        available = {
            f.replace(".graphml", "")
            for f in os.listdir(graph_dir)
            if f.endswith(".graphml")
        }

        self.triplets = []

        for item in raw:
            arts = item["articles"]
            valid = {
                k: v for k, v in arts.items()
                if k in LABEL_MAP and v in available
            }
            if len(valid) >= 2:   # allow partial (still useful)
                self.triplets.append(valid)

        print(f"Loaded {len(self.triplets)} BASIL triplets / pairs.")

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        return self.triplets[idx]

# ===============================================================
# MODEL
# ===============================================================
class BasilSiameseNetwork(nn.Module):
    def __init__(self, sent_dim):
        super().__init__()

        self.proj = nn.Linear(sent_dim + 1, HIDDEN_DIM)

        self.attn = nn.Sequential(
            nn.Linear(HIDDEN_DIM, ATTN_DIM),
            nn.Tanh(),
            nn.Linear(ATTN_DIM, 1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(HIDDEN_DIM + 3, HIDDEN_DIM),  # +3 for avg/max/std bias
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM, 3)
        )

    def forward_article(self, sent_emb, bias_prob):
        # Fuse sentence embedding + bias intensity
        x = torch.cat([sent_emb, bias_prob.unsqueeze(-1)], dim=-1)
        x = F.relu(self.proj(x))

        # Attention without arbitrary multiplier
        scores = self.attn(x)
        attn = F.softmax(scores, dim=0)

        article_emb = torch.sum(x * attn, dim=0, keepdim=True)
        article_emb = F.normalize(article_emb, dim=-1)

        # Richer bias statistics
        avg_bias = bias_prob.mean().view(1, 1)
        max_bias = bias_prob.max().view(1, 1)
        std_bias = bias_prob.std().view(1, 1) if len(bias_prob) > 1 else torch.zeros(1, 1, device=bias_prob.device)

        vec = torch.cat([article_emb, avg_bias, max_bias, std_bias], dim=-1)
        logits = self.classifier(vec)

        return logits, article_emb

# ===============================================================
# ENCODER (STEP 6)
# ===============================================================
def encode_article(aid, step6, lm, tok, static_dim, lm_dim):
    G, s_feats, e_feats, meta = load_article(aid, static_dim)
    data = prepare_data(G, s_feats, e_feats, meta, static_dim)

    feats = data["node_features"].to(DEVICE)
    sent_mask = data["sentence_mask"].to(DEVICE)
    sent_idx = data["sentence_indices"].to(DEVICE)
    texts = data["sentence_texts"]

    with torch.no_grad():
        lm_emb = encode_sentences(tok, lm, texts)

        lm_full = torch.zeros(feats.size(0), lm_dim, device=DEVICE)
        lm_full[sent_idx] = lm_emb
        combined = torch.cat([feats, lm_full], dim=-1)

        node_emb = step6.encode_graph(
            combined,
            data["edge_index"].to(DEVICE),
            data["edge_types"].to(DEVICE)
        )

        sent_node_emb = node_emb[sent_mask]
        logits = step6.classifier(sent_node_emb)
        bias_prob = F.softmax(logits, dim=-1)[:, 1]

    return sent_node_emb, bias_prob

# ===============================================================
# TRAINING
# ===============================================================
def train():
    step6, lm, tok, static_dim, lm_dim = load_model()
    step6.eval(); lm.eval()
    for p in step6.parameters(): p.requires_grad = False
    for p in lm.parameters():   p.requires_grad = False

    dataset = BasilTripletDataset(TRIPLET_JSON, GRAPH_DIR)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=lambda x: x[0])

    # infer sentence embedding dim once
    sample_id = next(iter(dataset.triplets[0].values()))
    sent_emb, _ = encode_article(sample_id, step6, lm, tok, static_dim, lm_dim)
    sent_dim = sent_emb.size(-1)

    model = BasilSiameseNetwork(sent_dim).to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='max', factor=0.5, patience=3
    )
    ce_loss = nn.CrossEntropyLoss()
    triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)

    print("\n===== STARTING STEP 9 (BASIL TRIPLETS) =====")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        all_preds, all_labels = [], []
        total_loss = 0.0

        for trip in loader:
            opt.zero_grad()
            
            losses = []
            embeds = {}
            
            for ideology, aid in trip.items():
                y = torch.tensor([LABEL_MAP[ideology]], device=DEVICE)
                
                sent_emb, bias_prob = encode_article(aid, step6, lm, tok, static_dim, lm_dim)
                logits, emb = model.forward_article(sent_emb, bias_prob)
                
                embeds[ideology] = emb
                losses.append(ce_loss(logits, y))
                
                all_preds.append(logits.argmax().item())
                all_labels.append(y.item())
            
            # Sum all CE losses
            batch_loss = sum(losses)
            
            # Add triplet loss if we have full triplet
            if len(embeds) == 3:
                loss_tri = (
                    triplet_loss(embeds["left"], embeds["center"], embeds["right"]) +
                    triplet_loss(embeds["right"], embeds["center"], embeds["left"])
                ) * 0.5 * TRIPLET_LAMBDA
                batch_loss += loss_tri
            
            batch_loss.backward()
            opt.step()
            
            total_loss += batch_loss.item()

        acc = accuracy_score(all_labels, all_preds)
        p, r, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="macro", zero_division=0
        )

        print(
            f"Epoch {epoch:02d} | "
            f"Loss={total_loss/len(loader):.3f} "
            f"Acc={acc:.3f} "
            f"P={p:.3f} "
            f"R={r:.3f} "
            f"F1={f1:.3f}"
        )
        scheduler.step(f1)  

        if epoch % 5 == 0:
            torch.save(model.state_dict(), f"{OUTPUT_DIR}/basil_ep{epoch}.pt")

if __name__ == "__main__":
    train()