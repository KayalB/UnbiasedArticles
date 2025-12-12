#!/usr/bin/env python3
# ===============================================================
# STEP 10 — Triplet Contrastive Training on Article Embeddings
# ---------------------------------------------------------------
# Uses:
#   • Step 6 RGAT + RoBERTa encoder (frozen)
#   • Sentence-level bias probs as attention features
#   • Triplets:
#       - anchor = center article
#       - positive = left article  (same event, different framing)
#       - negative = right article (same event, opposite framing)
#
# Goal:
#   Learn an article-level embedding space where:
#     d(anchor, positive) < d(anchor, negative)
#   using cosine-based triplet loss.
#
# Output:
#   triplet_step10/triplet_model.pt
#   (projection + attention head, Step 6 remains separate)
# ===============================================================

import os
import json
import random
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np

# Reuse Step 7 helpers (sentence-level encoder & graph utilities)
from step7 import (
    load_model,      # loads Step 6 RGAT + RoBERTa
    load_article,    # (article_id, static_dim) -> G, sent_feats, ev_feats, metadata
    prepare_data,    # (G, sent_feats, ev_feats, metadata, static_dim) -> tensors
    encode_sentences,
    DEVICE,
    GRAPH_DIR,
    FEATURES_DIR
)

# ---------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------
TRIPLET_PATH = "basil_triplets.json"  # <-- change if needed

OUTPUT_DIR = "triplet_step10"
os.makedirs(OUTPUT_DIR, exist_ok=True)

HIDDEN_DIM = 64       # must match Step 6 RGAT hidden_dim
ATTN_DIM   = 32       # attention MLP size
PROJ_DIM   = 64       # final embedding dimension

DROPOUT    = 0.3
LR         = 1e-3
WEIGHT_DECAY = 0.01
NUM_EPOCHS   = 30
PATIENCE     = 5
MARGIN       = 0.3    # triplet margin

BATCH_SIZE   = 1      # each "batch" = one triplet


# ===============================================================
# Dataset
# ===============================================================
class TripletDataset(Dataset):
    """
    Stores triplets: (anchor=center, positive=left, negative=right)
    Uses only triplets where all three article UUIDs exist in FEATURES_DIR/GRAPH_DIR.
    """

    def __init__(self, triplet_json_path):
        with open(triplet_json_path, "r") as f:
            raw = json.load(f)

        self.samples = []

        missing_graph = 0
        missing_feats = 0
        total = 0

        print(f"📦 Loading triplets from {triplet_json_path}...")

        for t in raw:
            total += 1
            try:
                left_id   = t["left"]["uuid"]
                center_id = t["center"]["uuid"]
                right_id  = t["right"]["uuid"]

                # Require graph + features to exist for all 3
                def has_files(aid: str) -> bool:
                    g = os.path.join(GRAPH_DIR, f"{aid}.graphml")
                    s = os.path.join(FEATURES_DIR, f"{aid}_sentence.pt")
                    e = os.path.join(FEATURES_DIR, f"{aid}_event.pt")
                    m = os.path.join(FEATURES_DIR, f"{aid}_meta.json")
                    return all(os.path.exists(p) for p in [g, s, e, m])

                ok_left   = has_files(left_id)
                ok_center = has_files(center_id)
                ok_right  = has_files(right_id)

                if not (ok_left and ok_center and ok_right):
                    # crude diagnostics
                    if not ok_left or not ok_center or not ok_right:
                        # We won't over-log here; just skip
                        pass
                    continue

                self.samples.append({
                    "triplet_uuid": t["triplet_uuid"],
                    "anchor_id": center_id,
                    "pos_id": left_id,
                    "neg_id": right_id,
                    "anchor_ideology": t["center"]["ideology_id"],
                    "pos_ideology": t["left"]["ideology_id"],
                    "neg_ideology": t["right"]["ideology_id"],
                })

            except KeyError as e:
                print(f"⚠️  Bad triplet entry (missing key {e}); skipping.")
                continue

        print(f"✅ Usable triplets: {len(self.samples)} / {total}")

        # ideology distribution (center should mostly be 1)
        counts = Counter([s["anchor_ideology"] for s in self.samples])
        if counts:
            print("\n[Anchor ideology distribution]")
            for k in sorted(counts.keys()):
                name = ["Left", "Center", "Right"][k] if k in [0, 1, 2] else str(k)
                print(f"  {name}: {counts[k]}")
        print()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_triplets(batch):
    """
    Ensure we get a single dict (not a list of dicts) for batch_size=1.
    """
    return batch[0]


# ===============================================================
# Article Encoder Utilities (reuse Step 6/7)
# ===============================================================
def encode_article(article_id, step6_model, lm, tokenizer, static_dim, lm_dim):
    """
    Encode a single article into:
        sent_emb:  [num_sentences, HIDDEN_DIM]
        bias_prob: [num_sentences]
    using your Step 6 RGAT + RoBERTa.
    """
    # Load article data from Step 5 outputs
    G, sent_feats, ev_feats, metadata = load_article(article_id, static_dim)
    data = prepare_data(G, sent_feats, ev_feats, metadata, static_dim)

    feats = data["node_features"].to(DEVICE)
    edge_index = data["edge_index"].to(DEVICE)
    edge_types = data["edge_types"].to(DEVICE)
    sentence_mask = data["sentence_mask"].to(DEVICE)
    sent_idx = data["sentence_indices"].to(DEVICE)
    texts = data["sentence_texts"]

    with torch.no_grad():
        # LM embeddings for sentences
        lm_emb = encode_sentences(tokenizer, lm, texts)

        N = feats.size(0)
        lm_full = torch.zeros(N, lm_emb.size(1), device=DEVICE)
        lm_full[sent_idx] = lm_emb

        combined = torch.cat([feats, lm_full], dim=-1)

        # Node embeddings from RGAT
        node_emb = step6_model.encode_graph(combined, edge_index, edge_types)
        sent_emb = node_emb[sentence_mask]  # [S, HIDDEN_DIM]

        # Sentence-level bias probabilities
        logits = step6_model.classifier(sent_emb)  # [S, 2]
        probs = F.softmax(logits, dim=-1)
        bias_prob = probs[:, 1]  # P(biased)

    return sent_emb, bias_prob


# ===============================================================
# Triplet Article Encoder Model
# ===============================================================
class ArticleTripletEncoder(nn.Module):
    """
    Takes sentence embeddings + bias probabilities, does attention pooling,
    then projects to a low-dimensional normalized embedding.

      Input:
        sent_emb:  [S, H]
        bias_prob: [S]

      Output:
        z:         [1, D] (L2-normalized article embedding)
    """
    def __init__(self, hidden_dim, attn_dim, proj_dim, dropout=0.3):
        super().__init__()
        self.attn_mlp = nn.Sequential(
            nn.Linear(hidden_dim + 1, attn_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
        )
        self.attn_vec = nn.Linear(attn_dim, 1)

        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, proj_dim),
        )

    def forward(self, sent_emb, bias_prob):
        if sent_emb.size(0) == 0:
            raise ValueError("No sentences in article for triplet encoding")

        # Attention over sentences using bias_prob as a feature
        att_in = torch.cat([sent_emb, bias_prob.unsqueeze(-1)], dim=-1)  # [S, H+1]
        u = self.attn_mlp(att_in)                                       # [S, A]
        scores = self.attn_vec(u).squeeze(-1)                           # [S]
        weights = F.softmax(scores, dim=0)                              # [S]

        # Weighted sum → article representation
        article_emb = (weights.unsqueeze(-1) * sent_emb).sum(dim=0, keepdim=True)  # [1, H]

        # Project and normalize
        z = self.proj(article_emb)   # [1, D]
        z = F.normalize(z, p=2, dim=-1)
        return z, weights


# ===============================================================
# Triplet Loss
# ===============================================================
def triplet_cosine_loss(anchor, positive, negative, margin=0.3):
    """
    anchor, positive, negative: [1, D]
    Distance = 1 - cosine_similarity

    L = max(0, d(ap) - d(an) + margin)
    """
    d_ap = 1.0 - F.cosine_similarity(anchor, positive)  # [1]
    d_an = 1.0 - F.cosine_similarity(anchor, negative)  # [1]
    loss = torch.relu(d_ap - d_an + margin)
    return loss.mean()


# ===============================================================
# Training / Evaluation Loops
# ===============================================================
def train_epoch(model, step6_model, lm, tokenizer, static_dim, lm_dim, loader, optimizer):
    model.train()
    step6_model.eval()
    lm.eval()

    total_loss = 0.0
    n_steps = 0

    for batch in loader:
        anchor_id = batch["anchor_id"]
        pos_id    = batch["pos_id"]
        neg_id    = batch["neg_id"]

        try:
            # Encode all three articles
            a_sent, a_bias = encode_article(anchor_id, step6_model, lm, tokenizer, static_dim, lm_dim)
            p_sent, p_bias = encode_article(pos_id,    step6_model, lm, tokenizer, static_dim, lm_dim)
            n_sent, n_bias = encode_article(neg_id,    step6_model, lm, tokenizer, static_dim, lm_dim)

            a_z, _ = model(a_sent, a_bias)
            p_z, _ = model(p_sent, p_bias)
            n_z, _ = model(n_sent, n_bias)

            loss = triplet_cosine_loss(a_z, p_z, n_z, margin=MARGIN)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_steps += 1

        except Exception as e:
            print(f"  ⚠️  Error in triplet ({anchor_id}, {pos_id}, {neg_id}): {e}")
            continue

    if n_steps == 0:
        return 0.0

    return total_loss / n_steps


def eval_epoch(model, step6_model, lm, tokenizer, static_dim, lm_dim, loader):
    """
    We don't have a direct supervised metric here, but we can track:
      • mean loss
      • mean cosine distances (for monitoring)
    """
    model.eval()
    step6_model.eval()
    lm.eval()

    total_loss = 0.0
    n_steps = 0

    d_ap_list = []
    d_an_list = []

    with torch.no_grad():
        for batch in loader:
            anchor_id = batch["anchor_id"]
            pos_id    = batch["pos_id"]
            neg_id    = batch["neg_id"]

            try:
                a_sent, a_bias = encode_article(anchor_id, step6_model, lm, tokenizer, static_dim, lm_dim)
                p_sent, p_bias = encode_article(pos_id,    step6_model, lm, tokenizer, static_dim, lm_dim)
                n_sent, n_bias = encode_article(neg_id,    step6_model, lm, tokenizer, static_dim, lm_dim)

                a_z, _ = model(a_sent, a_bias)
                p_z, _ = model(p_sent, p_bias)
                n_z, _ = model(n_sent, n_bias)

                loss = triplet_cosine_loss(a_z, p_z, n_z, margin=MARGIN)

                d_ap = 1.0 - F.cosine_similarity(a_z, p_z)  # [1]
                d_an = 1.0 - F.cosine_similarity(a_z, n_z)  # [1]

                total_loss += loss.item()
                d_ap_list.append(d_ap.item())
                d_an_list.append(d_an.item())
                n_steps += 1

            except Exception as e:
                print(f"  ⚠️  Error in eval triplet ({anchor_id}, {pos_id}, {neg_id}): {e}")
                continue

    if n_steps == 0:
        return 0.0, 0.0, 0.0

    mean_loss = total_loss / n_steps
    mean_d_ap = float(np.mean(d_ap_list))
    mean_d_an = float(np.mean(d_an_list))

    return mean_loss, mean_d_ap, mean_d_an


# ===============================================================
# MAIN
# ===============================================================
def main():
    print(f"Using device: {DEVICE}\n")

    # 1) Load Step 6 sentence encoder
    print("🔧 Loading Step 6 RGAT + RoBERTa...")
    step6_model, lm, tokenizer, static_dim, lm_dim = load_model()

    # Freeze Step 6 + LM
    for p in step6_model.parameters():
        p.requires_grad = False
    for p in lm.parameters():
        p.requires_grad = False

    print("✅ Step 6 encoder loaded and frozen")
    print(f"   static_dim = {static_dim}, lm_dim = {lm_dim}\n")

    # 2) Load triplets
    dataset = TripletDataset(TRIPLET_PATH)
    if len(dataset) == 0:
        raise RuntimeError("No usable triplets found. Check TRIPLET_PATH and graph/feature files.")

    # simple train/val split
    indices = list(range(len(dataset)))
    random.seed(42)
    random.shuffle(indices)
    split = int(0.8 * len(indices))
    train_idx = indices[:split]
    val_idx   = indices[split:]

    train_subset = torch.utils.data.Subset(dataset, train_idx)
    val_subset   = torch.utils.data.Subset(dataset, val_idx)

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_triplets)
    val_loader   = DataLoader(val_subset,   batch_size=BATCH_SIZE,
                              shuffle=False, collate_fn=collate_triplets)

    print(f"Triplets: Train={len(train_subset)} | Val={len(val_subset)}\n")

    # 3) Triplet article encoder
    model = ArticleTripletEncoder(
        hidden_dim=HIDDEN_DIM,
        attn_dim=ATTN_DIM,
        proj_dim=PROJ_DIM,
        dropout=DROPOUT,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val_loss = float("inf")
    patience_counter = 0

    print("\n" + "="*70)
    print("TRAINING TRIPLET ARTICLE ENCODER (STEP 10)")
    print("="*70 + "\n")

    for epoch in range(1, NUM_EPOCHS + 1):
        tr_loss = train_epoch(
            model, step6_model, lm, tokenizer, static_dim, lm_dim,
            train_loader, optimizer
        )

        val_loss, mean_d_ap, mean_d_an = eval_epoch(
            model, step6_model, lm, tokenizer, static_dim, lm_dim,
            val_loader
        )

        print(f"E{epoch:02d} | TL={tr_loss:.4f} | VL={val_loss:.4f} "
              f"| d_ap={mean_d_ap:.3f} d_an={mean_d_an:.3f}")

        # Early stopping on val loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                {
                    "article_triplet_encoder": model.state_dict(),
                    "hidden_dim": HIDDEN_DIM,
                    "attn_dim": ATTN_DIM,
                    "proj_dim": PROJ_DIM,
                    "margin": MARGIN,
                },
                os.path.join(OUTPUT_DIR, "triplet_model.pt"),
            )
            print("   ✅ Saved new best triplet model")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("\n   ⏹ Early stopping")
                break

    print("\nDone. Best model saved to:", os.path.join(OUTPUT_DIR, "triplet_model.pt"))


if __name__ == "__main__":
    main()