#!/usr/bin/env python3
# ===============================================================
# STEP 7 — INFERENCE FOR STEP 6 RGAT + ROBERTA MODEL
# Clean output (no biased/unbiased labels)
# Includes ground truth
# ===============================================================

import os
import json
import torch
import torch.nn.functional as F
import networkx as nx
import numpy as np
from transformers import AutoTokenizer, AutoModel

# ============================================================
# CONFIG
# ============================================================
GRAPH_DIR = "erg_graphs"
FEATURES_DIR = "erg_features"

# GRAPH_DIR = "toy_erg_graphs"
# FEATURES_DIR = "toy_erg_features


MODEL_PATH = "rgat_model_step6/best_model_finetuned.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LM_MODEL_NAME = "roberta-base"

HIDDEN_DIM = 64
NUM_HEADS = 4
NUM_LAYERS = 1
DROPOUT = 0.5

RELATION_TYPES = {
    "causal": 0,
    "temporal": 1,
    "coreference": 2,
    "continuation": 3,
    "sequential": 4,
    "causal_discourse": 5,
    "consequence_discourse": 6,
    "belongs_to": 7
}
NUM_RELATIONS = len(RELATION_TYPES)


# ============================================================
# MODEL (same as Step 6)
# ============================================================
class RelationalGATLayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, num_relations, num_heads=4, dropout=0.5):
        super().__init__()
        assert out_dim % num_heads == 0
        self.num_rel = num_relations
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.out_dim = out_dim

        self.W_rel = torch.nn.ModuleList([
            torch.nn.Linear(in_dim, out_dim, bias=False)
            for _ in range(num_relations)
        ])
        self.attn = torch.nn.Parameter(torch.Tensor(num_relations, num_heads, 2 * self.head_dim))
        self.dropout = torch.nn.Dropout(dropout)
        self.leaky = torch.nn.LeakyReLU(0.2)
        torch.nn.init.xavier_uniform_(self.attn)

    def forward(self, x, edge_index, edge_types):
        N = x.size(0)
        out = torch.zeros(N, self.out_dim, device=x.device, dtype=x.dtype)

        for r in range(self.num_rel):
            mask = (edge_types == r)
            if mask.sum() == 0:
                continue

            edges = edge_index[:, mask]
            src, dst = edges[0], edges[1]

            x_rel = self.W_rel[r](x).view(N, self.num_heads, self.head_dim)
            src_feat = x_rel[src]
            dst_feat = x_rel[dst]

            cat = torch.cat([src_feat, dst_feat], dim=-1)
            alpha = self.leaky((cat * self.attn[r]).sum(-1))

            alpha_soft = torch.zeros_like(alpha)
            for n in range(N):
                idx = (dst == n)
                if idx.sum() > 0:
                    alpha_soft[idx] = F.softmax(alpha[idx], dim=0)

            alpha_soft = self.dropout(alpha_soft)
            msg = src_feat * alpha_soft.unsqueeze(-1)
            msg = msg.view(-1, self.out_dim)

            out.index_add_(0, dst, msg)

        return out


class RGAT(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, num_relations, num_heads=4, num_layers=1, dropout=0.5):
        super().__init__()
        self.input_proj = torch.nn.Linear(in_dim, hidden_dim)
        self.layers = torch.nn.ModuleList([
            RelationalGATLayer(hidden_dim, hidden_dim, num_relations, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.norms = torch.nn.ModuleList([torch.nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.dropout = torch.nn.Dropout(dropout)

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim // 2, 2)
        )

    def encode_graph(self, node_features, edge_index, edge_types):
        x = F.relu(self.input_proj(node_features))
        for i, layer in enumerate(self.layers):
            x_new = layer(x, edge_index, edge_types)
            x_new = self.norms[i](x_new)
            x_new = F.relu(x_new)
            x = x + x_new
        return x

    def forward(self, node_features, edge_index, edge_types, sentence_mask):
        x = self.encode_graph(node_features, edge_index, edge_types)
        sent_emb = x[sentence_mask]
        logits = self.classifier(sent_emb)
        return logits


# ============================================================
# LOAD MODEL
# ============================================================
def load_model():
    print(f"Loading model from {MODEL_PATH}...")

    tokenizer = AutoTokenizer.from_pretrained(LM_MODEL_NAME)
    lm = AutoModel.from_pretrained(LM_MODEL_NAME).to(DEVICE)

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    # determine LM_dim
    with torch.no_grad():
        tmp = tokenizer(["hello"], return_tensors="pt")
        tmp = {k: v.to(DEVICE) for k, v in tmp.items()}
        lm_dim = lm(**tmp).last_hidden_state.size(-1)

    # determine static_dim from the input_proj layer
    in_dim = checkpoint['rgat']['input_proj.weight'].shape[1]
    static_dim = in_dim - lm_dim

    print(f"Static_dim={static_dim}, LM_dim={lm_dim}, Total={in_dim}")

    model = RGAT(
        in_dim=in_dim,
        hidden_dim=HIDDEN_DIM,
        num_relations=NUM_RELATIONS,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(DEVICE)

    model.load_state_dict(checkpoint['rgat'])
    lm.load_state_dict(checkpoint['lm'])

    model.eval()
    lm.eval()

    return model, lm, tokenizer, static_dim, lm_dim


# ============================================================
# LOAD ARTICLE (Step 5 outputs)
# ============================================================
def load_article(article_id, expected_feat_dim):
    graph_path = os.path.join(GRAPH_DIR, f"{article_id}.graphml")
    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"No graph for {article_id}")

    G = nx.read_graphml(graph_path)

    sent_pt = os.path.join(FEATURES_DIR, f"{article_id}_sentence.pt")
    ev_pt = os.path.join(FEATURES_DIR, f"{article_id}_event.pt")
    meta = os.path.join(FEATURES_DIR, f"{article_id}_meta.json")

    if not (os.path.exists(sent_pt) and os.path.exists(ev_pt) and os.path.exists(meta)):
        raise FileNotFoundError(f"Missing Step 5 features for {article_id}")

    sentence_feats = torch.load(sent_pt)
    event_feats = torch.load(ev_pt)
    metadata = json.load(open(meta))

    return G, sentence_feats, event_feats, metadata


# ============================================================
# PREPARE DATA
# ============================================================
def prepare_data(G, sent_feats, ev_feats, metadata, static_dim):
    sentence_nodes = sorted(
        [n for n, d in G.nodes(data=True) if d.get("node_type") == "sentence"],
        key=lambda n: int(G.nodes[n].get("sentence_idx", 0))
    )
    event_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "event"]

    node_order = sentence_nodes + event_nodes
    node_to_idx = {n: i for i, n in enumerate(node_order)}

    feats = torch.cat([sent_feats, ev_feats], dim=0)

    edge_index, edge_types = [], []
    for u, v, d in G.edges(data=True):
        rel = d.get("relation")
        if rel in RELATION_TYPES:
            edge_index.append([node_to_idx[u], node_to_idx[v]])
            edge_types.append(RELATION_TYPES[rel])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t()
    edge_types = torch.tensor(edge_types, dtype=torch.long)

    N = len(node_order)
    sentence_mask = torch.zeros(N, dtype=torch.bool)
    texts = []
    sent_idx = []
    ground_truth = []

    for idx, node in enumerate(node_order):
        if G.nodes[node].get("node_type") == "sentence":
            sentence_mask[idx] = True
            texts.append(G.nodes[node].get("text", ""))
            sent_idx.append(idx)

            gt = G.nodes[node].get("has_bias", False)
            if isinstance(gt, str):
                gt = gt.lower() == "true"
            ground_truth.append(gt)

    return {
        "node_features": feats,
        "edge_index": edge_index,
        "edge_types": edge_types,
        "sentence_mask": sentence_mask,
        "sentence_indices": torch.tensor(sent_idx),
        "sentence_texts": texts,
        "ground_truth": ground_truth,
    }


# ============================================================
# SENTENCE ENCODING
# ============================================================
def encode_sentences(tokenizer, lm, texts):
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt"
    )
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    with torch.no_grad():
        out = lm(**enc)
    return out.last_hidden_state[:, 0, :]


# ============================================================
# PREDICT
# ============================================================
def predict_article(article_id, model, lm, tokenizer, static_dim, lm_dim, threshold=0.5):
    print(f"\n=== Predicting {article_id} ===")

    G, sent_feats, ev_feats, meta = load_article(article_id, static_dim)
    data = prepare_data(G, sent_feats, ev_feats, meta, static_dim)

    feats = data["node_features"].to(DEVICE)
    edge = data["edge_index"].to(DEVICE)
    etypes = data["edge_types"].to(DEVICE)
    mask = data["sentence_mask"].to(DEVICE)
    sent_idx = data["sentence_indices"].to(DEVICE)

    texts = data["sentence_texts"]
    lm_emb = encode_sentences(tokenizer, lm, texts)

    N = feats.size(0)
    lm_full = torch.zeros(N, lm_dim, device=DEVICE)
    lm_full[sent_idx] = lm_emb

    combined = torch.cat([feats, lm_full], dim=-1)

    with torch.no_grad():
        logits = model(combined, edge, etypes, mask)
        probs = F.softmax(logits, dim=1)
        scores = probs[:, 1].cpu().numpy()
        preds = (scores > threshold)

    results = []
    for i, (t, pred, score, gt) in enumerate(zip(texts, preds, scores, data["ground_truth"])):
        results.append({
            "sentence_id": i,
            "text": t,
            "probability": float(score),
            "predicted": bool(pred),
            "ground_truth": gt
        })

    return results

# ============================================================
# PUBLIC API FOR STEP 9
# ============================================================
# ============================================================
# Step 9 helper — unified run_inference() interface
# ============================================================


# ============================================================
# Public function used by Step 9
# ============================================================
def run_inference(article_id, threshold=0.5):
    """
    Wrapper so Step 9 can call a single stable function.
    Returns ONLY:
        results (list of sentence dicts)
    """
    # load model once and cache it
    global _MODEL_CACHE
    if "_MODEL_CACHE" not in globals():
        _MODEL_CACHE = load_model()

    model, lm, tok, static_dim, lm_dim = _MODEL_CACHE

    # run prediction
    results = predict_article(
        article_id,
        model,
        lm,
        tok,
        static_dim,
        lm_dim,
        threshold=threshold
    )

    return results

# ============================================================
# MAIN
# ============================================================
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--article_id", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    model, lm, tok, static_dim, lm_dim = load_model()

    results = predict_article(
        args.article_id,
        model, lm, tok,
        static_dim, lm_dim,
        threshold=args.threshold
    )

    print("\n=== RESULTS ===")
    for r in results:
        print(f"- (prob={r['probability']:.3f}) [GT={r['ground_truth']}]")
        print(f"    {r['text'][:100]}...\n")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()