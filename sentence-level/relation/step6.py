# ===============================================================
# STEP 6 — RGAT + Fine-tuned RoBERTa (sentence-level bias)
# Adapted for your Step 2-5 pipeline
# ===============================================================

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import networkx as nx
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix
)

from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

# ============================================================
# CONFIG
# ============================================================
GRAPH_DIR = "erg_graphs"
FEATURES_DIR = "erg_features"
OUTPUT_DIR = "rgat_model_step6"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LM_MODEL_NAME = "roberta-base"

HIDDEN_DIM = 64
NUM_HEADS = 4
NUM_LAYERS = 1
DROPOUT = 0.5
LEARNING_RATE = 5e-4
LM_LEARNING_RATE = 1e-5
NUM_EPOCHS = 40
PATIENCE = 10
WARMUP_STEPS_RATIO = 0.1
ACCUMULATION_STEPS = 4

RELATION_TYPES = {
    "causal": 0,
    "temporal": 1,
    "coreference": 2,
    "continuation": 3,
    "sequential": 4,  # sentence-to-sentence
    "causal_discourse": 5,
    "consequence_discourse": 6,
    "belongs_to": 7  # event-to-sentence
}
NUM_RELATIONS = len(RELATION_TYPES)

def collate_single(batch):
    return batch[0]


# ============================================================
# FOCAL LOSS
# ============================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=1.5, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss)
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# ============================================================
# DATASET (adapted to your Step 5 output)
# ============================================================
class ERGDataset(Dataset):
    def __init__(self, article_ids):
        self.data = []
        
        print("Loading graphs and features...")
        
        skipped = {"no_graph": 0, "no_features": 0, "no_edges": 0, "no_sentences": 0}

        for aid in tqdm(article_ids):
            try:
                graph_path = os.path.join(GRAPH_DIR, f"{aid}.graphml")
                if not os.path.exists(graph_path):
                    skipped["no_graph"] += 1
                    continue
                G = nx.read_graphml(graph_path)

                # Load your Step 5 features (they're .pt files, not .json)
                sent_feat_path = os.path.join(FEATURES_DIR, f"{aid}_sentence.pt")
                event_feat_path = os.path.join(FEATURES_DIR, f"{aid}_event.pt")
                meta_path = os.path.join(FEATURES_DIR, f"{aid}_meta.json")
                label_path = os.path.join(FEATURES_DIR, f"{aid}_label.pt")

                if not all(os.path.exists(p) for p in [sent_feat_path, event_feat_path, meta_path]):
                    skipped["no_features"] += 1
                    continue

                # Load features as tensors
                sent_feats_tensor = torch.load(sent_feat_path)  # [num_sents, feat_dim]
                event_feats_tensor = torch.load(event_feat_path)  # [num_events, feat_dim]
                metadata = json.load(open(meta_path))

                if os.path.exists(label_path):
                    article_label = torch.load(label_path).item()
                else:
                    article_label = 1  # default to center

                # Get sentence and event nodes in order
                sentence_nodes = sorted(
                    [n for n, d in G.nodes(data=True) if d.get("node_type") == "sentence"],
                    key=lambda n: int(G.nodes[n].get("sentence_idx", 0))
                )
                event_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "event"]
                
                if len(sentence_nodes) == 0:
                    skipped["no_sentences"] += 1
                    continue

                # Build node order: sentences first, then events
                node_order = sentence_nodes + event_nodes
                node_to_idx = {node: i for i, node in enumerate(node_order)}

                # Stack features in same order
                if len(event_nodes) > 0:
                    feats = torch.cat([sent_feats_tensor, event_feats_tensor], dim=0)
                else:
                    feats = sent_feats_tensor

                # Build edges
                edge_index = []
                edge_types = []
                
                for u, v, d in G.edges(data=True):
                    rel = d.get("relation", "")
                    if rel not in RELATION_TYPES:
                        continue
                    if u not in node_to_idx or v not in node_to_idx:
                        continue
                    
                    edge_index.append([node_to_idx[u], node_to_idx[v]])
                    edge_types.append(RELATION_TYPES[rel])

                if len(edge_index) == 0:
                    skipped["no_edges"] += 1
                    continue

                edge_index = torch.tensor(edge_index, dtype=torch.long).t()
                edge_types = torch.tensor(edge_types, dtype=torch.long)

                # Create sentence mask and labels
                N = len(node_order)
                labels = torch.zeros(N, dtype=torch.long)
                sentence_mask = torch.zeros(N, dtype=torch.bool)

                sent_texts = []
                sent_idx = []

                for i, node in enumerate(node_order):
                    node_data = G.nodes[node]
                    if node_data.get("node_type") == "sentence":
                        sentence_mask[i] = True
                        # Get bias label from graph
                        has_bias = node_data.get("has_bias", "False")
                        if isinstance(has_bias, str):
                            has_bias = has_bias.lower() == "true"
                        labels[i] = 1 if has_bias else 0
                        
                        sent_texts.append(node_data.get("text", ""))
                        sent_idx.append(i)

                if sentence_mask.sum() == 0:
                    skipped["no_sentences"] += 1
                    continue

                self.data.append({
                    "article_id": aid,
                    "node_features": feats,
                    "edge_index": edge_index,
                    "edge_types": edge_types,
                    "labels": labels,
                    "sentence_mask": sentence_mask,
                    "sentence_indices": torch.tensor(sent_idx),
                    "sentence_texts": sent_texts,
                    "article_label": article_label
                })

            except Exception as e:
                print(f"Error in {aid}: {e}")
                import traceback
                traceback.print_exc()
                continue

        print(f"\nLoaded: {len(self.data)}")
        print(f"Skipped: {skipped}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


# ============================================================
# SAMPLE WEIGHTING
# ============================================================
def get_sample_weights(dataset):
    weights = []
    for item in dataset:
        labels = item["labels"][item["sentence_mask"]]
        num_biased = int(labels.sum())
        total_sents = len(labels)
        weight = 1.0 + (num_biased / max(1, total_sents)) * 3.0
        weights.append(weight)
    return weights


# ============================================================
# RGAT MODEL
# ============================================================
class RelationalGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_relations, num_heads=4, dropout=0.5):
        super().__init__()
        assert out_dim % num_heads == 0

        self.num_rel = num_relations
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.out_dim = out_dim

        self.W_rel = nn.ModuleList([
            nn.Linear(in_dim, out_dim, bias=False)
            for _ in range(num_relations)
        ])
        self.attn = nn.Parameter(torch.Tensor(num_relations, num_heads, 2 * self.head_dim))
        self.dropout = nn.Dropout(dropout)
        self.leaky = nn.LeakyReLU(0.2)
        nn.init.xavier_uniform_(self.attn)

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


class RGAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_relations, num_heads=4, num_layers=1, dropout=0.5):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList([
            RelationalGATLayer(hidden_dim, hidden_dim, num_relations, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)
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
# TRAINING HELPERS
# ============================================================
def encode_sentences(tokenizer, lm, texts, training=False):
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt"
    )
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    
    if training:
        out = lm(**enc)
        return out.last_hidden_state[:, 0, :]
    else:
        with torch.no_grad():
            out = lm(**enc)
            return out.last_hidden_state[:, 0, :]


def train_epoch(model, lm, tok, loader, opt, sched, crit):
    model.train()
    lm.train()
    
    total_loss = 0.0
    preds_all, labels_all = [], []
    
    opt.zero_grad()

    for i, batch in enumerate(loader):
        feats = batch["node_features"].to(DEVICE)
        edge = batch["edge_index"].to(DEVICE)
        etypes = batch["edge_types"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        mask = batch["sentence_mask"].to(DEVICE)
        sent_idx = batch["sentence_indices"].to(DEVICE)
        texts = batch["sentence_texts"]

        lm_emb = encode_sentences(tok, lm, texts, training=True)
        
        N, _ = feats.size()
        D_lm = lm_emb.size(1)
        lm_full = torch.zeros(N, D_lm, device=DEVICE)
        lm_full[sent_idx] = lm_emb

        combined = torch.cat([feats, lm_full], dim=-1)

        logits = model(combined, edge, etypes, mask)

        sent_labels = labels[mask]


        loss = crit(logits, sent_labels) / ACCUMULATION_STEPS
        loss.backward()

        if (i + 1) % ACCUMULATION_STEPS == 0:
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            nn.utils.clip_grad_norm_(lm.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            if sched is not None:
                sched.step()
        
        total_loss += loss.item() * ACCUMULATION_STEPS

        preds = torch.argmax(logits, dim=1).cpu().numpy()
        labels_np = sent_labels.cpu().numpy()
        preds_all.extend(preds)
        labels_all.extend(labels_np)

    f1 = f1_score(labels_all, preds_all, average="macro", zero_division=0)
    return total_loss / len(loader), f1


def evaluate(model, lm, tok, loader, crit):
    model.eval()
    lm.eval()
    total_loss = 0.0
    preds_all, labels_all = [], []

    with torch.no_grad():
        for batch in loader:
            feats = batch["node_features"].to(DEVICE)
            edge = batch["edge_index"].to(DEVICE)
            etypes = batch["edge_types"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            mask = batch["sentence_mask"].to(DEVICE)
            sent_idx = batch["sentence_indices"].to(DEVICE)
            texts = batch["sentence_texts"]

            lm_emb = encode_sentences(tok, lm, texts, training=False)
            N, _ = feats.size()
            D_lm = lm_emb.size(1)
            lm_full = torch.zeros(N, D_lm, device=DEVICE)
            lm_full[sent_idx] = lm_emb

            combined = torch.cat([feats, lm_full], dim=-1)

            logits = model(combined, edge, etypes, mask)
            sent_labels = labels[mask]

            loss = crit(logits, sent_labels)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            labels_np = sent_labels.cpu().numpy()
            preds_all.extend(preds)
            labels_all.extend(labels_np)

    p, r, f1, _ = precision_recall_fscore_support(labels_all, preds_all, average="macro", zero_division=0)
    return total_loss / len(loader), p, r, f1, preds_all, labels_all


# ============================================================
# MAIN
# ============================================================
def main():
    print(f"Using device: {DEVICE}\n")
    
    article_ids = [f.replace(".graphml", "") for f in os.listdir(GRAPH_DIR) if f.endswith(".graphml")]
    print(f"Found {len(article_ids)} articles\n")
    
    train_ids, temp_ids = train_test_split(article_ids, test_size=0.3, random_state=42)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)

    train_ds = ERGDataset(train_ids)
    val_ds   = ERGDataset(val_ids)
    test_ds  = ERGDataset(test_ids)

    train_weights = get_sample_weights(train_ds)
    sampler = WeightedRandomSampler(
        weights=train_weights,
        num_samples=len(train_weights),
        replacement=True
    )

    train_loader = DataLoader(train_ds, batch_size=1, sampler=sampler, collate_fn=collate_single)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False, collate_fn=collate_single)
    test_loader  = DataLoader(test_ds,  batch_size=1, shuffle=False, collate_fn=collate_single)

    # Get input dimension from first sample
    sample = train_ds[0]
    static_dim = sample["node_features"].size(1)
    print(f"Static feature dimension: {static_dim}")

    # Load RoBERTa
    print("="*70)
    print("FINE-TUNING CONFIGURATION")
    print("="*70)
    
    tokenizer = AutoTokenizer.from_pretrained(LM_MODEL_NAME)
    lm = AutoModel.from_pretrained(LM_MODEL_NAME).to(DEVICE)
    
    # Freeze all, then unfreeze last 2 layers
    for param in lm.parameters():
        param.requires_grad = False
    
    num_unfrozen = 0
    for i in [10, 11]:
        for param in lm.encoder.layer[i].parameters():
            param.requires_grad = True
            num_unfrozen += 1
    
    print(f"✅ Unfroze last 2 RoBERTa layers ({num_unfrozen} parameters)")
    print(f"✅ RoBERTa learning rate: {LM_LEARNING_RATE}")
    print(f"✅ RGAT learning rate: {LEARNING_RATE}")
    print("="*70 + "\n")
    
    lm_dim = lm.config.hidden_size
    in_dim = static_dim + lm_dim

    model = RGAT(
        in_dim=in_dim, 
        hidden_dim=HIDDEN_DIM, 
        num_relations=NUM_RELATIONS, 
        num_layers=NUM_LAYERS, 
        dropout=DROPOUT
    ).to(DEVICE)

    # Calculate class weights
    num_unbiased = sum([int((item["labels"][item["sentence_mask"]] == 0).sum()) for item in train_ds])
    num_biased = sum([int((item["labels"][item["sentence_mask"]] == 1).sum()) for item in train_ds])

    print(f"[Class balance] Unbiased: {num_unbiased}, Biased: {num_biased}")
    
    w_unbiased = 1.0
    w_biased = np.sqrt(num_unbiased / max(1, num_biased))
    class_weights = torch.tensor([w_unbiased, w_biased], dtype=torch.float32).to(DEVICE)
    print(f"[Class weights] 0={w_unbiased:.2f}, 1={w_biased:.2f}\n")

    # Optimizer with two learning rates
    rgat_params = list(model.parameters())
    lm_params = [p for p in lm.parameters() if p.requires_grad]
    
    opt = torch.optim.AdamW([
        {'params': rgat_params, 'lr': LEARNING_RATE, 'weight_decay': 0.05},
        {'params': lm_params, 'lr': LM_LEARNING_RATE, 'weight_decay': 0.01}
    ])
    
    total_steps = len(train_loader) * NUM_EPOCHS
    warmup = int(total_steps * WARMUP_STEPS_RATIO)
    sched = get_linear_schedule_with_warmup(opt, warmup, total_steps)

    crit = FocalLoss(alpha=class_weights, gamma=1.5)

    best_val_loss = float('inf')
    patience_counter = 0

    print("="*70)
    print("TRAINING WITH ROBERTA FINE-TUNING")
    print("="*70 + "\n")

    for epoch in range(NUM_EPOCHS):
        tr_loss, tr_f1 = train_epoch(model, lm, tokenizer, train_loader, opt, sched, crit)
        val_loss, vp, vr, vf1, vpreds, vlabels = evaluate(model, lm, tokenizer, val_loader, crit)

        print(f"E{epoch+1}: TL={tr_loss:.4f}, TF1={tr_f1:.3f} | VL={val_loss:.4f}, VP={vp:.3f}, VR={vr:.3f}, VF1={vf1:.3f}")
        
        if (epoch + 1) % 5 == 0:
            cm = confusion_matrix(vlabels, vpreds)
            print(f"CM: {cm.tolist()}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'rgat': model.state_dict(),
                'lm': lm.state_dict()
            }, os.path.join(OUTPUT_DIR, "best_model_finetuned.pt"))
            patience_counter = 0
            print(f"  ✅ Saved (VL={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("  ⏹ Early stopping.")
                break

    # TEST
    print("\n" + "="*70)
    print("LOADING BEST MODEL FOR TEST")
    print("="*70)
    
    checkpoint = torch.load(os.path.join(OUTPUT_DIR, "best_model_finetuned.pt"))
    model.load_state_dict(checkpoint['rgat'])
    lm.load_state_dict(checkpoint['lm'])
    
    test_loss, p, r, f1, preds, labels = evaluate(model, lm, tokenizer, test_loader, crit)

    print("\n" + "="*70)
    print("FINAL TEST RESULTS")
    print("="*70)
    print(f"Loss:      {test_loss:.4f}")
    print(f"Precision: {p:.3f}")
    print(f"Recall:    {r:.3f}")
    print(f"F1 Score:  {f1:.3f}")
    print(f"Accuracy:  {np.mean(np.array(preds) == np.array(labels)):.3f}")
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(labels, preds))
    
    print("\nClassification Report:")
    print(classification_report(labels, preds, digits=3, target_names=['Unbiased', 'Biased']))
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"Saved model: {OUTPUT_DIR}/best_model_finetuned.pt")


if __name__ == "__main__":
    main()