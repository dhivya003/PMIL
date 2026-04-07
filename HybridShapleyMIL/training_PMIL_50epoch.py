

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler  # MIXED PRECISION

# PATHS
FEATURE_DIR = r"E:\FYP\HybridShapleyMIL\features"
CSV_PATH    = r"E:\FYP\HybridShapleyMIL\data_splits\DATA_SPLIT.csv"
MODEL_DIR   = r"E:\FYP\HybridShapleyMIL\models"
RESULT_DIR  = r"E:\FYP\HybridShapleyMIL\results"
CKPT_FILE   = r"E:\FYP\HybridShapleyMIL\models\pmil_ultra_ckpt.pth"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df)} slides")

class WSIDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        slide_id = row['slide_id'].replace('.tif', '.pt')
        feats = torch.load(os.path.join(FEATURE_DIR, slide_id))
        label = torch.tensor(int(row['label']), dtype=torch.long)
        return feats, label, row['slide_id']

class PMILHybrid(nn.Module):
    def __init__(self, feat_dim=1280):
        super().__init__()
        self.attention_V = nn.Linear(feat_dim, 128)
        self.attention_U = nn.Linear(128, 1)
        self.classifier = nn.Linear(feat_dim, 2)

    def forward(self, x):
        if x.dim() == 1: x = x.unsqueeze(0)
        V = torch.tanh(self.attention_V(x))
        A = self.attention_U(V).squeeze(-1)
        A = torch.softmax(A, dim=0)
        M = torch.mm(A.unsqueeze(0), x)
        logits = self.classifier(M)
        return logits, A

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training on: {device}")

def fast_shapley(feats, model, samples=10):
    N = feats.shape[0]
    if N <= 1: return torch.zeros(N, device=device)
    _, attn = model(feats)
    K = min(300, N)
    topk_idx = torch.topk(attn, K).indices
    topk_feats = feats[topk_idx]
    shap_topk = torch.zeros(K, device=device)
    model.eval()
    with torch.no_grad():
        for i in range(K):
            contrib = 0.0
            valid = 0
            for _ in range(samples):
                perm = torch.randperm(K, device=device)
                full = topk_feats[perm]
                minus = topk_feats[perm[perm != i]]
                if len(minus) == 0: continue
                _, A_full = model(full)
                _, A_minus = model(minus)
                pos = (perm == i).nonzero(as_tuple=True)[0]
                if len(pos) == 0: continue
                contrib += (A_full[pos] - A_minus.mean()).item()
                valid += 1
            if valid > 0:
                shap_topk[i] = contrib / valid
    shap = torch.zeros(N, device=device)
    shap[topk_idx] = shap_topk
    return shap / (shap.std() + 1e-8)

def create_pseudo_bags(feats, iis, M):
    idx = torch.argsort(-iis)
    bags = [[] for _ in range(M)]
    for rank, i in enumerate(idx):
        bags[rank % M].append(i.item())
    return [feats[b] for b in bags if len(b) > 10]  # skip tiny bags

# RESUME
start_fold = 0
all_aucs = []
all_accs = []
if os.path.exists(CKPT_FILE):
    ckpt = torch.load(CKPT_FILE)
    start_fold = ckpt['fold']
    all_aucs = ckpt['aucs']
    all_accs = ckpt['accs']

scaler = GradScaler()  # for mixed precision

for fold in range(start_fold, 5):
    print(f"\nFOLD {fold+1}/5 —  M-STEP")
    train_df = df[df['fold'] != fold]
    val_df = df[df['fold'] == fold]

    train_loader = DataLoader(WSIDataset(train_df), batch_size=1, shuffle=True)
    val_loader = DataLoader(WSIDataset(val_df), batch_size=1)

    model = PMILHybrid().to(device)
    optimizer = optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    M = 1
    best_auc = 0
    best_acc = 0

    for round in range(10):
        print(f"  Round {round+1}/10 | Pseudo-bags: {M}")

        # E-step (fast)
        model.eval()
        iis_dict = {}
        for feats, _, sid in tqdm(train_loader, desc="E-step"):
            feats = feats.squeeze(0).to(device)
            _, attn = model(feats)
            shap = fast_shapley(feats, model, samples=10)
            iis = 0.5 * attn + 0.5 * shap
            iis_dict[sid[0]] = iis.detach()

        # M-STEP — OPTIMIZED (TOP-3 BAGS + MIXED PRECISION)
        model.train()
        for epoch in tqdm(range(50), desc="M-step "):
            for feats, label, sid in train_loader:
                feats = feats.squeeze(0).to(device)
                label = label.to(device)
                bags = create_pseudo_bags(feats, iis_dict[sid[0]], M)
                bags = bags[:3]  # ONLY TOP 3 BAGS

                if not bags: continue

                optimizer.zero_grad()
                with autocast():  # MIXED PRECISION 
                    loss = 0
                    for bag in bags:
                        logits, _ = model(bag)
                        loss += criterion(logits, label)
                    loss /= len(bags)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        # Validation (same)
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for feats, label, _ in val_loader:
                feats = feats.squeeze(0).to(device)
                logits, _ = model(feats)
                prob = torch.softmax(logits, 1)[0,1].item()
                preds.append(prob)
                trues.append(label.item())
        auc = roc_auc_score(trues, preds)
        acc = accuracy_score(trues, np.array(preds) > 0.5)

        if auc > best_auc:
            best_auc = auc
            best_acc = acc
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, f"pmil_fold{fold}.pth"))

        print(f"    AUC: {auc:.4f} | ACC: {acc:.4f} → Best: {best_auc:.4f} | {best_acc:.4f}")

        if (round + 1) % 3 == 0 and M < 8:
            M += 2

    all_aucs.append(best_auc)
    all_accs.append(best_acc)

mean_auc = np.mean(all_aucs)
mean_acc = np.mean(all_accs)

print(f"FINAL AUC: {mean_auc:.4f} | ACCURACY: {mean_acc:.4f}")

with open(os.path.join(RESULT_DIR, "FINAL_RESULT.txt"), "w") as f:
    f.write(f" PMIL\nMean AUC: {mean_auc:.4f}\nMean Accuracy: {mean_acc:.4f}\n")
