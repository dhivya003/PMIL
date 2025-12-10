import argparse, os, torch, numpy as np, pandas as pd
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import json

parser = argparse.ArgumentParser()
parser.add_argument('--feat_dir', default='features_effnet')
parser.add_argument('--csv_path', default='data_splits/DATA_SPLIT.csv')
parser.add_argument('--ckpt_dir', default='ckpts_effnet')
parser.add_argument('--log_dir', default='logs_effnet')
parser.add_argument('--resume', action='store_true')  # ← resume from checkpoint
args = parser.parse_args()

os.makedirs(args.ckpt_dir, exist_ok=True)
os.makedirs(args.log_dir, exist_ok=True)
writer = SummaryWriter(args.log_dir)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
df = pd.read_csv(args.csv_path)
results = {}

for fold in range(5):
    print(f"\nFOLD {fold} / 5")
    train_df = df[df['fold'] != fold].reset_index(drop=True)
    val_df = df[df['fold'] == fold].reset_index(drop=True)
    
    # Resume logic
    ckpt_path = f"{args.ckpt_dir}/fold{fold}.pth"
    start_round = 0
    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path)
        start_round = ckpt['round'] + 1
        print(f"Resuming from round {start_round}")

    model = ABMIL(feat_dim=1280).to(device)
    if args.resume and os.path.exists(ckpt_path):
        model.load_state_dict(ckpt['model'])

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    best_auc = 0
    for round in range(start_round, 10):
        # ... [same E-M loop as before]
        # After each round → SAVE CHECKPOINT
        torch.save({
            'round': round,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'val_auc': val_auc
        }, ckpt_path)
        print(f"Checkpoint saved: round {round}, AUC: {val_auc:.4f}")

    results[f'fold_{fold}'] = best_auc
    writer.add_scalar(f'Fold_{fold}/AUC', best_auc)

# Final result
mean_auc = np.mean(list(results.values()))
print(f"\nFINAL RESULT: {mean_auc:.4f} ± {np.std(list(results.values())):.4f}")
with open("final_results.json", "w") as f:
    json.dump({"mean_auc": mean_auc, "fold_aucs": results}, f, indent=2)