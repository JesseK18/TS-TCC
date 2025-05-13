import os
import argparse
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split

def load_tsv(path):
    df = pd.read_csv(path, sep='\t', header=None)
    labels_raw = df.iloc[:, 0].to_numpy()
    data   = df.iloc[:, 1:].to_numpy(dtype=np.float32)
    uniq = np.unique(labels_raw)
    label_map = {l: i for i, l in enumerate(uniq)}
    labels = np.vectorize(label_map.get)(labels_raw).astype(np.int64)
    return data, labels

def zscore(train, test):
    mean = np.nanmean(train)
    std  = np.nanstd(train)
    train = (train - mean) / std
    test  = (test  - mean) / std
    return np.nan_to_num(train), np.nan_to_num(test)

def to_tensor(x: np.ndarray):
    return torch.from_numpy(x).unsqueeze(1)   # (N,1,T)

def process_one(dataset, raw_root, out_root, val_size, seed):
    src = os.path.join(raw_root, dataset)
    trg = os.path.join(out_root, dataset)
    os.makedirs(trg, exist_ok=True)
    train_tsv = os.path.join(src, f"{dataset}_TRAIN.tsv")
    test_tsv  = os.path.join(src, f"{dataset}_TEST.tsv")
    X_train, y_train = load_tsv(train_tsv)
    X_test,  y_test  = load_tsv(test_tsv)
    X_train, X_test = zscore(X_train, X_test)
    
    # --- NEW: only stratify if every class has >=2 samples ---
    uniq, counts = np.unique(y_train, return_counts=True)
    if np.any(counts < 2):
        print(f"[{dataset}] WARNING: some classes <2 samples; doing a random (non-stratified) split")
        stratify_arg = None
    else:
        stratify_arg = y_train
        
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train,
        test_size=val_size,
        random_state=seed,
        stratify=stratify_arg
    )
    for split, X, y in [
        ('train', X_tr, y_tr),
        ('val',   X_val, y_val),
        ('test',  X_test, y_test)
    ]:
        samples = to_tensor(X)
        labels  = torch.from_numpy(y)
        torch.save({'samples': samples, 'labels': labels},
                   os.path.join(trg, f"{split}.pt"))
        print(f"[{dataset}] Saved {split}.pt → {samples.shape}, labels {labels.shape}")

def main():
    p = argparse.ArgumentParser(
        description="Batch‐convert all UCR TSVs → .pt splits"
    )
    p.add_argument('--raw_dir',   type=str, required=True,
                   help='Root of UCR raw folders (each subfolder = one dataset)')
    p.add_argument('--out_dir',   type=str, required=True,
                   help='Where to write the per‐dataset train/val/test .pt')
    p.add_argument('--val_size',  type=float, default=0.2,
                   help='Fraction of train → val split')
    p.add_argument('--seed',      type=int,   default=42)
    args = p.parse_args()

    for ds in sorted(os.listdir(args.raw_dir)):
        path = os.path.join(args.raw_dir, ds)
        # skip non‐directories and hidden
        if not os.path.isdir(path) or ds.startswith('.'):
            continue
        print(args.raw_dir, ds)
        print(f"\nProcessing dataset: {ds}")
        process_one(ds, args.raw_dir, args.out_dir, args.val_size, args.seed)

if __name__ == '__main__':
    main()