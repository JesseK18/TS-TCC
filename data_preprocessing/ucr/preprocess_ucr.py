import os
import argparse
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split

def load_tsv(path):
    """
    Expect a TSV where the first column is the label and the rest are time‐series features.
    Returns:
      data: np.ndarray of shape (N, T)
      labels: np.ndarray of shape (N,)
    """
    df = pd.read_csv(path, sep='\t', header=None)
    #labels = df.iloc[:,0].to_numpy(dtype=np.int64)
    labels_raw = df.iloc[:, 0].to_numpy()
    data   = df.iloc[:,1:].to_numpy(dtype=np.float32)
    
    uniq = np.unique(labels_raw)
    label_map = {l: i for i, l in enumerate(uniq)}
    labels = np.vectorize(label_map.get)(labels_raw).astype(np.int64)
    return data, labels

def zscore(train, test):
    mean = np.nanmean(train)
    std  = np.nanstd(train)
    train = (train - mean) / std
    test  = (test  - mean) / std
    # replace any remaining NaNs with 0
    train = np.nan_to_num(train, nan=0.0)
    test  = np.nan_to_num(test,  nan=0.0)
    return train, test

def to_tensor(x: np.ndarray):
    """
    Converts (N, T) → torch.FloatTensor of shape (N, 1, T)
    """
    t = torch.from_numpy(x)
    # add channel dim
    return t.unsqueeze(1)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset',   type=str, required=True,
                   help='Name of the UCR dataset folder (e.g. ECG5000)')
    p.add_argument('--raw_dir',   type=str, required=True,
                   help='Where train.tsv/test.tsv live')
    p.add_argument('--out_dir',   type=str, required=True,
                   help='Where to write train.pt/val.pt/test.pt')
    p.add_argument('--val_size',  type=float, default=0.2,
                   help='Fraction of train → val split')
    p.add_argument('--seed',      type=int, default=42)
    args = p.parse_args()

    # paths
    train_tsv = os.path.join(args.raw_dir, args.dataset + '_TRAIN.tsv')
    test_tsv  = os.path.join(args.raw_dir, args.dataset + '_TEST.tsv')
    os.makedirs(args.out_dir, exist_ok=True)

    # load raw
    X_train, y_train = load_tsv(train_tsv)
    X_test,  y_test  = load_tsv(test_tsv)
    X_train, X_test = zscore(X_train, X_test)

    # train → train/val
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train,
        test_size=args.val_size,
        random_state=args.seed,
        stratify=y_train
    )

    # save each split
    for split, X, y in [
        ('train', X_tr, y_tr),
        ('val',   X_val, y_val),
        ('test',  X_test, y_test)
    ]:
        samples = to_tensor(X)         # (N,1,T)
        labels  = torch.from_numpy(y)  # (N,)
        torch.save({'samples': samples, 'labels': labels},
                   os.path.join(args.out_dir, f'{split}.pt'))
        print(f"Saved {split}.pt → {samples.shape}, labels {labels.shape}")

if __name__ == '__main__':
    main()


"""
python data_preprocessing/ucr/preprocess_ucr.py \
  --dataset FaceFour \
  --raw_dir  data/UCR/FaceFour \
  --out_dir  data/UCR/FaceFour \
  --val_size 0.2
  
  """