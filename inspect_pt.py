import argparse
import torch
import numpy as np

def inspect(obj, name, indent=0):
    prefix = ' ' * indent
    if isinstance(obj, torch.Tensor) or isinstance(obj, np.ndarray):
        arr = obj if isinstance(obj, np.ndarray) else obj.detach().cpu().numpy()
        print(f"{prefix}{name}: array, dtype={arr.dtype}, shape={arr.shape}")
        # show first few elements
        flat = arr.reshape(-1)
        n = min(5, flat.shape[0])
        print(f"{prefix}  sample values: {flat[:n].tolist()}{'...' if flat.shape[0]>n else ''}")
    else:
        print(f"{prefix}{name}: {type(obj)} = {obj}")

def main():
    p = argparse.ArgumentParser(
        description="Inspect contents of a .pt data file (dict of tensors/arrays)"
    )
    p.add_argument('--pt_path', type=str, required=True,
                   help="Path to .pt file, e.g. data/UCR/Adiac/val.pt")
    args = p.parse_args()

    data = torch.load(args.pt_path, map_location='cpu')
    print(f"Loaded {args.pt_path}, top‐level type: {type(data)}")

    if isinstance(data, dict):
        for k, v in data.items():
            inspect(v, k, indent=2)
    else:
        inspect(data, 'data', indent=2)

if __name__ == '__main__':
    main()
    
    """
    python inspect_pt.py --pt_path data/UCR/Adiac/val.pt
    """