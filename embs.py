# import torch
# from models.model import base_Model
# from models.TC import TC
# from config_files.UCR_Configs import Config

# #get embs from SSL training from saved file. 

# # 1) re-build config and model
# cfg = Config()

# # (auto-overwrite cfg.input_channels, cfg.features_len, cfg.num_classes exactly as in main.py)
# meta = torch.load("data/UCR/ECG5000/train.pt", map_location="cpu")
# cfg.input_channels = meta["samples"].shape[1]
# cfg.features_len   = meta["samples"].shape[2]
# cfg.num_classes    = len(torch.unique(meta["labels"]))

# enc = base_Model(cfg).eval().to("cuda")
# tc  = TC(cfg, "cuda").eval().to("cuda")

# # 2) load weights
# ckpt = torch.load("experiments_logs/exp1/run1/self_supervised_seed_42/saved_models/ckp_last.pt",
#                   map_location="cuda")
# enc.load_state_dict(ckpt["model_state_dict"])
# tc.load_state_dict(ckpt["temporal_contr_model_state_dict"])

# # 3) pick your split loader
# from dataloader.dataloader import data_generator
# train_dl, valid_dl, test_dl = data_generator("data/UCR/ECG5000", cfg, "self_supervised")

# # 4) extract and save
# all_embeds = []
# all_labels = []
# with torch.no_grad():
#   for x, y, a1, a2 in train_dl:
#     x = x.to("cuda").float()
#     # forward through encoder
#     _, feats = enc(x)                # feats: (B, C_out, T_after)
#     # either global‐pool
#     embeds = feats.mean(dim=2)       # (B, C_out)
#     # or run through proj‐head for final embedding
#     embeds = tc.projection_head(embeds)  # (B, D)
#     all_embeds.append(embeds.cpu())
#     all_labels.append(y)

# all_embeds = torch.cat(all_embeds)
# all_labels = torch.cat(all_labels)
# torch.save({"embeddings": all_embeds, "labels": all_labels},
#            "UCR_ECG5000_selfsup_embeddings.pt")

import os
import argparse
import torch
from dataloader.dataloader import data_generator
from models.model import base_Model
from models.TC import TC
from config_files.UCR_Configs import Config

def main():
    p = argparse.ArgumentParser(
        description="Extract self-supervised embeddings for a UCR dataset"
    )
    p.add_argument('--dataset_name',   type=str, required=True,
                   help='UCR series folder under data/UCR (e.g. ECG5000)')
    p.add_argument('--exp_desc',       type=str, required=True,
                   help='Experiment description (same as main.py)')
    p.add_argument('--run_desc',       type=str, required=True,
                   help='Run description (same as main.py)')
    p.add_argument('--seed',           type=int, default=42,
                   help='Random seed used for self_supervised run')
    p.add_argument('--ckpt_root',      type=str, default='experiments_logs',
                   help='Root dir where main.py wrote checkpoints')
    p.add_argument('--device',         type=str, default='cpu',
                   help='cpu or cuda')
    p.add_argument('--output_path',    type=str, default=None,
                   help='Where to save embeddings (defaults to {dataset}_embs.pt)')
    args = p.parse_args()

    # Rebuild config and infer C,T,K from train.pt
    cfg = Config()
    data_dir = os.path.join('data', 'UCR', args.dataset_name)
    meta = torch.load(os.path.join(data_dir, 'train.pt'), map_location='cpu')
    cfg.input_channels = meta['samples'].shape[1]
    cfg.features_len   = meta['samples'].shape[2]
    cfg.num_classes    = int(len(torch.unique(meta['labels'])))

    device = torch.device(args.device)
    enc = base_Model(cfg).to(device).eval()
    tc  = TC(cfg, device).to(device).eval()

    # Load self-supervised checkpoint
    #home_dir = "/Users/jessekroll/Desktop/Bachelor Thesis 24/Repos for TSCAR models/TS-TCC"
    home_dir = os.getcwd()
    ckpt_path = os.path.join(
        home_dir,
        args.ckpt_root,
        args.exp_desc,
        args.run_desc,
        f"self_supervised_seed_{args.seed}",
        "saved_models",
        "ckp_last.pt"
    )
    
    # if not os.path.isfile(ckpt_path):
    #     # try to auto-discover any run_* folder under exp_desc
    #     base = os.path.join(home_dir, args.ckpt_root, args.exp_desc)
    #     for d in os.listdir(base):
    #         print(f"→ checking {d}")
    #         cand = os.path.join(base, d,
    #                             f"self_supervised_seed_{args.seed}",
    #                             "saved_models",
    #                             "ckp_last.pt")
    #         if os.path.isfile(cand):
    #             print(f"→ auto-found checkpoint in {cand}")
    #             ckpt_path = cand
    #             break
    #     else:
    #         raise FileNotFoundError(f"No checkpoint found at {ckpt_path}")
    #print(f"Loading checkpoint from {ckpt_path}")
    
    ckpt = torch.load(ckpt_path, map_location=device)
    # 1) only keep encoder weights (drop the old classification head)
    pretrained_dict = {
        k: v
        for k,v in ckpt["model_state_dict"].items()
        if not k.startswith("logits.")
    }

    # 2) load into encoder (strict=False will ignore missing keys)
    enc.load_state_dict(pretrained_dict, strict=False)

    # 3) load the TC head as before
    tc.load_state_dict(ckpt["temporal_contr_model_state_dict"])
    # pretrained_dict = ckpt["model_state_dict"]
    # model_dict = enc.state_dict()
    # model_dict.update(pretrained_dict)
    # enc.load_state_dict(model_dict)
    # tc.load_state_dict(ckpt['temporal_contr_model_state_dict'])

    # DataLoader in self_supervised mode
    train_dl, _, _ = data_generator(data_dir, cfg, 'self_supervised')

    # Extract embeddings
    all_embeds, all_labels = [], []
    with torch.no_grad():
        for x, y, _, _ in train_dl:
            x = x.to(device).float()
            _, feats = enc(x)               # feats: (B, C_out, T_after)
            #pooled   = feats.mean(dim=2)
            _, embeds = tc(feats, feats) # (B, C_out)
            #embeds   = tc.projection_head(pooled)  # (B, D)
            all_embeds.append(embeds.cpu())
            all_labels.append(y)

    all_embeds = torch.cat(all_embeds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    print(all_embeds.shape, all_labels.shape)
    print(all_embeds)
    out_path = args.output_path or f"{args.dataset_name}_ssl_embs.pt"
    torch.save({'embeddings': all_embeds, 'labels': all_labels}, out_path)
    print(f"Saved {all_embeds.shape[0]} embeddings of dim {all_embeds.shape[1]} to {out_path}")

if __name__ == '__main__':
    main()
    
"""
python embs.py \
  --dataset_name   ECG5000 \
  --exp_desc       exp1 \
  --run_desc       run2 \
  --seed           18 \
  --ckpt_root      experiments_logs \
  --device         cpu
"""