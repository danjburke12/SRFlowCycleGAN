
import torch
import numpy as np
import os
import sys
import csv
import yaml
sys.path.append('code')
from models.SRFlow_model import SRFlowModel

# Settings
DAT_DIR = r"D:\Daniel\Ionization-Vis-2008\H"
DEPTH, HEIGHT, WIDTH = 600, 248, 248
CSV_OUT = "inference_results.csv"
PATCH_SIZE = 64  # Patch size for each dimension

# Load config and model
with open('configs/ionization_3d_h.yml', 'r') as f:
    opt = yaml.safe_load(f)
opt['is_train'] = False
opt['dist'] = False
model = SRFlowModel(opt, step=0)
model.netG.eval()

# Load checkpoint (update path if needed)
ckpt_path = 'experiments/ionization_3d_h/models/latest_G.pth'
if os.path.exists(ckpt_path):
    model.load_network(ckpt_path, model.netG, strict=False)
else:
    print(f'Checkpoint not found: {ckpt_path}')

def load_dat_volume(path):
    arr = np.fromfile(path, dtype=np.float32)
    arr = arr.reshape(DEPTH, HEIGHT, WIDTH)
    return torch.from_numpy(arr)

def split_into_patches(volume, patch_size):
    """Yield patches and their indices from a 3D tensor."""
    D, H, W = volume.shape
    for d in range(0, D, patch_size):
        for h in range(0, H, patch_size):
            for w in range(0, W, patch_size):
                patch = volume[
                    d:min(d+patch_size, D),
                    h:min(h+patch_size, H),
                    w:min(w+patch_size, W)
                ]
                yield patch, (d, h, w)

def reconstruct_from_patches(patches, volume_shape, patch_size):
    """Reconstruct full volume from patches and their indices."""
    D, H, W = volume_shape
    full = torch.zeros((D, H, W), dtype=patches[0][0].dtype)
    for patch, (d, h, w) in patches:
        full[
            d:d+patch.shape[0],
            h:h+patch.shape[1],
            w:w+patch.shape[2]
        ] = patch
    return full

# List .dat files
dat_files = [f for f in os.listdir(DAT_DIR) if f.lower().endswith('.dat')]
results = []
import torch.nn.functional as F
scale = opt.get('scale', 2)  # Default to 2 if not set

for fname in dat_files:
    fpath = os.path.join(DAT_DIR, fname)
    volume = load_dat_volume(fpath)
    sr_patches = []
    for patch, (d, h, w) in split_into_patches(volume, PATCH_SIZE):
        hr_tensor = patch.unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
        lr_tensor = F.interpolate(hr_tensor, scale_factor=1/scale, mode='trilinear', align_corners=False, recompute_scale_factor=True)
        with torch.no_grad():
            sr_patch = model.get_sr(lq=lr_tensor, heat=1.0)
        # Remove batch/channel dims
        sr_patch = sr_patch.squeeze(0).squeeze(0)
        sr_patches.append((sr_patch, (d, h, w)))
    # Reconstruct full SR volume
    sr_full = reconstruct_from_patches(sr_patches, volume.shape, PATCH_SIZE)
    results.append({
        'filename': fname,
        'mean_pred': sr_full.mean().item()
    })

# Write results to CSV
with open(CSV_OUT, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=['filename', 'mean_pred'])
    writer.writeheader()
    for row in results:
        writer.writerow(row)
print(f'Saved inference results to {CSV_OUT}')
