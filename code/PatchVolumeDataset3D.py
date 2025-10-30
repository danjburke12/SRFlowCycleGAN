import os
import torch
import torch.utils.data as data
import numpy as np

class PatchVolumeDataset3D(data.Dataset):
    """
    Patch-based 3D volume dataset for bidirectional mapping (He <-> H+).
    Each sample is a pair of patches: (He[i], H+[i]) from the same location.
    """
    def __init__(self, he_dir, hp_dir, indices, depth, height, width, crop_size=(16,16,16), croptimes=1, normalize=True):
        self.he_files = sorted([os.path.join(he_dir, f) for f in os.listdir(he_dir) if f.lower().endswith('.dat')])
        self.hp_files = sorted([os.path.join(hp_dir, f) for f in os.listdir(hp_dir) if f.lower().endswith('.dat')])
        assert len(self.he_files) == len(self.hp_files), "He and H+ must have same number of files"
        self.indices = indices
        self.depth = depth
        self.height = height
        self.width = width
        self.crop_size = crop_size if isinstance(crop_size, (tuple, list)) else (crop_size, crop_size, crop_size)
        self.croptimes = croptimes
        self.normalize = normalize
        self.patches_per_volume = croptimes

    def __len__(self):
        return len(self.indices) * self.patches_per_volume

    def _load_volume(self, path):
        arr = np.fromfile(path, dtype=np.float32)
        arr = arr.reshape(self.depth, self.height, self.width)
        if self.normalize:
            arr = 2 * (arr - np.min(arr)) / (np.max(arr) - np.min(arr) + 1e-8) - 1
        return torch.from_numpy(arr)

    def __getitem__(self, idx):
        # Calculate which volume and which crop in that volume
        volume_idx = self.indices[idx // self.patches_per_volume]
        crop_idx = idx % self.patches_per_volume

        # Load volumes
        he_vol = self._load_volume(self.he_files[volume_idx])
        hp_vol = self._load_volume(self.hp_files[volume_idx])

        # Random crop
        D, H, W = self.crop_size
        max_d = self.depth - D
        max_h = self.height - H
        max_w = self.width - W
        if max_d > 0:
            d = np.random.randint(0, max_d + 1)
        else:
            d = 0
        if max_h > 0:
            h = np.random.randint(0, max_h + 1)
        else:
            h = 0
        if max_w > 0:
            w = np.random.randint(0, max_w + 1)
        else:
            w = 0

        he_patch = he_vol[d:d+D, h:h+H, w:w+W]
        hp_patch = hp_vol[d:d+D, h:h+H, w:w+W]

        # Add batch/channel dims for model (N,C,D,H,W)
        he_patch = he_patch.unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
        hp_patch = hp_patch.unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)

        metadata = {
            'He_path': self.he_files[volume_idx],
            'H+_path': self.hp_files[volume_idx],
            'patch_location': (d, h, w)
        }

        return {
            'LQ': he_patch,
            'GT': hp_patch,
            'LQ_path': metadata['He_path'],
            'GT_path': metadata['H+_path']
        }
