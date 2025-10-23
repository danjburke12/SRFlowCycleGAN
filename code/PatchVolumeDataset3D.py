import os
import torch
import torch.utils.data as data
import numpy as np

class PatchVolumeDataset3D(data.Dataset):
    """
    Patch-based 3D volume dataset for bidirectional mapping (He <-> H+).
    Each sample is a pair of patches: (He[i], H+[i]) from the same location.
    """
    def __init__(self, he_dir, hp_dir, indices, depth, height, width, patch_size=64):
        self.he_files = sorted([os.path.join(he_dir, f) for f in os.listdir(he_dir) if f.lower().endswith('.dat')])
        self.hp_files = sorted([os.path.join(hp_dir, f) for f in os.listdir(hp_dir) if f.lower().endswith('.dat')])
        assert len(self.he_files) == len(self.hp_files), "He and H+ must have same number of files"
        self.indices = indices
        self.depth = depth
        self.height = height
        self.width = width
        self.patch_size = patch_size
        # Each volume will yield multiple patches
        self.patches_per_volume = 4  # Can adjust this to control dataset size

    def __len__(self):
        return len(self.indices) * self.patches_per_volume

    def _load_volume(self, path):
        arr = np.fromfile(path, dtype=np.float32)
        arr = arr.reshape(self.depth, self.height, self.width)
        return torch.from_numpy(arr)

    def __getitem__(self, idx):
        # Calculate which volume and which patch in that volume
        volume_idx = self.indices[idx // self.patches_per_volume]
        patch_idx = idx % self.patches_per_volume

        # Load volumes
        he_vol = self._load_volume(self.he_files[volume_idx])
        hp_vol = self._load_volume(self.hp_files[volume_idx])

        # Use deterministic patch locations based on patch_idx
        # This ensures we get different patches from each volume
        patch_locations = [
            (0, 0, 0),  # Top-left-front
            (0, 0, self.width - self.patch_size),  # Top-left-back
            (0, self.height - self.patch_size, 0),  # Top-right-front
            (self.depth - self.patch_size, 0, 0),  # Bottom-left-front
        ]
        d, h, w = patch_locations[patch_idx]

        # Extract patches
        he_patch = he_vol[d:d+self.patch_size, h:h+self.patch_size, w:w+self.patch_size]
        hp_patch = hp_vol[d:d+self.patch_size, h:h+self.patch_size, w:w+self.patch_size]

        # Add batch/channel dims for model (N,C,D,H,W)
        he_patch = he_patch.unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
        hp_patch = hp_patch.unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)

        # Return metadata about patch location for logging/debugging
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
