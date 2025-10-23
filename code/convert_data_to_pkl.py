import os
import numpy as np
import pickle
from tqdm import tqdm

def read_dat_file(file_path):
    """Read a .dat file and return the 3D volume data"""
    return np.fromfile(file_path, dtype=np.float32)

def convert_and_save_pkl(input_dir, output_dir, prefix, n_train=180, patch_size=64):
    """Convert .dat files to pickle format and split into train/val sets"""
    # Get all .dat files
    files = sorted([f for f in os.listdir(input_dir) if f.endswith('.dat')])
    
    # Lists to store volumes
    train_volumes = []
    val_volumes = []
    
    print(f"Processing {prefix} files...")
    for i, file in enumerate(tqdm(files)):
        # Read volume data
        volume = read_dat_file(os.path.join(input_dir, file))
        
        # Reshape to 3D (248x248x600)
        volume = volume.reshape(248, 248, 600)
        
        # Add channel dimension (C,D,H,W format)
        volume = volume[np.newaxis, ...]
        
        # Split into train/val sets
        if i < n_train:
            train_volumes.append(volume)
        else:
            val_volumes.append(volume)
    
    # Save train data
    train_path = os.path.join(output_dir, f'{prefix}_train.pklv4')
    print(f"Saving {len(train_volumes)} training volumes to {train_path}")
    with open(train_path, 'wb') as f:
        pickle.dump(train_volumes, f, protocol=4)
    
    # Save validation data
    val_path = os.path.join(output_dir, f'{prefix}_val.pklv4')
    print(f"Saving {len(val_volumes)} validation volumes to {val_path}")
    with open(val_path, 'wb') as f:
        pickle.dump(val_volumes, f, protocol=4)

def main():
    # Convert He data
    convert_and_save_pkl(
        input_dir='D:/Daniel/SRFlowCycleGAN/data/He',
        output_dir='D:/Daniel/SRFlowCycleGAN/data/He',
        prefix='He'
    )
    
    # Convert H+ data
    convert_and_save_pkl(
        input_dir='D:/Daniel/SRFlowCycleGAN/data/Hp',
        output_dir='D:/Daniel/SRFlowCycleGAN/data/Hp',
        prefix='Hp'
    )

if __name__ == '__main__':
    main()
