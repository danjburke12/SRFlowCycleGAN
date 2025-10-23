import numpy as np
import os
import glob

def check_volume_files(data_dir):
    print(f"Checking files in {data_dir}...")
    
    # List all .dat files
    files = glob.glob(os.path.join(data_dir, "*.dat"))
    if not files:
        print(f"No .dat files found in {data_dir}")
        return
    
    print(f"Found {len(files)} .dat files")
    
    # Check first file
    first_file = files[0]
    print(f"\nChecking first file: {first_file}")
    
    try:
        data = np.fromfile(first_file, dtype=np.float32)
        print(f"Data shape: {data.shape}")
        print(f"Data size: {data.size}")
        
        # Try different cube root factors
        cube_root = round(np.cbrt(data.size))
        print(f"\nPossible 3D dimensions:")
        for d in range(cube_root-2, cube_root+3):
            if data.size % d == 0:
                remaining = data.size // d
                sqrt_remaining = round(np.sqrt(remaining))
                if remaining % sqrt_remaining == 0:
                    h = sqrt_remaining
                    w = remaining // h
                    print(f"{d} x {h} x {w} = {d*h*w}")
    
    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    # Check He data
    he_dir = "D:\\Daniel\\Ionization-Vis-2008\\He"
    hp_dir = "D:\\Daniel\\Ionization-Vis-2008\\H+"
    
    check_volume_files(he_dir)
    check_volume_files(hp_dir)
