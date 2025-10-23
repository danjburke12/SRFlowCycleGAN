import sys
import os

# Add code directory to Python path
code_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'code'))
sys.path.insert(0, code_dir)
print(f"Added {code_dir} to Python path")

# Clear import cache
for mod in list(sys.modules.keys()):
    if mod.startswith('models.modules'):
        del sys.modules[mod]
print("Cleared module cache")

# Try importing
print("Testing imports...")
from models.modules.Split import Split2d, Split3d
print("Successfully imported Split2d and Split3d")

# Test instantiation
print("Testing instantiation...")
split2d = Split2d(num_channels=64)
split3d = Split3d(num_channels=64)
print("Successfully created Split2d and Split3d instances")
