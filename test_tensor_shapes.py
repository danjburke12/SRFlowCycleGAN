#!/usr/bin/env python3
import sys
import os
sys.path.append('code')
sys.path.append('.')

import torch

def test_tensor_shapes():
    # Test concatenation of 4D and 5D tensors
    try:
        z1_4d = torch.randn(1, 384, 8, 8)  # 4D
        ft_5d = torch.randn(1, 32, 8, 8, 8)  # 5D
        result = torch.cat([z1_4d, ft_5d], dim=1)
        print(f"4D+5D concat success: {result.shape}")
    except Exception as e:
        print(f"4D+5D concat failed: {e}")
    
    # Test what should happen in 3D
    try:
        z1_5d = torch.randn(1, 384, 8, 8, 8)  # 5D
        ft_5d = torch.randn(1, 32, 8, 8, 8)  # 5D  
        result = torch.cat([z1_5d, ft_5d], dim=1)
        print(f"5D+5D concat success: {result.shape}")
    except Exception as e:
        print(f"5D+5D concat failed: {e}")
        
    # Test tensor split behavior
    print("\n=== Testing tensor split ===")
    z_5d = torch.randn(1, 768, 8, 8, 8)
    z1, z2 = z_5d.chunk(2, dim=1)
    print(f"Original z: {z_5d.shape}")
    print(f"After split z1: {z1.shape}, z2: {z2.shape}")
    
    # Test tensor split in wrong way
    z_4d = torch.randn(1, 768, 8, 8)  
    z1_wrong, z2_wrong = z_4d.chunk(2, dim=1)
    print(f"4D Original z: {z_4d.shape}")
    print(f"4D After split z1: {z1_wrong.shape}, z2: {z2_wrong.shape}")

if __name__ == '__main__':
    test_tensor_shapes()
