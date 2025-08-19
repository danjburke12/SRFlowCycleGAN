#!/usr/bin/env python
import sys
import os
sys.path.append('code')

import torch
import traceback
from models.SRFlow_model import SRFlowModel
import yaml

# Use CPU
device = 'cpu' if not torch.cuda.is_available() else 'cuda'

def debug_training_error():
    print("Using device:", device)
    
    # Load config
    with open('configs/ionization_3d_h.yml', 'r') as f:
        opt = yaml.safe_load(f)
    
    # Add missing fields
    opt['is_train'] = True
    opt['dist'] = False
    
    # Add missing training parameters
    if 'train' not in opt:
        opt['train'] = {}
    
    train_params = {
        'lr_G': 5e-4,
        'weight_decay_G': 0,
        'beta1': 0.9,
        'beta2': 0.999,
        'lr_scheme': 'MultiStepLR',
        'lr_steps_rel': [0.5, 0.75, 0.9, 0.95],
        'lr_gamma': 0.5,
        'restarts': None,
        'restart_weights': None,
        'eta_min': 1e-7,
        'niter': 200000,
        'warmup_iter': -1,
        'clear_state': True,
        'manual_seed': 12345,
        'val_freq': 5e3
    }
    
    for key, value in train_params.items():
        if key not in opt['train']:
            opt['train'][key] = value
    
    # Create model  
    print(f"Creating model with isotropic3d flag: {opt['network_G'].get('isotropic3d', False)}")
    model = SRFlowModel(opt, 0)
    
    # Create 5D test tensors: batch x channels x depth x height x width
    lr_tensor = torch.randn(1, 1, 32, 32, 32).to(device)
    gt_tensor = torch.randn(1, 1, 64, 64, 64).to(device)
    
    print("LR tensor shape:", lr_tensor.shape)
    print("GT tensor shape:", gt_tensor.shape)
    
    # Create data dictionary
    data = {
        'LQ': lr_tensor,
        'GT': gt_tensor
    }
    
    # Feed data to model
    model.feed_data(data)
    
    # Try to run optimize_parameters with detailed error tracking
    try:
        print("Calling optimize_parameters...")
        nll = model.optimize_parameters(0)
        print(f"Success! NLL: {nll}")
    except Exception as e:
        print("ERROR occurred:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print("\nFull traceback:")
        traceback.print_exc()
        
        # Try to find where the Conv2d call is coming from
        tb = traceback.format_exc()
        lines = tb.split('\n')
        for i, line in enumerate(lines):
            if 'conv2d' in line.lower():
                print(f"\nFound Conv2d reference at line {i}: {line}")
                # Print surrounding context
                for j in range(max(0, i-3), min(len(lines), i+4)):
                    marker = " >>> " if j == i else "     "
                    print(f"{marker}{lines[j]}")

if __name__ == "__main__":
    debug_training_error()
