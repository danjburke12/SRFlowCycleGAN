#!/usr/bin/env python
import sys
import os
sys.path.append('code')

import torch
import torch.nn as nn
from collections import OrderedDict
from models.SRFlow_model import SRFlowModel
import yaml

# Use CPU
device = 'cpu' if not torch.cuda.is_available() else 'cuda'

def debug_optimize_parameters():
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
    model = SRFlowModel(opt, 0)
    
    # Create 5D test tensors: batch x channels x depth x height x width
    lr_tensor = torch.randn(1, 1, 32, 32, 32).to(device)
    gt_tensor = torch.randn(1, 1, 64, 64, 64).to(device)
    
    print("LR tensor shape:", lr_tensor.shape)
    print("GT tensor shape:", gt_tensor.shape)
    
    # Set model data
    model.var_L = lr_tensor
    model.real_H = gt_tensor
    
    # Test step by step what optimize_parameters does
    print("\n=== Testing optimize_parameters step by step ===")
    
    model.netG.train()
    
    try:
        print("Step 1: Testing weight_fl branch (NLL loss)")
        # This is the weight_fl branch from optimize_parameters
        weight_fl = 1.0
        if weight_fl > 0:
            z, nll, y_logits = model.netG(gt=model.real_H, lr=model.var_L, reverse=False)
            nll_loss = torch.mean(nll)
            loss_1 = nll_loss * weight_fl
            print(f"NLL loss successful: {loss_1.item()}")
        
    except Exception as e:
        print(f"NLL loss failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    try:
        print("Step 2: Testing weight_l1 branch (L1 loss)")
        # This is the weight_l1 branch from optimize_parameters  
        weight_l1 = 0.01
        if weight_l1 > 0:
            z = model.get_z(heat=0, seed=None, batch_size=model.var_L.shape[0], lr_shape=model.var_L.shape)
            print(f"Generated z shape: {z.shape}")
            
            # The critical part - reverse with gradients enabled
            sr, logdet = model.netG(lr=model.var_L, z=z, eps_std=0, reverse=True, reverse_with_grad=True)
            print(f"Reverse with gradients successful: sr.shape={sr.shape}")
            
            l1_loss = (sr - model.real_H).abs().mean()
            loss_2 = l1_loss * weight_l1
            print(f"L1 loss successful: {loss_2.item()}")
            
    except Exception as e:
        print(f"L1 loss (reverse with grad) failed: {e}")
        import traceback
        traceback.print_exc()
        return
        
    try:
        print("Step 3: Testing backward pass")
        total_loss = loss_1 + loss_2
        print(f"Total loss: {total_loss.item()}")
        
        # This is where the error might be happening
        total_loss.backward()
        print("Backward pass successful!")
        
    except Exception as e:
        print(f"Backward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n=== All optimize_parameters steps completed successfully! ===")

if __name__ == "__main__":
    debug_optimize_parameters()
