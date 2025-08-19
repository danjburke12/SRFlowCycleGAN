#!/usr/bin/env python
import sys
import os
sys.path.append('code')

import torch
import torch.nn as nn
from collections import OrderedDict
from options.options import parse
from models import create_model
import numpy as np

def create_dummy_data():
    """Create dummy 3D data for testing"""
    # Create 5D tensor: batch x channels x depth x height x width
    lr_data = torch.randn(1, 1, 32, 32, 32)  # Low resolution input
    gt_data = torch.randn(1, 1, 64, 64, 64)  # Ground truth (2x upscaled)
    
    data = {
        'LQ': lr_data,
        'GT': gt_data
    }
    return data

def test_forward_pass():
    print("Testing forward pass step by step...")
    
    # Configuration
    opt = {
        'model': 'SRFlow',
        'distortion': 'sr',
        'scale': 2,
        'device': 'cpu',
        'network_G': {
            'which_model_G': 'SRFlowNet',
            'in_nc': 1,
            'out_nc': 1, 
            'nf': 32,
            'nb': 4,
            'flow': {
                'K': 8,
                'L': 3,
                'hidden_channels': 32,
                'coupling': 'CondAffineSeparatedAndCond',
                'condAff': {'in_channels_rrdb': 32},
                'split': {'enable': True, 'type': 'Split3d', 'consume_ratio': 0.5},
                'fea_up0': True,
                'fea_up-1': True
            },
            'isotropic3d': False,
            'scale': 2
        },
        'train': {
            'niter': 1000,
            'warmup_iter': 0,
            'manual_seed': 42,
            'lr_G': 0.0002,
            'weight_decay_G': 0,
            'beta1': 0.9,
            'beta2': 0.999,
            'lr_scheme': 'MultiStepLR',
            'lr_steps': [500, 800],
            'lr_gamma': 0.5,
            'weight_fl': 1.0,
            'weight_l1': 0.01
        },
        'val': {
            'heats': [0.0, 0.5],
            'n_sample': 1
        },
        'dist': False,
        'is_train': True,
        'gpu_ids': [],
        'use_tb_logger': False
    }
    
    # Create model
    model = create_model(opt)
    print("Model created successfully")
    
    # Create and feed data
    data = create_dummy_data()
    model.feed_data(data)
    
    print(f"LR input shape: {data['LQ'].shape}")
    print(f"GT input shape: {data['GT'].shape}")
    
    # Test the specific failing operations step by step
    try:
        print("\n=== Testing forward pass (encode) ===")
        with torch.no_grad():
            # This is what happens in weight_fl branch
            z, nll, y_logits = model.netG(gt=model.real_H, lr=model.var_L, reverse=False)
            print(f"Forward (encode) successful!")
            print(f"z shape: {z.shape}")
            print(f"nll: {nll}")
            
    except Exception as e:
        print(f"Forward (encode) failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    try:
        print("\n=== Testing reverse pass (decode) ===")
        with torch.no_grad():
            # This is what happens in weight_l1 branch
            z_for_decode = model.get_z(heat=0, seed=None, batch_size=model.var_L.shape[0], lr_shape=model.var_L.shape)
            print(f"Generated z shape: {z_for_decode.shape}")
            
            sr, logdet = model.netG(lr=model.var_L, z=z_for_decode, eps_std=0, reverse=True, reverse_with_grad=True)
            print(f"Reverse (decode) successful!")
            print(f"sr shape: {sr.shape}")
            
    except Exception as e:
        print(f"Reverse (decode) failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n=== Both forward and reverse passes successful! ===")

if __name__ == "__main__":
    test_forward_pass()
