#!/usr/bin/env python3
import sys
import os
sys.path.append('code')
sys.path.append('.')

import torch
import yaml

# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def load_config(config_path):
    with open(config_path, 'r') as f:
        opt = yaml.safe_load(f)
    return opt

def debug_forward_pass():
    # Load config
    opt = load_config('configs/ionization_3d_h.yml')
    
    # Add necessary options
    opt['gpu_ids'] = [0] if torch.cuda.is_available() else None
    opt['dist'] = False
    opt['path'] = {
        'log': 'experiments/debug',
        'experiments_root': 'experiments/debug',
        'models': 'experiments/debug/models',
        'training_states': 'experiments/debug/training_states',
        'val_images': 'experiments/debug/val_images',
        'results_root': 'results/debug',
        'pretrain_model_G': None  # Disable pretrained model
    }
    opt['is_train'] = True
    
    # Add missing train options
    if 'train' not in opt:
        opt['train'] = {}
    opt['train']['restarts'] = [200, 400, 600]
    opt['train']['restart_weights'] = [1, 1, 1]
    opt['train']['eta_min'] = 1e-7
    opt['train']['clear_state'] = False
    
    # Create model
    from models.SR_model import SRModel
    from models.SRFlowBidirectional_model import SRFlowBidirectionalModel
    
    if opt['model'] == 'SRFlowBidirectional':
        model = SRFlowBidirectionalModel(opt, step=0)
    else:
        model = SRModel(opt)
    else:
        model = SRModel(opt)
    
    # Create test data
    batch_size = 1
    channels = 1
    depth, height, width = 32, 32, 32  # Scale down for debugging
    
    # Create LR and GT tensors in the format the model expects
    lr_data = torch.randn(batch_size, channels, depth, height, width)
    gt_data = torch.randn(batch_size, channels, depth * 2, height * 2, width * 2)  # 2x upscale
    
    if torch.cuda.is_available():
        lr_data = lr_data.cuda()
        gt_data = gt_data.cuda()
    
    print(f"LR tensor shape: {lr_data.shape}")
    print(f"GT tensor shape: {gt_data.shape}")
    
    # Prepare data dict
    data = {
        'LQ': lr_data,
        'GT': gt_data
    }
    
    try:
        # Feed data to model
        model.feed_data(data)
        
        # Perform forward pass
        print("Starting forward pass...")
        model.optimize_parameters(1)  # step 1
        
    except Exception as e:
        import traceback
        print(f"Error during forward pass: {e}")
        print("Full traceback:")
        traceback.print_exc()

if __name__ == '__main__':
    debug_forward_pass()
