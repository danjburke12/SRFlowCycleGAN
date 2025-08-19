#!/usr/bin/env python3

"""Debug script to trace channel flow in SRFlow 3D."""

import sys
import os
sys.path.append('code')

from models.modules.FlowUpsamplerNet import FlowUpsamplerNet3D
from utils.util import opt_get

# Mock options similar to ionization_3d_h.yml
opt = {
    'scale': 2,
    'network_G': {
        'flow': {
            'K': 8,
            'L': 3,
            'hidden_channels': 32,
            'flow_coupling': 'CondAffineSeparatedAndCond',
            'split': {
                'enable': True,
                'type': 'Split3d',
                'consume_ratio': 0.5
            }
        }
    }
}

print("Testing FlowUpsamplerNet3D channel flow...")

try:
    # Test with 1 channel (grayscale)
    image_shape = (32, 64, 64, 1)  # DHWC
    print(f"Input image_shape: {image_shape}")
    
    flow_net = FlowUpsamplerNet3D(
        image_shape=image_shape,
        hidden_channels=32,
        K=8,
        L=3,
        flow_coupling='CondAffineSeparatedAndCond',
        opt=opt
    )
    
    print("FlowUpsamplerNet3D created successfully!")
    
    # Print the first FlowStep to see what channels it has
    first_flowstep = None
    for i, layer in enumerate(flow_net.layers):
        if hasattr(layer, 'actnorm3d'):
            first_flowstep = layer
            print(f"First FlowStep found at layer {i}")
            print(f"ActNorm3d num_features: {layer.actnorm3d.num_features}")
            print(f"ActNorm2d num_features: {layer.actnorm2d.num_features}")
            break
    
    if first_flowstep is None:
        print("No FlowStep with ActNorm found")
    else:
        print("\\nNow testing channel flow with dummy data...")
        import torch
        
        # Create dummy input - start with 1 channel
        batch_size = 1
        gt = torch.randn(batch_size, 1, 32, 64, 64)  # Start with (B,C,D,H,W) = (1,1,32,64,64)
        
        print(f"Input GT shape: {gt.shape}")
        
        # Create dummy RRDB results
        rrdbResults = {}
        for key in flow_net.levelToName.values():
            # These would normally come from the RRDB encoder
            rrdbResults[key] = torch.randn(batch_size, 64, 16, 32, 32)  # Dummy values
        
        try:
            z, logdet = flow_net.forward(gt=gt, rrdbResults=rrdbResults, reverse=False)
            print(f"Forward pass successful! Output shape: {z.shape if torch.is_tensor(z) else [zi.shape for zi in z]}")
        except Exception as forward_e:
            print(f"Forward pass error: {forward_e}")
            import traceback
            traceback.print_exc()
    
except Exception as e:
    print(f"Error during FlowUpsamplerNet3D creation: {e}")
    import traceback
    traceback.print_exc()
