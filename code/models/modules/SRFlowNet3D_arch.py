import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from . import module_util as mutil
from .FlowUpsamplerNet import FlowUpsamplerNet3D
from . import thops
from . import flow
from .RRDBNet_arch import RRDBNet3D
from utils.util import opt_get
from .checkpoint_util import checkpoint_wrapper


class SRFlowNet3D(nn.Module):
    """3D version of SRFlowNet adapted for volumetric data.

    Key differences:
    1. Uses RRDBNet3D as the base feature extractor
    2. Flow operations preserve 5D tensor structure (B,C,D,H,W)
    3. Isotropic handling of spatial dimensions
    4. Support for gradient checkpointing to reduce memory usage
    """
    def __init__(self, in_nc, out_nc, nf, nb, gc=32, scale=4, K=None, opt=None, step=None):
        super(SRFlowNet3D, self).__init__()
        self.use_checkpoint = False  # Flag to enable/disable gradient checkpointing
        
        self.opt = opt
        self.step = step
        hidden_channels = opt_get(opt, ['network_G', 'flow', 'hidden_channels']) or 64
        
        # Set up RRDB feature extractor (3D version)
        # Note: We don't need an explicit check for input dimensions since RRDBNet3D 
        # automatically expects 5D input
        self.RRDB = RRDBNet3D(in_nc=in_nc, out_nc=out_nc, nf=nf, nb=nb, gc=gc, scale=scale, opt=opt)

        # Initialize Flow-based model for 3D data
        patch_size = opt_get(opt, ['datasets', 'train', 'patch_size'])
        self.flow = FlowUpsamplerNet3D(
            image_shape=(opt_get(opt, ['datasets', 'train', 'patch_size'], 248), 
                        opt_get(opt, ['datasets', 'train', 'patch_size'], 248),
                        opt_get(opt, ['datasets', 'train', 'patch_size'], 248),
                        in_nc),  # (D,H,W,C) format
            hidden_channels=hidden_channels, 
            K=K, 
            L=opt_get(opt, ['network_G', 'flow', 'L']) or 3,
            actnorm_scale=1.0,
            flow_permutation=None,
            flow_coupling=opt_get(opt, ["network_G", "flow", "flow_coupling"]),
            LU_decomposed=False,
            opt=opt)

    def forward(self, gt=None, lr=None, z=None, eps_std=None, reverse=False, epses=None, reverse_with_grad=False):
        if not reverse:
            return self.normal_flow(gt, lr, epses=epses)
        else:
            # Reverse flow (sampling/super-resolution)
            if z is None:
                assert eps_std is not None
                z = self.get_z(gt, eps_std)
            
            if reverse_with_grad:
                return self.reverse_flow(lr, z, reconstruct=True)
            else:
                with torch.no_grad():
                    return self.reverse_flow(lr, z, reconstruct=True)

    def normal_flow(self, gt, lr, epses=None):
        """Forward flow: data -> latent"""
        # Extract features using 3D RRDB
        pixel_logdet = None
        feats = self.extract_features(lr)  # Get intermediate features
        
        if epses is not None:
            z, logdet = self.flow(gt=gt, logdet=0, lr=lr, z=epses, eps_std=None, reverse=False, feats=feats)
        else:
            z, logdet = self.flow(gt=gt, logdet=0, lr=lr, eps_std=None, reverse=False, feats=feats)
            
        logdet = logdet + pixel_logdet if pixel_logdet is not None else logdet
        nll = -logdet
        return z, nll, feats

    def reverse_flow(self, lr, z, reconstruct=False):
        """Reverse flow: latent -> data"""
        # Extract features for conditioning
        feats = self.extract_features(lr)  # Get intermediate features
        
        sr, logdet = self.flow(gt=None, logdet=0, lr=lr, z=z, eps_std=None, reverse=True, feats=feats)
        return sr, logdet

    def extract_features(self, lr, get_steps=False):
        """Extract features using 3D RRDB network."""
        lr = lr.contiguous()
        if self.use_checkpoint and self.training:
            if not get_steps:
                # Use gradient checkpointing for feature extraction
                out = checkpoint_wrapper(self.RRDB, (lr, get_steps))
            else:
                # Get intermediate features with checkpointing
                out = checkpoint_wrapper(self.RRDB, (lr, get_steps))
                feats = out if isinstance(out, dict) else {'last_lr_fea': out, 'fea_up1': out}
        else:
            if not get_steps:
                # Normal feature extraction
                out = self.RRDB(lr, get_steps=get_steps)
            else:
                # Get intermediate features if needed
                out = self.RRDB(lr, get_steps=get_steps)
                feats = out if isinstance(out, dict) else {'last_lr_fea': out, 'fea_up1': out}
        return out
        
    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing to save memory."""
        self.use_checkpoint = True
        
    def disable_gradient_checkpointing(self):
        """Disable gradient checkpointing."""
        self.use_checkpoint = False

    def get_z(self, gt, eps_std):
        """Get random z for sampling."""
        b, c, d, h, w = gt.shape
        z = torch.randn(b, c * 8, d//4, h//4, w//4) * eps_std
        return z
