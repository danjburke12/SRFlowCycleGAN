# Copyright (c) 2020 Huawei Technologies Co., Ltd.
# Licensed under CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International) (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#
# The code is released for academic research use only. For commercial use, please contact Huawei Technologies Co., Ltd.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file contains content licensed by https://github.com/chaiyujin/glow-pytorch/blob/master/LICENSE

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.modules.RRDBNet_arch import RRDBNet, RRDBNet3D
from models.modules.FlowUpsamplerNet import FlowUpsamplerNet, FlowUpsamplerNet3D
from models.modules import thops, flow
from utils.util import opt_get


class SRFlowNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32, scale=4, K=None, opt=None, step=None):
        super(SRFlowNet, self).__init__()

        self.opt = opt
        self.quant = 255 if opt_get(opt, ['datasets', 'train', 'quant']) is \
                            None else opt_get(opt, ['datasets', 'train', 'quant'])
        use_iso3d = opt_get(opt, ['network_G', 'isotropic3d']) or False
        self.is_3d = use_iso3d
        self.in_nc = in_nc
        print(f"DEBUG SRFlowNet: opt['network_G'] = {opt.get('network_G', None)}")
        print(f"DEBUG SRFlowNet: use_iso3d = {use_iso3d}")
        # Always use RRDBNet3D for 3D input (5D tensor)
        if use_iso3d:
            print("DEBUG SRFlowNet: Using RRDBNet3D for volumetric data")
            self.RRDB = RRDBNet3D(in_nc, out_nc, nf, nb, gc, scale, opt)
        else:
            print("DEBUG SRFlowNet: Using RRDBNet (2D) for image data")
            self.RRDB = RRDBNet(in_nc, out_nc, nf, nb, gc, scale, opt)

        # Runtime assertion: if input is 5D, must use RRDBNet3D
        original_forward = self.RRDB.forward
        def rrdb_forward_patch(x, get_steps=False):
            if x.dim() == 5:
                # Error if not using RRDBNet3D for volumetric data
                if not use_iso3d or not isinstance(self.RRDB, RRDBNet3D):
                    print(f"RRDB type: {type(self.RRDB)}")
                    print(f"use_iso3d: {use_iso3d}")
                    raise ValueError("ERROR: 5D input detected but RRDBNet3D not properly configured! Check isotropic3d setting.")
            return original_forward(x, get_steps=get_steps)
        self.RRDB.forward = rrdb_forward_patch
        hidden_channels = opt_get(opt, ['network_G', 'flow', 'hidden_channels'])
        hidden_channels = hidden_channels or 64
        self.RRDB_training = True  # Default is true

        train_RRDB_delay = opt_get(self.opt, ['network_G', 'train_RRDB_delay'])
        set_RRDB_to_train = False
        if set_RRDB_to_train:
            self.set_rrdb_training(True)

        # Get patch size from config or use default
        patch_size = opt_get(opt, ['datasets', 'train', 'patch_size']) or 64
        
        if use_iso3d:
            # For 3D data, shape is (D, H, W, C)
            input_shape = (patch_size, patch_size, patch_size, in_nc)
            self.flowUpsamplerNet = FlowUpsamplerNet3D(
                input_shape, 
                hidden_channels, 
                K,
                flow_coupling=opt['network_G']['flow']['coupling'],
                opt=opt)
        else:
            # For 2D data, shape is (H, W, C)
            input_shape = (patch_size, patch_size, in_nc)
            self.flowUpsamplerNet = FlowUpsamplerNet(
                input_shape,
                hidden_channels,
                K,
                flow_coupling=opt['network_G']['flow']['coupling'],
                opt=opt)
                
        self.i = 0

    def set_rrdb_training(self, trainable):
        if self.RRDB_training != trainable:
            for p in self.RRDB.parameters():
                p.requires_grad = trainable
            self.RRDB_training = trainable
            return True
        return False

    def forward(self, gt=None, lr=None, z=None, eps_std=None, reverse=False, epses=None, reverse_with_grad=False,
                lr_enc=None, add_gt_noise=False, step=None, y_label=None, max_depth=100):
        if not reverse:
            return self.normal_flow(gt, lr, epses=epses, lr_enc=lr_enc, add_gt_noise=add_gt_noise, step=step,
                                    y_onehot=y_label, max_depth=max_depth)
        else:
            # Remove channel assertion since we're handling temporal data as channels
            batch_size, channels, height, width = lr.shape
            # Store original shape for reconstruction
            self.temporal_shape = (batch_size, channels, height, width)
            # assert lr.shape[2] == 20
            # assert lr.shape[3] == 20
            # assert z.shape[0] == 1
            # assert z.shape[1] == 3 * 8 * 8
            # assert z.shape[2] == 20
            # assert z.shape[3] == 20
            if reverse_with_grad:
                return self.reverse_flow(lr, z, y_onehot=y_label, eps_std=eps_std, epses=epses, lr_enc=lr_enc,
                                         add_gt_noise=add_gt_noise, max_depth=max_depth)
            else:
                with torch.no_grad():
                    return self.reverse_flow(lr, z, y_onehot=y_label, eps_std=eps_std, epses=epses, lr_enc=lr_enc,
                                             add_gt_noise=add_gt_noise, max_depth=max_depth)

    def normal_flow(self, gt, lr, y_onehot=None, epses=None, lr_enc=None, add_gt_noise=True, step=None, max_depth=100):
        if lr_enc is None:
            lr_enc = self.rrdbPreprocessing(lr)

        if self.is_3d:
            # For 3D data (B, C, D, H, W)
            logdet = torch.zeros_like(gt[:, 0, 0, 0, 0])
            pixels = gt.shape[2] * gt.shape[3] * gt.shape[4]  # D * H * W
        else:
            # Original 2D behavior
            logdet = torch.zeros_like(gt[:, 0, 0, 0])
            pixels = thops.pixels(gt)

        z = gt

        if add_gt_noise:
            # Setup
            noiseQuant = opt_get(self.opt, ['network_G', 'flow', 'augmentation', 'noiseQuant'], True)
            if noiseQuant:
                z = z + ((torch.rand(z.shape, device=z.device) - 0.5) / self.quant)
            logdet = logdet + float(-np.log(self.quant) * pixels)

        # Encode
        epses, logdet = self.flowUpsamplerNet(rrdbResults=lr_enc, gt=z, logdet=logdet, reverse=False, epses=epses,
                                              y_onehot=y_onehot, max_depth=max_depth)

        objective = logdet.clone()

        if isinstance(epses, (list, tuple)):
            z = epses[-1]
        else:
            z = epses

        objective = objective + flow.GaussianDiag.logp(None, None, z)

        nll = (-objective) / float(np.log(2.) * pixels)

        # Return appropriate format based on whether epses is a list
        if isinstance(epses, list):
            # When intermediate results are requested
            if self.is_3d:
                # For 3D data, ensure proper shape
                return [eps.contiguous() for eps in epses], nll.contiguous(), logdet.contiguous()
            else:
                # Original 2D behavior
                return epses, nll, logdet
        else:
            # Single result case
            if self.is_3d:
                # For 3D data, ensure proper shape
                return z.contiguous(), nll.contiguous(), logdet.contiguous()
            else:
                # Original 2D behavior
                return z, nll, logdet

    def rrdbPreprocessing(self, lr):
        rrdbResults = self.RRDB(lr, get_steps=True)
        block_idxs = opt_get(self.opt, ['network_G', 'flow', 'stackRRDB', 'blocks']) 
        if block_idxs and not isinstance(block_idxs, (list, tuple)):
            block_idxs = [block_idxs]  # Convert single value to list
        block_idxs = block_idxs or []  # Ensure we have a list, even if empty
        if block_idxs:
            concat = torch.cat([rrdbResults["block_{}".format(idx)] for idx in block_idxs], dim=1)

            if opt_get(self.opt, ['network_G', 'flow', 'stackRRDB', 'concat']) or False:
                keys = ['last_lr_fea', 'fea_up1', 'fea_up2', 'fea_up4']
                if 'fea_up0' in rrdbResults.keys():
                    keys.append('fea_up0')
                if 'fea_up-1' in rrdbResults.keys():
                    keys.append('fea_up-1')
                if self.opt['scale'] >= 8:
                    keys.append('fea_up8')
                if self.opt['scale'] == 16:
                    keys.append('fea_up16')
                for k in keys:
                    if k not in rrdbResults:
                        # Fallback: create zeros of correct shape
                        if self.is_3d:
                            # Use shape from last_lr_fea
                            d, h, w = rrdbResults['last_lr_fea'].shape[2:5]
                            rrdbResults[k] = torch.zeros_like(rrdbResults['last_lr_fea'])
                        else:
                            h = rrdbResults['last_lr_fea'].shape[2]
                            w = rrdbResults['last_lr_fea'].shape[3]
                            rrdbResults[k] = torch.zeros_like(rrdbResults['last_lr_fea'])
                    if self.is_3d:
                        d, h, w = rrdbResults[k].shape[2:5]
                        rrdbResults[k] = torch.cat([rrdbResults[k], F.interpolate(concat, size=(d, h, w))], dim=1)
                    else:
                        h = rrdbResults[k].shape[2]
                        w = rrdbResults[k].shape[3]
                        rrdbResults[k] = torch.cat([rrdbResults[k], F.interpolate(concat, (h, w))], dim=1)
        return rrdbResults

    def get_score(self, disc_loss_sigma, z):
        score_real = 0.5 * (1 - 1 / (disc_loss_sigma ** 2)) * thops.sum(z ** 2, dim=[1, 2, 3]) - \
                     z.shape[1] * z.shape[2] * z.shape[3] * math.log(disc_loss_sigma)
        return -score_real

    def reverse_flow(self, lr, z, y_onehot, eps_std, epses=None, lr_enc=None, add_gt_noise=True, max_depth=100):
        logdet = torch.zeros_like(lr[:, 0, 0, 0])
        pixels = thops.pixels(lr) * self.opt['scale'] ** 2

        if add_gt_noise:
            logdet = logdet - float(-np.log(self.quant) * pixels)

        if lr_enc is None:
            lr_enc = self.rrdbPreprocessing(lr)

        x, logdet = self.flowUpsamplerNet(rrdbResults=lr_enc, z=z, eps_std=eps_std, reverse=True, epses=epses,
                                          logdet=logdet, max_depth=max_depth)

        return x, logdet
