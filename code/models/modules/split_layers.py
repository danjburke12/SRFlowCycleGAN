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

import torch
from torch import nn as nn

from models.modules import thops
from models.modules.FlowStep import FlowStep
from models.modules.flow import Conv2dZeros, Conv3dZeros, GaussianDiag
from utils.util import opt_get

class Split2d(nn.Module):
    def __init__(self, num_channels, logs_eps=0, cond_channels=0, position=None, consume_ratio=0.5, opt=None):
        super(Split2d, self).__init__()

        self.num_channels_consume = int(round(num_channels * consume_ratio))
        self.num_channels_pass = num_channels - self.num_channels_consume

        self.conv = Conv2dZeros(in_channels=self.num_channels_pass + cond_channels,
                                out_channels=self.num_channels_consume * 2)
        self.logs_eps = logs_eps
        self.position = position
        self.opt = opt

    def split2d_prior(self, z, ft):
        if ft is not None:
            z = torch.cat([z, ft], dim=1)
        h = self.conv(z)
        return thops.split_feature(h, "cross")

    def exp_eps(self, logs):
        return torch.exp(logs) + self.logs_eps

    def forward(self, input, logdet=0., reverse=False, eps_std=None, eps=None, ft=None, y_onehot=None, max_depth=100):
        if max_depth <= 0:
            raise RuntimeError("Maximum recursion depth exceeded in Split2d")

        if not reverse:
            z1, z2 = self.split_ratio(input)
            mean, logs = self.split2d_prior(z1, ft)
            eps = (z2 - mean) / self.exp_eps(logs)
            logdet = logdet + self.get_logdet(logs, mean, z2)
            return z1, logdet, eps
        else:
            z1 = input
            mean, logs = self.split2d_prior(z1, ft)
            if eps is None:
                eps = GaussianDiag.sample_eps(mean.shape, eps_std)
            eps = eps.to(mean.device)
            z2 = mean + self.exp_eps(logs) * eps
            z = thops.cat_feature(z1, z2)
            logdet = logdet - self.get_logdet(logs, mean, z2)
            return z, logdet

    def get_logdet(self, logs, mean, z2):
        logdet_diff = GaussianDiag.logp(mean, logs, z2)
        return logdet_diff

    def split_ratio(self, input):
        z1, z2 = input[:, :self.num_channels_pass, ...], input[:, self.num_channels_pass:, ...]
        return z1, z2


class Split3d(nn.Module):
    """3D version of Split2d for volumetric flow."""
    
    def __init__(self, num_channels, logs_eps=0, cond_channels=0, position=None, consume_ratio=0.5, opt=None):
        super(Split3d, self).__init__()
        
        # Handle single channel case
        if num_channels == 1:
            consume_ratio = 0.0  # Don't consume any channels when we only have 1
            
        self.num_channels_consume = max(0, int(round(num_channels * consume_ratio)))
        self.num_channels_pass = num_channels - self.num_channels_consume
        
        # Skip creating conv layer if we're not consuming any channels
        if self.num_channels_consume > 0:
            self.conv = Conv3dZeros(in_channels=self.num_channels_pass + cond_channels,
                                    out_channels=self.num_channels_consume * 2)
        else:
            self.conv = None
            
        self.logs_eps = logs_eps
        self.position = position
        self.opt = opt
        
        # Track recursion depth
        self.current_depth = 0
        self.max_depth = 100

    def split3d_prior(self, z, ft):
        if ft is not None:
            z = torch.cat([z, ft], dim=1)
        if self.conv is not None:
            h = self.conv(z)
            return thops.split_feature(h, "cross")
        return None, None

    def exp_eps(self, logs):
        return torch.exp(logs) + self.logs_eps

    def forward(self, input, logdet=0., reverse=False, eps_std=None, eps=None, ft=None, y_onehot=None, max_depth=100):
        # Increment depth counter
        self.current_depth += 1
        
        if self.current_depth > max_depth:
            raise RuntimeError(f"Maximum recursion depth {max_depth} exceeded in Split3d")
        
        try:
            # Special case for single channel
            if self.num_channels_consume == 0:
                if not reverse:
                    return input, logdet, None
                else:
                    return input, logdet
            
            # Main forward pass
            if not reverse:
                z1, z2 = self.split_ratio(input)
                
                mean, logs = self.split3d_prior(z1, ft)
                if mean is None or logs is None:
                    return input, logdet, None
                
                eps = (z2 - mean) / self.exp_eps(logs)
                logdet = logdet + self.get_logdet(logs, mean, z2)
                
                return z1, logdet, eps
                
            # Main reverse pass    
            else:
                z1 = input
                
                mean, logs = self.split3d_prior(z1, ft)
                if mean is None or logs is None:
                    return input, logdet

                if eps is None:
                    eps = GaussianDiag.sample_eps(mean.shape, eps_std)

                eps = eps.to(mean.device)
                z2 = mean + self.exp_eps(logs) * eps
                
                z = thops.cat_feature(z1, z2)
                logdet = logdet - self.get_logdet(logs, mean, z2)
                
                return z, logdet
                
        finally:
            # Always decrement the depth counter when exiting
            self.current_depth -= 1

    def get_logdet(self, logs, mean, z2):
        logdet_diff = GaussianDiag.logp(mean, logs, z2)
        return logdet_diff

    def split_ratio(self, input):
        """Split input tensor along channel dimension according to consume ratio."""
        z1, z2 = input[:, :self.num_channels_pass, ...], input[:, self.num_channels_pass:, ...]
        return z1, z2
