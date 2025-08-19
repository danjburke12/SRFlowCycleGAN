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
from models.modules.flow import Conv3d as ConvND, Conv3dZeros as ConvNDZeros
from utils.util import opt_get


class CondAffineSeparatedAndCond(nn.Module):
    def __init__(self, in_channels, opt):
        super().__init__()
        self.need_features = True
        self.in_channels = in_channels
        
        # Debug: print the full opt structure for flow
        print(f"DEBUG opt flow keys: {opt.get('network_G', {}).get('flow', {}).keys()}")
        print(f"DEBUG condAff section: {opt.get('network_G', {}).get('flow', {}).get('condAff', {})}")
        
        self.in_channels_rrdb = opt_get(opt, ['network_G', 'flow', 'condAff', 'in_channels_rrdb'], 320)
        print(f"DEBUG opt_get result for in_channels_rrdb: {self.in_channels_rrdb}")
        
        self.kernel_hidden = 1
        self.affine_eps = 0.0001
        self.n_hidden_layers = 1
        hidden_channels = opt_get(opt, ['network_G', 'flow', 'CondAffineSeparatedAndCond', 'hidden_channels'])
        self.hidden_channels = 64 if hidden_channels is None else hidden_channels

        self.affine_eps = opt_get(opt, ['network_G', 'flow', 'CondAffineSeparatedAndCond', 'eps'],  0.0001)

        self.channels_for_nn = self.in_channels // 2
        self.channels_for_co = self.in_channels - self.channels_for_nn

        if self.channels_for_nn is None:
            self.channels_for_nn = self.in_channels // 2

        self.fAffine = self.F(in_channels=self.channels_for_nn + self.in_channels_rrdb,
                              out_channels=self.channels_for_co * 2,
                              hidden_channels=self.hidden_channels,
                              kernel_hidden=self.kernel_hidden,
                              n_hidden_layers=self.n_hidden_layers)
        
        print(f"DEBUG CondAffineSeparatedAndCond: in_channels={self.in_channels}, channels_for_nn={self.channels_for_nn}, channels_for_co={self.channels_for_co}, in_channels_rrdb={self.in_channels_rrdb}")
        print(f"DEBUG fAffine input channels: {self.channels_for_nn + self.in_channels_rrdb}")

        self.fFeatures = self.F(in_channels=self.in_channels_rrdb,
                                out_channels=self.in_channels * 2,
                                hidden_channels=self.hidden_channels,
                                kernel_hidden=self.kernel_hidden,
                                n_hidden_layers=self.n_hidden_layers)

    def forward(self, input: torch.Tensor, logdet=None, reverse=False, ft=None):
        if not reverse:
            z = input
            assert z.shape[1] == self.in_channels, (z.shape[1], self.in_channels)

            # Feature Conditional
            scaleFt, shiftFt = self.feature_extract(ft, self.fFeatures)
            z = z + shiftFt
            z = z * scaleFt
            logdet = logdet + self.get_logdet(scaleFt)

            # Self Conditional
            z1, z2 = self.split(z)
            scale, shift = self.feature_extract_aff(z1, ft, self.fAffine)
            self.asserts(scale, shift, z1, z2)
            z2 = z2 + shift
            z2 = z2 * scale

            logdet = logdet + self.get_logdet(scale)
            z = thops.cat_feature(z1, z2)
            output = z
        else:
            z = input

            # Self Conditional
            z1, z2 = self.split(z)
            scale, shift = self.feature_extract_aff(z1, ft, self.fAffine)
            self.asserts(scale, shift, z1, z2)
            z2 = z2 / scale
            z2 = z2 - shift
            z = thops.cat_feature(z1, z2)
            logdet = logdet - self.get_logdet(scale)

            # Feature Conditional
            scaleFt, shiftFt = self.feature_extract(ft, self.fFeatures)
            z = z / scaleFt
            z = z - shiftFt
            logdet = logdet - self.get_logdet(scaleFt)

            output = z
        return output, logdet

    def asserts(self, scale, shift, z1, z2):
        assert z1.shape[1] == self.channels_for_nn, (z1.shape[1], self.channels_for_nn)
        assert z2.shape[1] == self.channels_for_co, (z2.shape[1], self.channels_for_co)
        assert scale.shape[1] == shift.shape[1], (scale.shape[1], shift.shape[1])
        assert scale.shape[1] == z2.shape[1], (scale.shape[1], z1.shape[1], z2.shape[1])

    def get_logdet(self, scale):
        # Sum over all spatial dims (works for 4D or 5D tensors)
        dims = list(range(1, scale.dim()))  # exclude batch dim
        return thops.sum(torch.log(scale), dim=dims)

    def feature_extract(self, z, f):
        h = f(z)
        shift, scale = thops.split_feature(h, "cross")
        scale = (torch.sigmoid(scale + 2.) + self.affine_eps)
        return scale, shift

    def feature_extract_aff(self, z1, ft, f):
        print(f"DEBUG feature_extract_aff: z1.shape={z1.shape}, ft.shape={ft.shape if ft is not None else None}")
        z = torch.cat([z1, ft], dim=1)
        print(f"DEBUG feature_extract_aff: concatenated z.shape={z.shape}")
        h = f(z)
        shift, scale = thops.split_feature(h, "cross")
        scale = (torch.sigmoid(scale + 2.) + self.affine_eps)
        return scale, shift

    def split(self, z):
        z1 = z[:, :self.channels_for_nn]
        z2 = z[:, self.channels_for_nn:]
        assert z1.shape[1] + z2.shape[1] == z.shape[1], (z1.shape[1], z2.shape[1], z.shape[1])
        return z1, z2

    def F(self, in_channels, out_channels, hidden_channels, kernel_hidden=1, n_hidden_layers=1):
        """Build small conditional subnet. Uses 3D convs (ConvND) which also work on 4D by treating depth=1 if provided.
        """
        print(f"Building F network: in_channels={in_channels}, out_channels={out_channels}, hidden_channels={hidden_channels}")
        layers = [ConvND(in_channels, hidden_channels)]
        layers.append(nn.ReLU(inplace=False))
        for _ in range(n_hidden_layers):
            # Build isotropic kernel for 3D; ConvND ignores extra dims if input 2D-shaped
            k = [kernel_hidden] * 3
            layers.append(ConvND(hidden_channels, hidden_channels, kernel_size=k))
            layers.append(nn.ReLU(inplace=False))
        layers.append(ConvNDZeros(hidden_channels, out_channels))
        return nn.Sequential(*layers)
