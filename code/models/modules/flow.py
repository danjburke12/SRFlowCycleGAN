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
import torch.nn as nn
import numpy as np

from models.modules.FlowActNorms import ActNorm3d
from . import thops


class Conv3d(nn.Conv3d):
    pad_dict = {
        "same": lambda kernel, stride: [((k - 1) * s + 1) // 2 for k, s in zip(kernel, stride)],
        "valid": lambda kernel, stride: [0 for _ in kernel]
    }

    @staticmethod
    def get_padding(padding, kernel_size, stride):
        # make paddding
        if isinstance(padding, str):
            if isinstance(kernel_size, int):
                kernel_size = [kernel_size, kernel_size, kernel_size]
            if isinstance(stride, int):
                stride = [stride, stride, stride]
            padding = padding.lower()
            try:
                padding = Conv3d.pad_dict[padding](kernel_size, stride)
            except KeyError:
                raise ValueError("{} is not supported".format(padding))
        return padding

    def __init__(self, in_channels, out_channels,
                 kernel_size=[3, 3, 3], stride=[1, 1, 1],
                 padding="same", do_actnorm=True, weight_std=0.05):
        padding = Conv3d.get_padding(padding, kernel_size, stride)
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, bias=(not do_actnorm))
        # init weight with std
        self.weight.data.normal_(mean=0.0, std=weight_std)
        if not do_actnorm:
            self.bias.data.zero_()
        else:
            self.actnorm = ActNorm3d(out_channels)
        self.do_actnorm = do_actnorm

    def forward(self, input):
        x = super().forward(input)
        if self.do_actnorm:
            x, _ = self.actnorm(x)
        return x


class Conv3dZeros(nn.Conv3d):
    def __init__(self, in_channels, out_channels,
                 kernel_size=[3, 3, 3], stride=[1, 1, 1],
                 padding="same", logscale_factor=3):
        padding = Conv3d.get_padding(padding, kernel_size, stride)
        super().__init__(in_channels, out_channels, kernel_size, stride, padding)
        # logscale_factor
        self.logscale_factor = logscale_factor
        self.register_parameter("logs", nn.Parameter(torch.zeros(out_channels, 1, 1, 1)))
        # init
        self.weight.data.zero_()
        self.bias.data.zero_()

    def forward(self, input):
        output = super().forward(input)
        return output * torch.exp(self.logs * self.logscale_factor)


class GaussianDiag:
    Log2PI = float(np.log(2 * np.pi))

    @staticmethod
    def likelihood(mean, logs, x):
        """
        lnL = -1/2 * { ln|Var| + ((X - Mu)^T)(Var^-1)(X - Mu) + kln(2*PI) }
              k = 1 (Independent)
              Var = logs ** 2
        """
        if mean is None and logs is None:
            return -0.5 * (x ** 2 + GaussianDiag.Log2PI)
        else:
            return -0.5 * (logs * 2. + ((x - mean) ** 2) / torch.exp(logs * 2.) + GaussianDiag.Log2PI)

    @staticmethod
    def logp(mean, logs, x):
        likelihood = GaussianDiag.likelihood(mean, logs, x)
        return thops.sum(likelihood, dim=[1, 2, 3, 4])

    @staticmethod
    def sample(mean, logs, eps_std=None):
        eps_std = eps_std or 1
        eps = torch.normal(mean=torch.zeros_like(mean),
                           std=torch.ones_like(logs) * eps_std)
        return mean + torch.exp(logs) * eps

    @staticmethod
    def sample_eps(shape, eps_std, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        eps = torch.normal(mean=torch.zeros(shape),
                           std=torch.ones(shape) * eps_std)
        return eps


def squeeze3d(x, factor=2):
    """Squeeze 3D: (B,C,D,H,W)->(B,C*factor^3,D/f,H/f,W/f)."""
    assert factor >= 1 and isinstance(factor, int)
    if factor == 1:
        return x
    B, C, D, H, W = x.size()
    assert D % factor == 0 and H % factor == 0 and W % factor == 0, (D, H, W, factor)
    x = x.view(B, C, D // factor, factor, H // factor, factor, W // factor, factor)
    x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
    x = x.view(B, C * factor ** 3, D // factor, H // factor, W // factor)
    return x


def unsqueeze3d(x, factor=2):
    """Inverse of squeeze3d."""
    assert factor >= 1 and isinstance(factor, int)
    if factor == 1:
        return x
    B, C, D, H, W = x.size()
    factor3 = factor ** 3
    assert C % factor3 == 0, (C, factor)
    C_out = C // factor3
    x = x.view(B, C_out, factor, factor, factor, D, H, W)
    x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
    x = x.view(B, C_out, D * factor, H * factor, W * factor)
    return x


class SqueezeLayer(nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, x, logdet=None, reverse=False):
        if reverse:
            return unsqueeze3d(x, self.factor), logdet
        return squeeze3d(x, self.factor), logdet
