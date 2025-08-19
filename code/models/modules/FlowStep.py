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

import models.modules
import models.modules.Permutations
from models.modules import flow, thops, FlowAffineCouplingsAblation
from models.modules.FlowActNorms import ActNorm2d, ActNorm3d
from utils.util import opt_get


def getConditional(rrdbResults, position):
    img_ft = rrdbResults if isinstance(rrdbResults, torch.Tensor) else rrdbResults[position]
    return img_ft


class FlowStep(nn.Module):
    FlowPermutation = {
        "reverse": lambda obj, z, logdet, rev: (obj.reverse(z, rev), logdet),
        "shuffle": lambda obj, z, logdet, rev: (obj.shuffle(z, rev), logdet),
        "invconv": lambda obj, z, logdet, rev: obj.invconv(z, logdet, rev),
        "squeeze_invconv": lambda obj, z, logdet, rev: obj.invconv(z, logdet, rev),
        "resqueeze_invconv_alternating_2_3": lambda obj, z, logdet, rev: obj.invconv(z, logdet, rev),
        "resqueeze_invconv_3": lambda obj, z, logdet, rev: obj.invconv(z, logdet, rev),
        "InvertibleConv1x1GridAlign": lambda obj, z, logdet, rev: obj.invconv(z, logdet, rev),
        "InvertibleConv1x1SubblocksShuf": lambda obj, z, logdet, rev: obj.invconv(z, logdet, rev),
        "InvertibleConv1x1GridAlignIndepBorder": lambda obj, z, logdet, rev: obj.invconv(z, logdet, rev),
        "InvertibleConv1x1GridAlignIndepBorder4": lambda obj, z, logdet, rev: obj.invconv(z, logdet, rev),
    }

    def __init__(self, in_channels, hidden_channels,
                 actnorm_scale=1.0, flow_permutation="invconv", flow_coupling="additive",
                 LU_decomposed=False, opt=None, image_injector=None, idx=None, acOpt=None, normOpt=None, in_shape=None,
                 position=None):
        # check configures
        assert flow_permutation in FlowStep.FlowPermutation, \
            "float_permutation should be in `{}`".format(
                FlowStep.FlowPermutation.keys())
        super().__init__()
        self.flow_permutation = flow_permutation
        self.flow_coupling = flow_coupling
        self.image_injector = image_injector

        self.norm_type = normOpt['type'] if normOpt else 'ActNorm3d'
        self.position = normOpt['position'] if normOpt else None

        self.in_shape = in_shape
        self.position = position
        self.acOpt = acOpt

        # 1. actnorm
        # Create ActNorm layers dynamically based on actual input dimensions during forward pass
        self.actnorm3d = None
        self.actnorm2d = None
        self.actnorm_scale = actnorm_scale

        # 2. permute
        if flow_permutation == "invconv":
            self.invconv = models.modules.Permutations.InvertibleConv1x1(
                in_channels, LU_decomposed=LU_decomposed)

        # 3. coupling
        if flow_coupling == "CondAffineSeparatedAndCond":
            self.affine = models.modules.FlowAffineCouplingsAblation.CondAffineSeparatedAndCond(in_channels=in_channels,
                                                                                                opt=opt)
        elif flow_coupling == "noCoupling":
            pass
        else:
            raise RuntimeError("coupling not Found:", flow_coupling)

    def forward(self, input, logdet=None, reverse=False, rrdbResults=None):
        print(f"DEBUG FlowStep forward: input.shape={input.shape}, reverse={reverse}")
        if not reverse:
            return self.normal_flow(input, logdet, rrdbResults)
        else:
            return self.reverse_flow(input, logdet, rrdbResults)

    def normal_flow(self, z, logdet, rrdbResults=None):
        print(f"DEBUG FlowStep normal_flow: z.shape={z.shape}")
        
        if self.flow_coupling == "bentIdentityPreAct":
            z, logdet = self.bentIdentPar(z, logdet, reverse=False)

        # 1. actnorm
        if self.norm_type == "ConditionalActNormImageInjector":
            img_ft = getConditional(rrdbResults, self.position)
            actnorm_module = self._select_actnorm(z)
            z, logdet = actnorm_module(z, logdet=logdet, reverse=False)
        elif self.norm_type == "noNorm":
            pass
        else:
            actnorm_module = self._select_actnorm(z)
            z, logdet = actnorm_module(z, logdet=logdet, reverse=False)

        print(f"DEBUG FlowStep normal_flow after actnorm: z.shape={z.shape}")

        # 2. permute
        z, logdet = FlowStep.FlowPermutation[self.flow_permutation](
            self, z, logdet, False)

        print(f"DEBUG FlowStep normal_flow after permutation: z.shape={z.shape}")

        need_features = self.affine_need_features()

        # 3. coupling
        if need_features or self.flow_coupling in ["condAffine", "condFtAffine", "condNormAffine"]:
            img_ft = getConditional(rrdbResults, self.position)
            print(f"DEBUG FlowStep normal_flow coupling: z.shape={z.shape}, img_ft.shape={img_ft.shape if img_ft is not None else None}")
            z, logdet = self.affine(input=z, logdet=logdet, reverse=False, ft=img_ft)
        return z, logdet

    def reverse_flow(self, z, logdet, rrdbResults=None):

        need_features = self.affine_need_features()

        # 1.coupling
        if need_features or self.flow_coupling in ["condAffine", "condFtAffine", "condNormAffine"]:
            img_ft = getConditional(rrdbResults, self.position)
            z, logdet = self.affine(input=z, logdet=logdet, reverse=True, ft=img_ft)

        # 2. permute
        z, logdet = FlowStep.FlowPermutation[self.flow_permutation](
            self, z, logdet, True)

        # 3. actnorm
        actnorm_module = self._select_actnorm(z)
        z, logdet = actnorm_module(z, logdet=logdet, reverse=True)

        return z, logdet

    def affine_need_features(self):
        need_features = False
        try:
            need_features = self.affine.need_features
        except:
            pass
        return need_features

    def _select_actnorm(self, z):
        """Dynamically create and select ActNorm based on input dimensions."""
        if z.dim() == 5:  # 3D tensor (B,C,D,H,W)
            if self.actnorm3d is None:
                self.actnorm3d = ActNorm3d(z.size(1), self.actnorm_scale)
                # Move to correct device
                if z.is_cuda:
                    self.actnorm3d = self.actnorm3d.cuda()
            return self.actnorm3d
        else:  # 2D tensor (B,C,H,W)
            if self.actnorm2d is None:
                self.actnorm2d = ActNorm2d(z.size(1), self.actnorm_scale)
                # Move to correct device  
                if z.is_cuda:
                    self.actnorm2d = self.actnorm2d.cuda()
            return self.actnorm2d
