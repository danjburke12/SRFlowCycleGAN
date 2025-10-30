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

import logging
from collections import OrderedDict
import torch
import torch.nn as nn
from .SRFlow_model import SRFlowModel
from utils.util import opt_get

logger = logging.getLogger('base')

class SRFlowBidirectionalModel(SRFlowModel):
    """Bidirectional SRFlow model for mapping between two domains X and Y."""
    
    def __init__(self, opt, step):
        super().__init__(opt, step)
        self.opt = opt
        
        # Initialize second flow network for the other direction
        self.netG_Y = self.netG  # Rename first network for clarity
        self.netG_X = networks.define_Flow(opt, step).to(self.device)  # Create second network
        
        if opt['dist']:
            self.netG_X = DistributedDataParallel(self.netG_X, device_ids=[torch.cuda.current_device()])
        else:
            self.netG_X = DataParallel(self.netG_X)
            
        # Initialize optimizers for both networks
        self.init_optimizers(opt['train'])
        
    def init_optimizers(self, train_opt):
        """Initialize optimizers for both networks."""
        self.optimizers = []
        wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
        
        # Optimizer for G_Y (He -> H+)
        optim_params_Y = []
        for k, v in self.netG_Y.named_parameters():
            if v.requires_grad:
                optim_params_Y.append(v)
                
        # Optimizer for G_X (H+ -> He) 
        optim_params_X = []
        for k, v in self.netG_X.named_parameters():
            if v.requires_grad:
                optim_params_X.append(v)
                
        # Create optimizers
        self.optimizer_G_Y = torch.optim.Adam(
            optim_params_Y,
            lr=train_opt['lr_G'],
            betas=(train_opt['beta1'], train_opt['beta2']),
            weight_decay=wd_G
        )
        
        self.optimizer_G_X = torch.optim.Adam(
            optim_params_X,
            lr=train_opt['lr_G'],
            betas=(train_opt['beta1'], train_opt['beta2']),
            weight_decay=wd_G
        )
        
        self.optimizers.extend([self.optimizer_G_Y, self.optimizer_G_X])
        
    def feed_data(self, data):
        """Feed both He and H+ data."""
        self.var_He = data['He'].to(self.device)  # He input
        self.var_Hp = data['H+'].to(self.device)  # H+ input
        
        # Handle dimensionality
        for tensor in [self.var_He, self.var_Hp]:
            if tensor.dim() == 6:  # If shape is [1, 1, 1, D, H, W]
                tensor = tensor.squeeze(2)  # Convert to [1, 1, D, H, W]
                
    def optimize_parameters(self, step):
        """Optimize networks with losses for one direction per iteration to reduce memory usage."""
        self.netG_Y.train()
        self.netG_X.train()

        losses = {}

        # Get loss weights from options
        weight_nll = opt_get(self.opt, ['train', 'weight_fl'], 1.0)
        weight_pair = opt_get(self.opt, ['train', 'weight_pair'], 5.0)

        # Alternate directions: even step = He->H+, odd step = H+->He
        if step % 2 == 0:
            # He -> H+ direction
            z_y, nll_y, _ = self.netG_Y(gt=self.var_Hp, lr=self.var_He, reverse=False)
            if weight_nll > 0:
                losses['nll_loss'] = torch.mean(nll_y) * weight_nll
            if weight_pair > 0:
                hp_direct = self.netG_Y(lr=self.var_He, z=None, eps_std=0, reverse=True)[0]
                pair_loss = (hp_direct - self.var_Hp).abs().mean()
                losses['pair_loss'] = pair_loss * weight_pair
            total_loss = sum(losses.values())
            self.optimizer_G_Y.zero_grad()
            total_loss.backward()
            self.optimizer_G_Y.step()
        else:
            # H+ -> He direction
            z_x, nll_x, _ = self.netG_X(gt=self.var_He, lr=self.var_Hp, reverse=False)
            if weight_nll > 0:
                losses['nll_loss'] = torch.mean(nll_x) * weight_nll
            if weight_pair > 0:
                he_direct = self.netG_X(lr=self.var_Hp, z=None, eps_std=0, reverse=True)[0]
                pair_loss = (he_direct - self.var_He).abs().mean()
                losses['pair_loss'] = pair_loss * weight_pair
            total_loss = sum(losses.values())
            self.optimizer_G_X.zero_grad()
            total_loss.backward()
            self.optimizer_G_X.step()
        self.log_dict = losses
        return total_loss.item()
        
    def test(self):
        self.netG_Y.eval()
        self.netG_X.eval()
        
        self.fake_Hp = {}
        self.fake_He = {}
        
        # Generate samples at different temperatures
        for heat in self.heats:
            for i in range(self.n_sample):
                # He -> H+
                z = self.get_z(heat, seed=None, batch_size=self.var_He.shape[0], lr_shape=self.var_He.shape)
                with torch.no_grad():
                    self.fake_Hp[(heat, i)], _ = self.netG_Y(lr=self.var_He, z=z, eps_std=heat, reverse=True)
                    
                # H+ -> He
                z = self.get_z(heat, seed=None, batch_size=self.var_Hp.shape[0], lr_shape=self.var_Hp.shape)
                with torch.no_grad():
                    self.fake_He[(heat, i)], _ = self.netG_X(lr=self.var_Hp, z=z, eps_std=heat, reverse=True)
                    
        # Compute NLL for both directions
        with torch.no_grad():
            _, nll_y, _ = self.netG_Y(gt=self.var_Hp, lr=self.var_He, reverse=False)
            _, nll_x, _ = self.netG_X(gt=self.var_He, lr=self.var_Hp, reverse=False)
            
        self.netG_Y.train()
        self.netG_X.train()
        
        return (nll_y.mean() + nll_x.mean()).item() / 2
        
    def get_current_visuals(self):
        out_dict = OrderedDict()
        # Original inputs
        out_dict['He'] = self.var_He.detach()[0].float().cpu()
        out_dict['H+'] = self.var_Hp.detach()[0].float().cpu()
        
        # Generated samples at different temperatures
        for heat in self.heats:
            for i in range(self.n_sample):
                out_dict[('He->H+', heat, i)] = self.fake_Hp[(heat, i)].detach()[0].float().cpu()
                out_dict[('H+->He', heat, i)] = self.fake_He[(heat, i)].detach()[0].float().cpu()
                
        return out_dict
        
    def save(self, iter_label):
        self.save_network(self.netG_Y, 'G_Y', iter_label)
        self.save_network(self.netG_X, 'G_X', iter_label)
