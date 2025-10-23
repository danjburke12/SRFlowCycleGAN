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
        """Optimize networks with all losses."""
        self.netG_Y.train()
        self.netG_X.train()
        
        losses = {}
        
        # Get loss weights from options
        weight_nll = opt_get(self.opt, ['train', 'weight_fl'], 1.0)      # NLL weight
        weight_cycle = opt_get(self.opt, ['train', 'weight_cycle'], 10.0) # Cycle consistency
        weight_pair = opt_get(self.opt, ['train', 'weight_pair'], 5.0)    # Paired supervision 
        weight_align = opt_get(self.opt, ['train', 'weight_align'], 1.0)  # Latent alignment
        
        # 1. Negative Log-Likelihood Loss (per domain)
        if weight_nll > 0:
            # He -> H+ direction
            z_y, nll_y, _ = self.netG_Y(gt=self.var_Hp, lr=self.var_He, reverse=False)
            nll_loss_y = torch.mean(nll_y)
            
            # H+ -> He direction  
            z_x, nll_x, _ = self.netG_X(gt=self.var_He, lr=self.var_Hp, reverse=False)
            nll_loss_x = torch.mean(nll_x)
            
            losses['nll_loss'] = (nll_loss_y + nll_loss_x) * weight_nll
            
        # 2. Cycle Consistency Loss
        if weight_cycle > 0:
            # He -> H+ -> He
            z1 = self.get_z(heat=0, seed=None, batch_size=self.var_He.shape[0], lr_shape=self.var_He.shape)
            hp_gen, _ = self.netG_Y(lr=self.var_He, z=z1, eps_std=0, reverse=True)
            z2 = self.get_z(heat=0, seed=None, batch_size=self.var_He.shape[0], lr_shape=self.var_He.shape) 
            he_cyc, _ = self.netG_X(lr=hp_gen, z=z2, eps_std=0, reverse=True)
            
            # H+ -> He -> H+
            z3 = self.get_z(heat=0, seed=None, batch_size=self.var_Hp.shape[0], lr_shape=self.var_Hp.shape)
            he_gen, _ = self.netG_X(lr=self.var_Hp, z=z3, eps_std=0, reverse=True)
            z4 = self.get_z(heat=0, seed=None, batch_size=self.var_Hp.shape[0], lr_shape=self.var_Hp.shape)
            hp_cyc, _ = self.netG_Y(lr=he_gen, z=z4, eps_std=0, reverse=True)
            
            cycle_loss = (he_cyc - self.var_He).abs().mean() + (hp_cyc - self.var_Hp).abs().mean()
            losses['cycle_loss'] = cycle_loss * weight_cycle
            
        # 3. Supervised Cross-Domain Loss (paired data)
        if weight_pair > 0:
            # He -> H+ direct mapping
            z5 = self.get_z(heat=0, seed=None, batch_size=self.var_He.shape[0], lr_shape=self.var_He.shape)
            hp_direct, _ = self.netG_Y(lr=self.var_He, z=z5, eps_std=0, reverse=True)
            
            # H+ -> He direct mapping
            z6 = self.get_z(heat=0, seed=None, batch_size=self.var_Hp.shape[0], lr_shape=self.var_Hp.shape)
            he_direct, _ = self.netG_X(lr=self.var_Hp, z=z6, eps_std=0, reverse=True)
            
            pair_loss = (hp_direct - self.var_Hp).abs().mean() + (he_direct - self.var_He).abs().mean()
            losses['pair_loss'] = pair_loss * weight_pair
            
        # 4. Latent Space Alignment Loss
        if weight_align > 0:
            # Get latent codes for matched pairs
            z_he, _, _ = self.netG_X(gt=self.var_He, lr=self.var_Hp, reverse=False)
            z_hp, _, _ = self.netG_Y(gt=self.var_Hp, lr=self.var_He, reverse=False)
            
            # MSE between matched latent codes
            align_loss = (z_he - z_hp).pow(2).mean()
            losses['align_loss'] = align_loss * weight_align
            
        # Combined loss
        total_loss = sum(losses.values())
        
        # Update both networks
        self.optimizer_G_Y.zero_grad()
        self.optimizer_G_X.zero_grad()
        total_loss.backward()
        self.optimizer_G_Y.step()
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
