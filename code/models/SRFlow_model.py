# Built-in imports
import os
import logging
from collections import OrderedDict
from typing import Dict, Union, Any

# Set OpenMP environment variable
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Third-party imports
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel

# Local imports
from utils.util import get_resume_paths, opt_get
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel

# Initialize logger
logger = logging.getLogger('base')

class SRFlowModel(BaseModel):
    """SRFlow model for super-resolution using normalizing flows."""

    def __init__(self, opt: Dict[str, Any], step: int):
        super(SRFlowModel, self).__init__(opt)
        self.opt = opt

        # Validation parameters
        self.heats = opt_get(opt, ['datasets', 'val', 'heats']) or opt_get(opt, ['val', 'heats']) or [0.0, 0.5, 0.7, 0.9]
        self.n_sample = opt_get(opt, ['datasets', 'val', 'n_sample']) or opt_get(opt, ['val', 'n_sample']) or 4

        # Training parameters
        self.use_fp16 = opt_get(opt, ['datasets', 'train', 'use_fp16']) or False
        if self.use_fp16:
            self.scaler = torch.cuda.amp.GradScaler()

        # Model dimensions
        self.hr_size = opt_get(opt, ['datasets', 'train', 'center_crop_hr_size']) or 160
        self.lr_size = self.hr_size // opt['scale']

        # Device and distributed training setup
        self.rank = torch.distributed.get_rank() if opt.get('dist', False) else -1
        use_cuda = opt.get('gpu_ids', None) not in (None, [], [None]) and torch.cuda.is_available()
        self.device = torch.device('cuda' if use_cuda else 'cpu')

        # Initialize network
        self.netG = networks.define_Flow(opt, step).to(self.device)
        self._setup_parallel_training(use_cuda, opt)

        # Load pretrained model if specified
        if opt_get(opt, ['path', 'resume_state'], 1) is not None:
            self.load()
        else:
            logger.warning("Skipping initial loading, due to resume_state None")

        # Initialize training
        if self.is_train:
            self.netG.train()
            self.init_optimizer_and_scheduler(opt['train'])
            self.log_dict = OrderedDict()

    def _setup_parallel_training(self, use_cuda: bool, opt: Dict[str, Any]) -> None:
        """Setup parallel or distributed training."""
        if use_cuda:
            if opt.get('dist', False):
                self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
            else:
                self.netG = DataParallel(self.netG)
            self.netG_module = self.netG.module if hasattr(self.netG, 'module') else self.netG
        else:
            self.netG_module = self.netG

    def init_optimizer_and_scheduler(self, train_opt: Dict[str, Any]) -> None:
        """Initialize optimizer and scheduler for training."""
        self.optimizer_G = torch.optim.Adam(
            self.netG.parameters(),
            lr=train_opt.get('lr_G', 1e-4),
            betas=train_opt.get('betas_G', (0.9, 0.99))
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer_G,
            step_size=train_opt.get('lr_step', 50000),
            gamma=train_opt.get('lr_gamma', 0.5)
        )
        self.optimizers = [self.optimizer_G]

    def feed_data(self, data: Dict[str, torch.Tensor], need_GT: bool = True) -> None:
        """Feed data to the model."""
        if 'He' in data and 'H+' in data:
            self.var_L = data['He'].to(self.device)
            if need_GT:
                self.real_H = data['H+'].to(self.device)
        else:
            self.var_L = data['LQ'].to(self.device)
            if need_GT:
                self.real_H = data['GT'].to(self.device)

        # Fix for Conv3d input shape: ensure input is 5D [N, C, D, H, W]
        while self.var_L.dim() > 5:
            self.var_L = self.var_L.squeeze(0)
        if need_GT:
            while self.real_H.dim() > 5:
                self.real_H = self.real_H.squeeze(0)

    def optimize_parameters(self, step: int) -> None:
        """Optimize model parameters."""
        if self.use_fp16:
            with torch.cuda.amp.autocast():
                loss_dict = self.netG(gt=self.real_H, lr=self.var_L)  # Removed step=step
                loss = loss_dict['loss_total']
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer_G)
            self.scaler.update()
        else:
            loss_dict = self.netG(gt=self.real_H, lr=self.var_L)  # Removed step=step
            loss = loss_dict['loss_total']
            loss.backward()
            self.optimizer_G.step()

        self.optimizer_G.zero_grad()
        self.scheduler.step()

        for k, v in loss_dict.items():
            self.log_dict[k] = v.item() if isinstance(v, torch.Tensor) else v

    def test(self) -> None:
        """Test model (inference)."""
        self.netG.eval()
        with torch.no_grad():
            self.output = self.netG(lr=self.var_L, reverse=True)
        self.netG.train()

    def get_current_log(self) -> OrderedDict:
        """Get current training logs."""
        return self.log_dict

    def get_current_visuals(self) -> Dict[str, torch.Tensor]:
        """Get current output visuals."""
        out_dict = OrderedDict()
        out_dict['LQ'] = self.var_L.detach().float().cpu()
        out_dict['Output'] = self.output.detach().float().cpu()
        if hasattr(self, 'real_H'):
            out_dict['GT'] = self.real_H.detach().float().cpu()
        return out_dict

    def to(self, device: Union[str, torch.device]) -> None:
        """Move model to specified device with error handling."""
        try:
            self.device = device
            self.netG.to(device)
        except RuntimeError as e:
            logger.error(f"Failed to move model to {device}: {str(e)}")
            raise