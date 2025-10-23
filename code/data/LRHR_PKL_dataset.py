# Copyright (c) 2020 Huawei Technologies Co., Ltd.
# Licensed under CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International) (the "License");
# you may not use this file except in compliance with the Li            # Convert to PyTorch tensors
            # Move time dimension to be the channel dimension for the network
            hr = torch.Tensor(np.transpose(hr, (0, 3, 1, 2)))  # [1,T,H,W]
            lr = torch.Tensor(np.transpose(lr, (0, 3, 1, 2)))  # [1,T,H,W]
            
            # Squeeze batch dimension since DataLoader will add it back
            hr = hr.squeeze(0)  # [T,H,W]
            lr = lr.squeeze(0)  # [T,H,W]

            return {'LQ': lr, 'GT': hr, 'LQ_path': str(item), 'GT_path': str(item)}.
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
# This file contains content licensed by https://github.com/xinntao/BasicSR/blob/master/LICENSE/LICENSE

import os
import subprocess
import torch.utils.data as data
import numpy as np
import time
import torch

import pickle


class LRHR_PKLDataset(data.Dataset):
    def __init__(self, opt):
        super(LRHR_PKLDataset, self).__init__()
        self.opt = opt
        self.crop_size = opt.get("GT_size", None)
        self.scale = None
        self.random_scale_list = [1]

        hr_file_path = opt["dataroot_GT"]
        lr_file_path = opt["dataroot_LQ"]
        y_labels_file_path = opt['dataroot_y_labels']

        gpu = True
        augment = True

        self.use_flip = opt["use_flip"] if "use_flip" in opt.keys() else False
        self.use_rot = opt["use_rot"] if "use_rot" in opt.keys() else False
        self.use_crop = opt["use_crop"] if "use_crop" in opt.keys() else False
        self.center_crop_hr_size = opt.get("center_crop_hr_size", None)

        n_max = opt["n_max"] if "n_max" in opt.keys() else int(1e8)

        t = time.time()
        self.lr_images = self.load_pkls(lr_file_path, n_max)
        self.hr_images = self.load_pkls(hr_file_path, n_max)

        min_val_hr = np.min([i.min() for i in self.hr_images[:20]])
        max_val_hr = np.max([i.max() for i in self.hr_images[:20]])

        min_val_lr = np.min([i.min() for i in self.lr_images[:20]])
        max_val_lr = np.max([i.max() for i in self.lr_images[:20]])

        t = time.time() - t
        print("Loaded {} HR images with [{:.2f}, {:.2f}] in {:.2f}s from {}".
              format(len(self.hr_images), min_val_hr, max_val_hr, t, hr_file_path))
        print("Loaded {} LR images with [{:.2f}, {:.2f}] in {:.2f}s from {}".
              format(len(self.lr_images), min_val_lr, max_val_lr, t, lr_file_path))

        self.gpu = gpu
        self.augment = augment

        self.measures = None

    def load_pkls(self, path, n_max):
        """Load and standardize images from pickle file.
        
        All images are converted to NHWC format with shape [1,H,W,1] and float32 dtype.
        This ensures consistent format before any processing is done.
        
        Args:
            path: Path to pickle file containing images
            n_max: Maximum number of images to load
            
        Returns:
            List of numpy arrays in NHWC format [1,H,W,1]
        """
        assert os.path.isfile(path), f"File not found: {path}"
        with open(path, "rb") as f:
            images = pickle.load(f)
        assert len(images) > 0, f"No images found in {path}"
        images = images[:n_max]
        
        processed_images = []
        for idx, img in enumerate(images):
            try:
                # Convert to numpy array if needed
                if isinstance(img, torch.Tensor):
                    img = img.cpu().numpy()
                
                original_shape = img.shape
                print(f"Processing image {idx}, original shape: {original_shape}")
                
                # Standardize input to NHWC format [1,H,W,1]
                if len(img.shape) == 2:  # H,W
                    img = img[np.newaxis, :, :, np.newaxis]  # -> [1,H,W,1]
                    
                elif len(img.shape) == 3:  # H,W,C or C,H,W
                    if img.shape[-1] == 1:  # H,W,1
                        img = img[np.newaxis, ...]  # -> [1,H,W,1]
                    elif img.shape[0] == 1:  # 1,H,W or C,H,W
                        img = np.transpose(img, (1, 2, 0))  # -> H,W,C
                        img = img[..., 0:1]  # Take first channel
                        img = img[np.newaxis, ...]  # -> [1,H,W,1]
                    else:  # Assume C,H,W if not H,W,C
                        img = np.transpose(img, (1, 2, 0))  # -> H,W,C
                        img = img[..., 0:1]  # Take first channel
                        img = img[np.newaxis, ...]  # -> [1,H,W,1]
                        
                elif len(img.shape) == 4:  # N,C,H,W or N,H,W,C
                    if img.shape[1] in [1, 3]:  # N,C,H,W
                        img = np.transpose(img, (0, 2, 3, 1))  # -> N,H,W,C
                    # Now in N,H,W,C format
                    img = img[0:1, ..., 0:1]  # Take first sample and channel -> [1,H,W,1]
                
                # Ensure float32 dtype
                img = img.astype(np.float32)
                
                # Verify final shape
                assert len(img.shape) == 4, f"Expected NHWC format [1,H,W,1], got shape {img.shape}"
                assert img.shape[0] == 1, f"Expected batch size 1, got {img.shape[0]}"
                assert img.shape[-1] == 1, f"Expected single channel, got {img.shape[-1]}"
                
                print(f"Processed image {idx} to shape: {img.shape}")
                processed_images.append(img)
                
            except Exception as e:
                print(f"Error processing image {idx} with shape {original_shape}: {str(e)}")
                raise
        
        return processed_images

    def __len__(self):
        return len(self.hr_images)

    def __getitem__(self, item):
        """Get a pair of LR and HR images.
        
        The data pipeline is:
        1. Load from storage (NHWC format [1,H,W,1])
        2. Convert to NCHW format [1,1,H,W] for processing
        3. Apply augmentations (maintaining NCHW format)
        4. Remove batch dimension and normalize to [1,H,W]
        5. Convert to torch tensors
        
        Returns:
            dict: Contains 'LQ' and 'GT' tensors in CHW format [1,H,W]
        """
        try:
            print(f"\nProcessing item {item}")
            
            # 1. Get images from storage (NHWC format)
            hr = self.hr_images[item]  # Shape: [1,H,W,1]
            lr = self.lr_images[item]  # Shape: [1,H,W,1]
            
            print(f"Initial shapes - HR: {hr.shape}, LR: {lr.shape}")
            print(f"Data types - HR: {hr.dtype}, LR: {lr.dtype}")
            print(f"Value ranges - HR: [{hr.min():.2f}, {hr.max():.2f}], LR: [{lr.min():.2f}, {lr.max():.2f}]")
            
            # Verify input shapes
            assert hr.shape[0] == 1 and hr.shape[-1] == 1, f"HR should be [1,H,W,1], got {hr.shape}"
            assert lr.shape[0] == 1 and lr.shape[-1] == 1, f"LR should be [1,H,W,1], got {lr.shape}"
            
            # 2. Convert NHWC -> NCHW for processing
            hr = np.transpose(hr, (0, 3, 1, 2))  # [1,H,W,1] -> [1,1,H,W]
            lr = np.transpose(lr, (0, 3, 1, 2))  # [1,H,W,1] -> [1,1,H,W]
            
            print(f"After NCHW conversion - HR: {hr.shape}, LR: {lr.shape}")
            
            # Verify shapes after transpose
            assert hr.shape[1] == 1 and lr.shape[1] == 1, f"Channel dimension should be 1, got HR: {hr.shape[1]}, LR: {lr.shape[1]}"
        
            # 3. Determine and verify scale factor
            if self.scale is None:
                self.scale = hr.shape[2] // lr.shape[2]
                assert hr.shape[2] == self.scale * lr.shape[2], (
                    f'Non-fractional scale ratio - HR shape: {hr.shape}, LR shape: {lr.shape}, '
                    f'Computed scale: {self.scale}'
                )
                print(f"Scale factor set to: {self.scale}")
            
            # 4. Apply augmentations (all expect NCHW format)
            if self.use_crop:
                hr, lr = random_crop(hr, lr, self.crop_size, self.scale, self.use_crop)
                print(f"After crop - HR: {hr.shape}, LR: {lr.shape}")
                
            if self.center_crop_hr_size:
                hr = center_crop(hr, self.center_crop_hr_size)
                lr = center_crop(lr, self.center_crop_hr_size // self.scale)
                print(f"After center crop - HR: {hr.shape}, LR: {lr.shape}")
                
            if self.use_flip:
                hr, lr = random_flip(hr, lr)
                
            if self.use_rot:
                hr, lr = random_rotation(hr, lr)
            
            # 5. Remove batch dimension and normalize
            hr = hr[0] / 255.0  # [1,1,H,W] -> [1,H,W]
            lr = lr[0] / 255.0  # [1,1,H,W] -> [1,H,W]
            
            print(f"Final shapes - HR: {hr.shape}, LR: {lr.shape}")
            print(f"Normalized ranges - HR: [{hr.min():.3f}, {hr.max():.3f}], LR: [{lr.min():.3f}, {lr.max():.3f}]")
            
            # Calculate statistics if needed
            if self.measures is None or np.random.random() < 0.05:
                if self.measures is None:
                    self.measures = {}
                self.measures['hr_means'] = float(np.mean(hr))
                self.measures['hr_stds'] = float(np.std(hr))
                self.measures['lr_means'] = float(np.mean(lr))
                self.measures['lr_stds'] = float(np.std(lr))
                print(f"Statistics - HR mean: {self.measures['hr_means']:.3f}, std: {self.measures['hr_stds']:.3f}")
                print(f"Statistics - LR mean: {self.measures['lr_means']:.3f}, std: {self.measures['lr_stds']:.3f}")

            # 6. Convert to PyTorch tensors
            hr = torch.Tensor(hr)  # Shape: [1,H,W]
            lr = torch.Tensor(lr)  # Shape: [1,H,W]

            return {'LQ': lr, 'GT': hr, 'LQ_path': str(item), 'GT_path': str(item)}
            
        except Exception as e:
            print(f"Error processing item {item}: {str(e)}")
            raise

    def print_and_reset(self, tag):
        m = self.measures
        kvs = []
        for k in sorted(m.keys()):
            kvs.append("{}={:.2f}".format(k, m[k]))
        print("[KPI] " + tag + ": " + ", ".join(kvs))
        self.measures = None


def random_flip(img, seg):
    random_choice = np.random.choice([True, False])
    # Assuming NCHW format, flip along the last dimension (W)
    img = img if random_choice else np.flip(img, 3).copy()
    seg = seg if random_choice else np.flip(seg, 3).copy()
    return img, seg


def random_rotation(img, seg):
    random_choice = np.random.choice([0, 1, 3])
    # For NCHW format, rotate in the H,W dimensions (2,3)
    img = np.rot90(img, random_choice, axes=(2, 3)).copy()
    seg = np.rot90(seg, random_choice, axes=(2, 3)).copy()
    return img, seg


def random_crop(hr, lr, size_hr, scale, random):
    # Ensure inputs are NCHW format
    size_lr = size_hr // scale

    size_lr_x = lr.shape[2]  # H dimension
    size_lr_y = lr.shape[3]  # W dimension

    start_x_lr = np.random.randint(low=0, high=(size_lr_x - size_lr) + 1) if size_lr_x > size_lr else 0
    start_y_lr = np.random.randint(low=0, high=(size_lr_y - size_lr) + 1) if size_lr_y > size_lr else 0

    # LR Patch - shape should be [N, C, H, W]
    lr_patch = lr[:, :, start_x_lr:start_x_lr + size_lr, start_y_lr:start_y_lr + size_lr]

    # HR Patch
    start_x_hr = start_x_lr * scale
    start_y_hr = start_y_lr * scale
    hr_patch = hr[:, :, start_x_hr:start_x_hr + size_hr, start_y_hr:start_y_hr + size_hr]

    return hr_patch, lr_patch


def center_crop(img, size):
    assert img.shape[2] == img.shape[3], img.shape  # Height and width should be equal
    border_double = img.shape[2] - size  # Use height dimension
    assert border_double % 2 == 0, (img.shape, size)
    border = border_double // 2
    return img[:, :, border:-border, border:-border]  # Preserve batch and channel dims


def center_crop_tensor(img, size):
    assert img.shape[2] == img.shape[3], img.shape
    border_double = img.shape[2] - size
    assert border_double % 2 == 0, (img.shape, size)
    border = border_double // 2
    return img[:, :, border:-border, border:-border]
