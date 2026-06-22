"""
Advanced data augmentation strategies for histopathology images
"""

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import numpy as np
import cv2
from PIL import Image
from typing import Tuple, Optional
import random


class StainAugmentation:
    """
    Stain augmentation for H&E histopathology images
    Based on Tellez et al. 2019 - "Quantifying the effects of data augmentation and stain color normalization"
    """
    def __init__(self, alpha_range: Tuple[float, float] = (0.7, 1.3),
                 beta_range: Tuple[float, float] = (-0.1, 0.1)):
        self.alpha_range = alpha_range
        self.beta_range = beta_range
        
        # Reference H&E matrix (from ImageJ color deconvolution)
        self.HE_ref = np.array([
            [0.644, 0.717, 0.267],  # Hematoxylin
            [0.093, 0.954, 0.283]   # Eosin
        ])
    
    def rgb_to_od(self, img: np.ndarray) -> np.ndarray:
        """Convert RGB to optical density"""
        img = np.maximum(img, 1e-6)
        return -np.log(img / 255.0)
    
    def od_to_rgb(self, od: np.ndarray) -> np.ndarray:
        """Convert optical density to RGB"""
        rgb = np.exp(-od) * 255
        return np.clip(rgb, 0, 255).astype(np.uint8)
    
    def __call__(self, img: Image.Image) -> Image.Image:
        """Apply stain augmentation"""
        img_np = np.array(img).astype(np.float32)
        h, w, c = img_np.shape
        
        # Convert to OD
        od = self.rgb_to_od(img_np)
        od = od.reshape(-1, 3)
        
        # Deconvolution
        try:
            # Normalize stain matrix
            HE_norm = self.HE_ref / np.linalg.norm(self.HE_ref, axis=1, keepdims=True)
            
            # Project to stain space
            stains = od @ HE_norm.T
            
            # Augment stain concentrations
            alpha_h = np.random.uniform(*self.alpha_range)
            alpha_e = np.random.uniform(*self.alpha_range)
            
            stains[:, 0] *= alpha_h  # Hematoxylin
            stains[:, 1] *= alpha_e  # Eosin
            
            # Add noise
            beta_h = np.random.uniform(*self.beta_range)
            beta_e = np.random.uniform(*self.beta_range)
            
            stains[:, 0] += beta_h
            stains[:, 1] += beta_e
            
            # Reconstruct
            od_aug = stains @ HE_norm
            od_aug = od_aug.reshape(h, w, c)
            
            # Convert back to RGB
            img_aug = self.od_to_rgb(od_aug)
            
            return Image.fromarray(img_aug)
        except:
            # If augmentation fails, return original
            return img


class ElasticTransform:
    """
    Elastic deformation for tissue-like augmentation
    """
    def __init__(self, alpha: float = 120, sigma: float = 12, p: float = 0.5):
        self.alpha = alpha
        self.sigma = sigma
        self.p = p
    
    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        
        img_np = np.array(img)
        h, w = img_np.shape[:2]
        
        # Generate random displacement fields
        dx = cv2.GaussianBlur(
            (np.random.rand(h, w) * 2 - 1), (0, 0), self.sigma
        ) * self.alpha
        dy = cv2.GaussianBlur(
            (np.random.rand(h, w) * 2 - 1), (0, 0), self.sigma
        ) * self.alpha
        
        # Create meshgrid
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        
        # Apply displacement
        map_x = (x + dx).astype(np.float32)
        map_y = (y + dy).astype(np.float32)
        
        # Remap
        img_warped = cv2.remap(
            img_np, map_x, map_y, 
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT
        )
        
        return Image.fromarray(img_warped)


class GridMask:
    """
    GridMask augmentation (Chen et al. 2020)
    """
    def __init__(self, ratio: float = 0.6, mode: int = 1, p: float = 0.3):
        self.ratio = ratio
        self.mode = mode  # 0: drop pixels, 1: drop regions
        self.p = p
    
    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        
        img_np = np.array(img)
        h, w = img_np.shape[:2]
        
        # Generate grid
        d = np.random.randint(h // 4, h // 2)
        l = int(d * self.ratio)
        
        mask = np.ones((h, w), dtype=np.float32)
        
        for i in range(0, h, d):
            for j in range(0, w, d):
                if self.mode == 0:
                    # Drop pixels in a grid pattern
                    mask[i:min(i+l, h), j:min(j+l, w)] = 0
                else:
                    # Randomly drop regions
                    if random.random() > 0.5:
                        mask[i:min(i+l, h), j:min(j+l, w)] = 0
        
        # Apply mask
        if len(img_np.shape) == 3:
            mask = mask[:, :, np.newaxis]
        
        img_masked = (img_np * mask).astype(np.uint8)
        
        return Image.fromarray(img_masked)


class MixUp:
    """
    MixUp augmentation (Zhang et al. 2018)
    Applied at batch level
    """
    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha
    
    def __call__(self, images: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Args:
            images: Batch of images [B, C, H, W]
            labels: Batch of labels [B]
        
        Returns:
            mixed_images, labels_a, labels_b, lambda
        """
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0
        
        batch_size = images.size(0)
        index = torch.randperm(batch_size).to(images.device)
        
        mixed_images = lam * images + (1 - lam) * images[index]
        labels_a, labels_b = labels, labels[index]
        
        return mixed_images, labels_a, labels_b, lam


class CutMix:
    """
    CutMix augmentation (Yun et al. 2019)
    Applied at batch level
    """
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
    
    def __call__(self, images: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Args:
            images: Batch of images [B, C, H, W]
            labels: Batch of labels [B]
        
        Returns:
            mixed_images, labels_a, labels_b, lambda
        """
        batch_size, _, h, w = images.shape
        
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0
        
        # Generate random box
        cut_ratio = np.sqrt(1.0 - lam)
        cut_h = int(h * cut_ratio)
        cut_w = int(w * cut_ratio)
        
        # Uniform sampling
        cx = np.random.randint(w)
        cy = np.random.randint(h)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)
        
        # Shuffle indices
        index = torch.randperm(batch_size).to(images.device)
        
        # Apply cutmix
        mixed_images = images.clone()
        mixed_images[:, :, bby1:bby2, bbx1:bbx2] = images[index, :, bby1:bby2, bbx1:bbx2]
        
        # Adjust lambda based on actual box size
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (h * w))
        
        labels_a, labels_b = labels, labels[index]
        
        return mixed_images, labels_a, labels_b, lam


class HistologyAugmentation:
    """
    Comprehensive augmentation pipeline for histopathology
    """
    def __init__(self, 
                 use_stain_aug: bool = True,
                 use_elastic: bool = True,
                 use_gridmask: bool = True,
                 image_size: int = 224):
        
        self.use_stain_aug = use_stain_aug
        self.use_elastic = use_elastic
        self.use_gridmask = use_gridmask
        
        # Base transforms
        self.base_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
        ])
        
        # Geometric augmentations
        self.geometric = [
            transforms.RandomRotation(180),  # Full rotation for histology
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
        ]
        
        # Color augmentations
        self.color = transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.02
        )
        
        # Advanced augmentations
        self.stain_aug = StainAugmentation() if use_stain_aug else None
        self.elastic = ElasticTransform() if use_elastic else None
        self.gridmask = GridMask() if use_gridmask else None
        
        # Normalization (ImageNet stats)
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def __call__(self, img: Image.Image) -> torch.Tensor:
        """Apply augmentation pipeline"""
        # Base resize
        img = self.base_transform(img)
        
        # Stain augmentation (before geometric)
        if self.stain_aug and random.random() > 0.5:
            img = self.stain_aug(img)
        
        # Elastic deformation
        if self.elastic:
            img = self.elastic(img)
        
        # Geometric augmentations
        for transform in self.geometric:
            img = transform(img)
        
        # Color augmentation
        if random.random() > 0.5:
            img = self.color(img)
        
        # GridMask
        if self.gridmask:
            img = self.gridmask(img)
        
        # Convert to tensor and normalize
        img = self.normalize(img)
        
        return img


class ValidationTransform:
    """Simple transform for validation (no augmentation)"""
    def __init__(self, image_size: int = 224):
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def __call__(self, img: Image.Image) -> torch.Tensor:
        return self.transform(img)


def get_train_transform(config) -> HistologyAugmentation:
    """Get training transform from config"""
    return HistologyAugmentation(
        use_stain_aug=config.data.use_stain_augmentation,
        use_elastic=config.data.use_advanced_augmentation,
        use_gridmask=config.data.use_advanced_augmentation,
        image_size=config.data.image_size
    )


def get_val_transform(config) -> ValidationTransform:
    """Get validation transform from config"""
    return ValidationTransform(image_size=config.data.image_size)
