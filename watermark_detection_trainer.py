"""
Production-grade watermark detection training system


Key features:
1. Smart dataset processing (supports two folder structures)
2. Adaptive loss function (Focal Loss + Dice Loss + weight balancing)
3. Foreground ratio monitoring and automatic adjustment
4. Full MPS/CUDA/CPU platform support
5. Advanced data augmentation and quality control
6. Real-time performance monitoring and automatic parameter tuning


"""

import os
import sys
import json
import time
import random
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
import torchvision.models as models

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
import albumentations as A
from albumentations.pytorch import ToTensorV2

warnings.filterwarnings('ignore')

# ==================== Device Manager ====================

class SmartDeviceManager:
    """Smart device manager with MPS support"""
    
    def __init__(self, preferred_device: str = "auto"):
        self.device = self._detect_optimal_device(preferred_device)
        self.device_info = self._get_device_info()
        self.memory_fraction = 0.85  # GPU memory usage fraction
        self._configure_device()
        
        logging.info(f"🚀 Device selected: {self.device}")
        logging.info(f"📊 Device info: {self.device_info}")
    
    def _detect_optimal_device(self, preferred: str) -> torch.device:
        """Select the optimal device"""
        if preferred == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        elif preferred == "mps" and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")
        elif preferred == "cpu":
            return torch.device("cpu")
        elif preferred == "auto":
            # Automatically select the best available device
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        else:
            logging.warning(f"Device {preferred} not available, falling back to CPU")
            return torch.device("cpu")
    
    def _get_device_info(self) -> Dict[str, Any]:
        """Get detailed device information"""
        info = {"device_type": str(self.device)}
        
        if self.device.type == "cuda":
            info.update({
                "name": torch.cuda.get_device_name(),
                "total_memory_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3),
                "compute_capability": f"{torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}",
                "multi_processor_count": torch.cuda.get_device_properties(0).multi_processor_count
            })
        elif self.device.type == "mps":
            info.update({
                "name": "Apple Silicon (MPS)",
                "unified_memory": True,
                "optimization": "MPS Backend"
            })
        else:
            import psutil
            info.update({
                "name": "CPU",
                "cores": psutil.cpu_count(logical=False),
                "threads": psutil.cpu_count(logical=True),
                "memory_gb": psutil.virtual_memory().total / (1024**3)
            })
        
        return info
    
    def _configure_device(self):
        """Configure device-specific optimizations"""
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            # Set GPU memory allocation strategy
            torch.cuda.empty_cache()
        elif self.device.type == "mps":
            # MPS optimization settings
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
            os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    
    def get_optimal_batch_size(self, base_batch_size: int = 16) -> int:
        """Compute the optimal batch size for the device"""
        if self.device.type == "cuda":
            memory_gb = self.device_info.get("total_memory_gb", 4)
            if memory_gb >= 24:
                return base_batch_size * 2
            elif memory_gb >= 12:
                return base_batch_size
            elif memory_gb >= 8:
                return max(4, base_batch_size // 2)
            else:
                return 4
        elif self.device.type == "mps":
            # MPS usually has ample memory but relatively slower compute
            return max(4, base_batch_size // 2)
        else:
            return 2  # Conservative setting for CPU

# ==================== Advanced Loss Functions ====================

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        
        # Compute alpha weights
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Focal weight
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        focal_loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class DiceLoss(nn.Module):
    """Dice Loss for segmentation tasks"""
    
    def __init__(self, smooth: float = 1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Apply sigmoid to inputs
        inputs = torch.sigmoid(inputs)
        
        # Flatten tensors
        inputs_flat = inputs.view(-1)
        targets_flat = targets.view(-1)
        
        # Compute Dice coefficient
        intersection = (inputs_flat * targets_flat).sum()
        dice_coeff = (2.0 * intersection + self.smooth) / (
            inputs_flat.sum() + targets_flat.sum() + self.smooth
        )
        
        return 1 - dice_coeff

class AdaptiveLossFunction(nn.Module):
    """Adaptive loss function that adjusts weights based on foreground ratio"""
    
    def __init__(self, base_pos_weight: float = 6.0):
        super(AdaptiveLossFunction, self).__init__()
        self.base_pos_weight = base_pos_weight
        self.focal_loss = FocalLoss(alpha=0.75, gamma=2.0)
        self.dice_loss = DiceLoss(smooth=1e-6)
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        
        # Track statistics
        self.fg_ratios = []
        self.loss_weights = {
            'focal': 2.0,
            'dice': 2.0, 
            'bce': 1.0
        }
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size = inputs.size(0)
        
        # Compute foreground ratio for each sample
        fg_ratios = []
        adaptive_losses = []
        
        for i in range(batch_size):
            single_input = inputs[i:i+1]
            single_target = targets[i:i+1]
            
            # Compute foreground ratio
            fg_ratio = single_target.mean().item()
            fg_ratios.append(fg_ratio)
            
            # Adaptive weights
            if fg_ratio < 0.001:  # Tiny foreground
                pos_weight = self.base_pos_weight * 3.0
                focal_weight = 3.0
                dice_weight = 1.0
            elif fg_ratio < 0.01:  # Small foreground
                pos_weight = self.base_pos_weight * 2.0
                focal_weight = 2.5
                dice_weight = 2.0
            elif fg_ratio < 0.05:  # Medium foreground
                pos_weight = self.base_pos_weight
                focal_weight = 2.0
                dice_weight = 2.0
            else:  # Large foreground
                pos_weight = self.base_pos_weight * 0.5
                focal_weight = 1.5
                dice_weight = 2.5
            
            # Compute individual losses
            focal_loss = self.focal_loss(single_input, single_target)
            dice_loss = self.dice_loss(single_input, single_target)
            
            # Weighted BCE
            bce_weights = torch.ones_like(single_target)
            bce_weights[single_target == 1] = pos_weight
            bce_loss = (self.bce_loss(single_input, single_target) * bce_weights).mean()
            
            # Combined loss
            combined_loss = (
                focal_weight * focal_loss +
                dice_weight * dice_loss +
                1.0 * bce_loss
            )
            adaptive_losses.append(combined_loss)
        
        # Record foreground ratios for monitoring
        self.fg_ratios.extend(fg_ratios)
        
        # Compute mean loss
        total_loss = torch.stack(adaptive_losses).mean()
        avg_fg_ratio = sum(fg_ratios) / len(fg_ratios)
        
        return {
            'total_loss': total_loss,
            'avg_fg_ratio': torch.tensor(avg_fg_ratio),
            'focal_loss': torch.stack([self.focal_loss(inputs[i:i+1], targets[i:i+1]) for i in range(batch_size)]).mean(),
            'dice_loss': torch.stack([self.dice_loss(inputs[i:i+1], targets[i:i+1]) for i in range(batch_size)]).mean()
        }
    
    def get_fg_ratio_stats(self) -> Dict[str, float]:
        """Get foreground ratio statistics"""
        if not self.fg_ratios:
            return {"mean": 0, "std": 0, "min": 0, "max": 0}
        
        ratios = np.array(self.fg_ratios[-1000:])  # Keep only the most recent 1000 samples
        return {
            "mean": float(ratios.mean()),
            "std": float(ratios.std()),
            "min": float(ratios.min()),
            "max": float(ratios.max())
        }

# ==================== Smart Dataset Processor ====================

class WatermarkDatasetProcessor:
    """Smart dataset processor supporting multiple folder structures"""
    
    def __init__(self, data_sources: List[str], quality_threshold: float = 0.8):
        self.data_sources = data_sources
        self.quality_threshold = quality_threshold
        self.pairs = []
        self.failed_pairs = []
        self.stats = {"total_found": 0, "valid_pairs": 0, "failed_pairs": 0}
        
        self._process_all_sources()
        
    def _process_all_sources(self):
        """Process all data sources"""
        for source in self.data_sources:
            source_path = Path(source)
            if not source_path.exists():
                logging.warning(f"Data source not found: {source}")
                continue
                
            logging.info(f"Processing data source: {source}")
            
            if self._is_separated_structure(source_path):
                pairs = self._process_separated_structure(source_path)
                logging.info(f"  Found {len(pairs)} pairs in separated structure")
            else:
                pairs = self._process_mixed_structure(source_path)
                logging.info(f"  Found {len(pairs)} pairs in mixed structure")
            
            # Quality check
            valid_pairs = self._quality_check_pairs(pairs)
            self.pairs.extend(valid_pairs)
            
        logging.info(f"📊 Dataset processing complete:")
        logging.info(f"  Total pairs found: {self.stats['total_found']}")
        logging.info(f"  Valid pairs: {self.stats['valid_pairs']}")
        logging.info(f"  Failed pairs: {self.stats['failed_pairs']}")
    
    def _is_separated_structure(self, path: Path) -> bool:
        """Check for separated structure (clean/ and watermarked/ folders)"""
        return (path / "clean").exists() and (path / "watermarked").exists()
    
    def _process_separated_structure(self, source_path: Path) -> List[Tuple[Path, Path]]:
        """Process a dataset with separated structure"""
        clean_folder = source_path / "clean"
        watermarked_folder = source_path / "watermarked"
        
        pairs = []
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        # Collect all clean images
        clean_files = []
        for ext in image_extensions:
            clean_files.extend(clean_folder.glob(f"*{ext}"))
            clean_files.extend(clean_folder.glob(f"*{ext.upper()}"))
        
        for clean_file in clean_files:
            clean_stem = clean_file.stem
            
            # Find the matching watermarked image
            watermark_candidates = []
            
            # Try multiple naming patterns
            patterns = [
                f"{clean_stem}_wm.*",      # img_wm.jpg
                f"{clean_stem}_watermark.*", # img_watermark.jpg
                f"{clean_stem}.*",         # Same name
                f"wm_{clean_stem}.*",      # wm_img.jpg
                f"watermark_{clean_stem}.*" # watermark_img.jpg
            ]
            
            for pattern in patterns:
                candidates = list(watermarked_folder.glob(pattern))
                watermark_candidates.extend(candidates)
            
            if watermark_candidates:
                # Pick the best match
                best_match = self._find_best_match(clean_file, watermark_candidates)
                if best_match:
                    pairs.append((clean_file, best_match))
        
        self.stats['total_found'] += len(pairs)
        return pairs
    
    def _process_mixed_structure(self, source_path: Path) -> List[Tuple[Path, Path]]:
        """Process a dataset with mixed structure (all files in one folder)"""
        pairs = []
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        # Collect all image files
        all_files = []
        for ext in image_extensions:
            all_files.extend(source_path.glob(f"*{ext}"))
            all_files.extend(source_path.glob(f"*{ext.upper()}"))
        
        # Classify files
        clean_files = []
        watermark_files = []
        
        for file in all_files:
            filename_lower = file.name.lower()
            if any(marker in filename_lower for marker in ['_wm.', '_watermark.', 'watermark_', 'wm_']):
                watermark_files.append(file)
            else:
                clean_files.append(file)
        
        # Match file pairs
        for clean_file in clean_files:
            clean_stem = clean_file.stem.lower()
            
            # Find the matching watermark file
            candidates = []
            for wm_file in watermark_files:
                wm_name_lower = wm_file.name.lower()
                
                # Multiple matching patterns
                if (f"{clean_stem}_wm." in wm_name_lower or
                    f"{clean_stem}_watermark." in wm_name_lower or
                    f"wm_{clean_stem}." in wm_name_lower or
                    f"watermark_{clean_stem}." in wm_name_lower):
                    candidates.append(wm_file)
            
            if candidates:
                best_match = self._find_best_match(clean_file, candidates)
                if best_match:
                    pairs.append((clean_file, best_match))
        
        self.stats['total_found'] += len(pairs)
        return pairs
    
    def _find_best_match(self, clean_file: Path, candidates: List[Path]) -> Optional[Path]:
        """Find the best match among candidate files"""
        if len(candidates) == 1:
            return candidates[0]
        
        # With multiple candidates, choose the closest file size
        try:
            clean_size = clean_file.stat().st_size
            best_match = None
            min_size_diff = float('inf')
            
            for candidate in candidates:
                candidate_size = candidate.stat().st_size
                size_diff = abs(clean_size - candidate_size)
                
                if size_diff < min_size_diff:
                    min_size_diff = size_diff
                    best_match = candidate
            
            return best_match
        except:
            return candidates[0]  # Return the first candidate on error
    
    def _quality_check_pairs(self, pairs: List[Tuple[Path, Path]]) -> List[Tuple[Path, Path]]:
        """Quality-check image pairs"""
        valid_pairs = []
        
        for clean_path, wm_path in pairs:
            try:
                # Check that files are readable
                clean_img = Image.open(clean_path)
                wm_img = Image.open(wm_path)
                
                # Check size match
                size_match_score = self._calculate_size_match(clean_img.size, wm_img.size)
                
                if size_match_score >= self.quality_threshold:
                    valid_pairs.append((clean_path, wm_path))
                    self.stats['valid_pairs'] += 1
                else:
                    self.failed_pairs.append((clean_path, wm_path, f"Size mismatch: {size_match_score:.2f}"))
                    self.stats['failed_pairs'] += 1
                    
            except Exception as e:
                self.failed_pairs.append((clean_path, wm_path, f"Load error: {str(e)}"))
                self.stats['failed_pairs'] += 1
        
        return valid_pairs
    
    def _calculate_size_match(self, size1: Tuple[int, int], size2: Tuple[int, int]) -> float:
        """Compute size match score"""
        w1, h1 = size1
        w2, h2 = size2
        
        # Aspect ratio difference
        ratio1 = w1 / h1
        ratio2 = w2 / h2
        ratio_diff = abs(ratio1 - ratio2) / max(ratio1, ratio2)
        
        # Area difference
        area1 = w1 * h1
        area2 = w2 * h2
        area_diff = abs(area1 - area2) / max(area1, area2)
        
        # Combined score
        score = 1.0 - (ratio_diff * 0.3 + area_diff * 0.7)
        return max(0.0, score)
    
    def get_pairs(self) -> List[Tuple[Path, Path]]:
        """Get valid image pairs"""
        return self.pairs
    
    def save_quality_report(self, output_path: str = "dataset_quality_report.txt"):
        """Save quality report"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("Dataset Quality Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Statistics:\n")
            f.write(f"  Total pairs found: {self.stats['total_found']}\n")
            f.write(f"  Valid pairs: {self.stats['valid_pairs']}\n")
            f.write(f"  Failed pairs: {self.stats['failed_pairs']}\n")
            # Guard against division by zero
            if self.stats['total_found'] > 0:
                success_rate = self.stats['valid_pairs']/self.stats['total_found']*100
                f.write(f"  Success rate: {success_rate:.1f}%\n\n")
            else:
                f.write(f"  Success rate: N/A (no pairs found)\n\n")
            
            if self.failed_pairs:
                f.write("Failed Pairs:\n")
                f.write("-" * 30 + "\n")
                for clean_path, wm_path, reason in self.failed_pairs:
                    f.write(f"Clean: {clean_path.name}\n")
                    f.write(f"Watermark: {wm_path.name}\n")
                    f.write(f"Reason: {reason}\n\n")

# ==================== Advanced Data Augmentation ====================

class SimpleWatermarkAugmentation:
    """Simplified watermark augmentation to avoid compatibility issues"""
    
    def __init__(self, image_size: int = 384):
        self.image_size = image_size
        self.train_transform = self._create_train_transform()
        self.val_transform = self._create_val_transform()
    
    def _create_train_transform(self):
        """Create training augmentation using basic, stable transforms"""
        return A.Compose([
            # Basic geometric transforms
            A.Resize(height=self.image_size, width=self.image_size),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.2, 
                rotate_limit=15,
                border_mode=cv2.BORDER_REFLECT,
                p=0.5
            ),
            
            # Basic color augmentation
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=10,
                p=0.3
            ),
            
            # Slight blur
            A.Blur(blur_limit=3, p=0.2),
            
            # Normalization
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    
    def _create_val_transform(self):
        """Create validation transforms"""
        return A.Compose([
            A.Resize(height=self.image_size, width=self.image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

# ==================== Smart Watermark Dataset ====================

class SmartWatermarkDataset(Dataset):
    """Smart watermark dataset with automatic high-quality mask generation"""
    
    def __init__(self, 
                 pairs: List[Tuple[Path, Path]], 
                 transform=None, 
                 is_training: bool = True,
                 mask_dilation: int = 9,
                 mask_threshold: float = 0.45,
                 feather_radius: int = 7):
        
        self.pairs = pairs
        self.transform = transform
        if self.transform is None:  # Fall back to the built-in joint image+mask augmentation
            self.transform = self._build_joint_transform(H=384, W=384)  # Adjust 384 to match image_size
        self.is_training = is_training
        self.mask_dilation = mask_dilation
        self.mask_threshold = mask_threshold
        self.feather_radius = feather_radius
        
        logging.info(f"Dataset initialized: {len(self.pairs)} pairs")
        logging.info(f"Parameters: dilation={mask_dilation}, threshold={mask_threshold}, feather={feather_radius}")
    import albumentations as A
    import cv2
    from albumentations.pytorch import ToTensorV2

    def _build_joint_transform(self, H: int = 384, W: int = 384):
        """
        Joint image+mask augmentation: geometric ops are synchronized (same random parameters for image and mask);
        pixel perturbations (compression/noise/color) apply to the image only; finally both are converted to tensors.
        """
        return A.Compose(
            [
                # -- Synchronized geometry: aspect-preserving resize -> pad -> crop to target size --
                A.LongestMaxSize(max_size=max(H, W), interpolation=cv2.INTER_AREA),
                A.PadIfNeeded(min_height=H, min_width=W,
                            border_mode=cv2.BORDER_CONSTANT,
                            value=(114,114,114),   # Image padding color
                            mask_value=0),         # Mask padding = 0 (background)
                A.RandomCrop(height=H, width=W, p=1.0),

                # -- Image-only perturbations (mask untouched) --
                A.ImageCompression(quality_lower=60, quality_upper=100, p=0.30),
                A.GaussNoise(var_limit=(10, 50), p=0.25),
                A.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.10, hue=0.06, p=0.30),

                # -- Normalize + convert to tensor (applies to both) --
                A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
                ToTensorV2(),
            ],
            additional_targets={'mask': 'mask'}  # Key: tells albumentations to apply the same geometry to 'mask'
        )
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        clean_path, wm_path = self.pairs[idx]
        
        try:
            # Load images
            clean_img = Image.open(clean_path).convert('RGB')
            wm_img = Image.open(wm_path).convert('RGB')
            
            # Align sizes
            clean_img, wm_img = self._align_sizes(clean_img, wm_img)
            
            # Generate high-quality mask
            mask = self._generate_smart_mask(clean_img, wm_img)
            
            # Data augmentation
            # Data augmentation
            # ====== Data augmentation (joint: image and mask share geometry) ======
            if self.transform is not None:
                # Albumentations applies the same random geometric params to image and mask
                aug = self.transform(
                    image=np.array(wm_img),           # Watermarked image (RGB)
                    mask=mask.astype(np.uint8)        # Single-channel 0/1 (or 0/255)
                )
                wm_tensor   = aug['image']            # torch.float32 [3,H,W]
                mask_tensor = aug['mask']             # torch [H,W] or [1,H,W] (varies by ToTensorV2 version)

                # Normalize to [1,H,W] float for BCEWithLogitsLoss
                if mask_tensor.dim() == 2:
                    mask_tensor = mask_tensor.unsqueeze(0)
                mask_tensor = mask_tensor.float()

                # The clean image only needs geometric consistency; skip color/noise perturbations to keep labels aligned
                # Sync the clean image geometrically by reusing the augmented output size for distortion-free alignment
                # For simplicity, resize clean_img to match wm_tensor (if clean_tensor is needed)
                clean_np = np.array(clean_img)
                if clean_np.shape[:2][::-1] != tuple(wm_tensor.shape[-2:][::-1]):  # (W,H) comparison
                    clean_np = cv2.resize(clean_np, (wm_tensor.shape[-1], wm_tensor.shape[-2]), interpolation=cv2.INTER_AREA)
                clean_tensor = torch.from_numpy(clean_np).permute(2,0,1).float() / 255.0

            else:
                # ====== No augmentation (original logic) ======
                to_tensor = transforms.ToTensor()
                wm_tensor    = to_tensor(wm_img)          # [3,H,W] float
                clean_tensor = to_tensor(clean_img)       # [3,H,W] float
                mask_tensor  = torch.FloatTensor(mask).unsqueeze(0)  # [1,H,W] float
            
            return {
                'watermarked': wm_tensor,
                'clean': clean_tensor,
                'mask': mask_tensor,
                'paths': {'clean': str(clean_path), 'watermarked': str(wm_path)}
            }
            
        except Exception as e:
            logging.error(f"Error loading pair {idx}: {clean_path} | {wm_path} | Error: {e}")
            # Return zero tensors as fallback
            zero_img = torch.zeros(3, 384, 384)
            zero_mask = torch.zeros(1, 384, 384)
            return {
                'watermarked': zero_img,
                'clean': zero_img, 
                'mask': zero_mask,
                'paths': {'clean': str(clean_path), 'watermarked': str(wm_path)}
            }
    
    def _align_sizes(self, clean_img: Image.Image, wm_img: Image.Image) -> Tuple[Image.Image, Image.Image]:
        """Align the sizes of two images"""
        clean_size = clean_img.size
        wm_size = wm_img.size
        
        if clean_size == wm_size:
            return clean_img, wm_img
        
        # Use the smaller size as target
        target_width = min(clean_size[0], wm_size[0])
        target_height = min(clean_size[1], wm_size[1])
        target_size = (target_width, target_height)
        
        # High-quality resampling
        clean_resized = clean_img.resize(target_size, Image.LANCZOS)
        wm_resized = wm_img.resize(target_size, Image.LANCZOS)
        
        return clean_resized, wm_resized
    
    def _generate_smart_mask(self, clean_img: Image.Image, wm_img: Image.Image) -> np.ndarray:
        """Generate a high-quality watermark mask"""
        clean_array = np.array(clean_img)
        wm_array = np.array(wm_img)
        
# 1. Initial detection based on color difference
        diff = np.abs(clean_array.astype(np.float32) - wm_array.astype(np.float32))
        diff_gray = np.mean(diff, axis=2)
        
        # 2. Adaptive threshold
        # Use OTSU to determine the optimal threshold automatically
        _, mask_binary = cv2.threshold(
            diff_gray.astype(np.uint8), 
            0, 255, 
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        
        # 3. Morphological operations to remove noise
        kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_cleaned = cv2.morphologyEx(mask_binary, cv2.MORPH_OPEN, kernel_clean)
        mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, kernel_clean)
        
        # 4. Dilation to fully cover the watermark region
        if self.mask_dilation > 0:
            kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.mask_dilation, self.mask_dilation))
            mask_dilated = cv2.dilate(mask_cleaned, kernel_dilate, iterations=1)
        else:
            mask_dilated = mask_cleaned
        
        # 5. Feather edges for smooth transitions
        if self.feather_radius > 0:
            mask_feathered = cv2.GaussianBlur(
                mask_dilated.astype(np.float32), 
                (self.feather_radius * 2 + 1, self.feather_radius * 2 + 1), 
                self.feather_radius / 3
            )
        else:
            mask_feathered = mask_dilated.astype(np.float32)
        
        # 6. Normalize to [0, 1]
        mask_normalized = mask_feathered / 255.0
        
        # 7. Apply threshold
        mask_final = (mask_normalized > self.mask_threshold).astype(np.float32)
        
        return mask_final

# ==================== Detection Model Architecture ====================

class WatermarkDetectionModel(nn.Module):
    """Watermark detection model with corrected output size"""
    
    def __init__(self, pretrained: bool = True):
        super(WatermarkDetectionModel, self).__init__()
        
        # Use ResNet18 for simplicity and stability
        resnet = models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        # Backbone output channels (512 for ResNet18)
        self.backbone_channels = 512
        
        # Upsampling decoder to produce the correct output size
        self.decoder = nn.ModuleList([
            # Layer 1: 512 -> 256
            nn.Sequential(
                nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            ),
            # Layer 2: 256 -> 128  
            nn.Sequential(
                nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True)
            ),
            # Layer 3: 128 -> 64
            nn.Sequential(
                nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            ),
            # Layer 4: 64 -> 32
            nn.Sequential(
                nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True)
            ),
            # Layer 5: 32 -> 16
            nn.Sequential(
                nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True)
            )
        ])
        
        # Final output layer
        self.final_conv = nn.Conv2d(16, 1, 3, padding=1)
        
    def forward(self, x):
        # Record input size
        input_size = x.shape[-2:]
        
        # Feature extraction
        features = self.backbone(x)
        
        # Upsample layer by layer
        x = features
        for decoder_layer in self.decoder:
            x = decoder_layer(x)
        
        # Final convolution
        x = self.final_conv(x)
        
        # Ensure output size exactly matches input
        if x.shape[-2:] != input_size:
            x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)
        
        return x
class SpatialAttentionBlock(nn.Module):
    """Spatial attention block"""
    
    def __init__(self, in_channels: int):
        super(SpatialAttentionBlock, self).__init__()
        
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, 1),
            nn.BatchNorm2d(in_channels // 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        attention_map = self.attention(x)
        return x * attention_map

# ==================== Training Monitor ====================

class TrainingMonitor:
    """Training process monitor"""
    
    def __init__(self):
        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'train_dice': [],
            'val_dice': [],
            'fg_ratio': [],
            'learning_rate': []
        }
        self.best_metrics = {
            'val_loss': float('inf'),
            'val_dice': 0.0,
            'epoch': 0
        }
        
    def update(self, epoch: int, metrics: Dict[str, float]):
        """Update monitored metrics"""
        for key, value in metrics.items():
            if key in self.metrics_history:
                self.metrics_history[key].append(value)
        
        # Update best metrics
        if 'val_loss' in metrics and metrics['val_loss'] < self.best_metrics['val_loss']:
            self.best_metrics['val_loss'] = metrics['val_loss']
            self.best_metrics['epoch'] = epoch
        
        if 'val_dice' in metrics and metrics['val_dice'] > self.best_metrics['val_dice']:
            self.best_metrics['val_dice'] = metrics['val_dice']
    
    def should_stop_early(self, patience: int = 15) -> bool:
        """Check whether to stop early"""
        if len(self.metrics_history['val_loss']) < patience:
            return False
        
        current_epoch = len(self.metrics_history['val_loss']) - 1
        epochs_since_improvement = current_epoch - self.best_metrics['epoch']
        
        return epochs_since_improvement >= patience
    
    def get_progress_summary(self) -> str:
        """Get training progress summary"""
        if not self.metrics_history['train_loss']:
            return "Training not started"
        
        current_epoch = len(self.metrics_history['train_loss'])
        latest_train_loss = self.metrics_history['train_loss'][-1]
        latest_val_loss = self.metrics_history['val_loss'][-1] if self.metrics_history['val_loss'] else 0
        latest_fg_ratio = self.metrics_history['fg_ratio'][-1] if self.metrics_history['fg_ratio'] else 0
        
        return f"Epoch {current_epoch}: Train Loss={latest_train_loss:.4f}, Val Loss={latest_val_loss:.4f}, FG Ratio={latest_fg_ratio:.4f}, Best Val Loss={self.best_metrics['val_loss']:.4f} @Epoch {self.best_metrics['epoch']}"

# ==================== Main Trainer ====================

class WatermarkDetectionTrainer:
    """Watermark detection trainer designed to avoid loss plateaus"""
    
    def __init__(self, 
                 data_sources: List[str],
                 output_dir: str = "watermark_detection_output",
                 config: Dict[str, Any] = None):
        
        # Set up output directory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Set up logging
        self._setup_logging()
        
        # Load configuration
        self.config = self._load_config(config)
        
        # Initialize device manager
        self.device_manager = SmartDeviceManager(self.config['device'])
        self.device = self.device_manager.device
        
        # Process dataset
        logging.info("Processing dataset...")
        self.dataset_processor = WatermarkDatasetProcessor(
            data_sources, 
            quality_threshold=self.config['quality_threshold']
        )
        # Check for valid data
        valid_pairs = self.dataset_processor.get_pairs()
        if len(valid_pairs) == 0:
            raise ValueError(f"No valid image pairs found in data sources: {data_sources}")
        # Save dataset quality report
        self.dataset_processor.save_quality_report(
            str(self.output_dir / "dataset_quality_report.txt")
        )
        
        # Create datasets
        self._create_datasets()
        
        # Create model
        self.model = WatermarkDetectionModel(pretrained=True).to(self.device)
        
        # Create loss function
        self.criterion = AdaptiveLossFunction(base_pos_weight=self.config['pos_weight'])
        
        # Create optimizer
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if self.device.type == 'cuda' else None
        
        # Training monitor
        self.monitor = TrainingMonitor()
        
        logging.info("Trainer initialized successfully!")
        logging.info(f"Device: {self.device}")
        logging.info(f"Training samples: {len(self.train_dataset)}")
        logging.info(f"Validation samples: {len(self.val_dataset)}")
    
    def _setup_logging(self):
        """Set up logging"""
        log_file = self.output_dir / "training.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    def _load_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Load configuration"""
        default_config = {
            # Device settings
            'device': 'auto',
            
            # Data settings
            'image_size': 384,
            'quality_threshold': 0.8,
            'mask_dilation': 9,
            'mask_threshold': 0.45,
            'feather_radius': 7,
            
            # Training settings
            'batch_size': 16,
            'num_epochs': 100,
            'learning_rate': 5e-4,  # Higher learning rate
            'weight_decay': 1e-5,
            'pos_weight': 6.0,
            
            # Optimization settings
            'patience': 15,
            'val_ratio': 0.15,
            'num_workers': 4,
            
            # Checkpoint settings
            'save_every': 5,
            'save_best_only': True
        }
        
        if config:
            default_config.update(config)
        
        return default_config
    
    def _create_datasets(self):
        """Create training and validation datasets"""
        pairs = self.dataset_processor.get_pairs()
        
        if len(pairs) == 0:
            raise ValueError("No valid image pairs found!")
        
        # Random train/val split
        random.shuffle(pairs)
        val_size = int(len(pairs) * self.config['val_ratio'])
        train_pairs = pairs[val_size:]
        val_pairs = pairs[:val_size]
        
        # Create augmentation
        augmentation = SimpleWatermarkAugmentation(self.config['image_size'])
        
        # Create datasets
        self.train_dataset = SmartWatermarkDataset(
            train_pairs,
            transform=augmentation.train_transform,
            is_training=True,
            mask_dilation=self.config['mask_dilation'],
            mask_threshold=self.config['mask_threshold'],
            feather_radius=self.config['feather_radius']
        )
        
        self.val_dataset = SmartWatermarkDataset(
            val_pairs,
            transform=augmentation.val_transform,
            is_training=False,
            mask_dilation=self.config['mask_dilation'],
            mask_threshold=self.config['mask_threshold'],
            feather_radius=self.config['feather_radius']
        )
        
        # Create data loaders
        batch_size = self.device_manager.get_optimal_batch_size(self.config['batch_size'])
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.config['num_workers'],
            pin_memory=True if self.device.type != 'cpu' else False,
            drop_last=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size // 2,
            shuffle=False,
            num_workers=self.config['num_workers'] // 2,
            pin_memory=True if self.device.type != 'cpu' else False
        )
        
        logging.info(f"Batch size optimized to: {batch_size}")
    
    def _create_optimizer(self):
        """Create optimizer"""
        # Group parameters to use different learning rates per layer
        backbone_params = []
        decoder_params = []
        
        for name, param in self.model.named_parameters():
            if 'features' in name:  # backbone
                backbone_params.append(param)
            else:  # decoder
                decoder_params.append(param)
        
        optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': self.config['learning_rate'] * 0.1},  # Lower learning rate for backbone
            {'params': decoder_params, 'lr': self.config['learning_rate']}  # Normal learning rate for decoder
        ], weight_decay=self.config['weight_decay'])
        
        return optimizer
    
    def _create_scheduler(self):
        """Create learning rate scheduler"""
        # Use Cosine Annealing with Warm Restarts
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,  # Initial cycle length
            T_mult=2,  # Cycle multiplier
            eta_min=1e-7  # Minimum learning rate
        )
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        total_dice = 0.0
        num_batches = len(self.train_loader)
        
        # Foreground ratio monitoring
        fg_ratios = []
        
        for batch_idx, batch in enumerate(self.train_loader):
            watermarked = batch['watermarked'].to(self.device, non_blocking=True)
            masks = batch['mask'].to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            if self.scaler and self.device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    outputs = self.model(watermarked)
                    loss_dict = self.criterion(outputs, masks)
            else:
                outputs = self.model(watermarked)
                loss_dict = self.criterion(outputs, masks)
            
            loss = loss_dict['total_loss']
            
            # Backward pass
            if self.scaler and self.device.type == 'cuda':
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            
            # Compute Dice coefficient
            with torch.no_grad():
                pred_masks = torch.sigmoid(outputs) > 0.5
                dice_score = self._calculate_dice_score(pred_masks, masks)
                total_dice += dice_score
            
            total_loss += loss.item()
            fg_ratios.append(loss_dict['avg_fg_ratio'].item())
            
            # Periodically log progress and foreground ratio
            if batch_idx % 50 == 0:
                current_fg_ratio = loss_dict['avg_fg_ratio'].item()
                logging.info(
                    f'Epoch {epoch}, Batch {batch_idx}/{num_batches}: '
                    f'Loss={loss.item():.4f}, FG_ratio={current_fg_ratio:.4f}'
                )
                
                # Detect abnormal foreground ratio
                if current_fg_ratio < 0.001:
                    logging.warning("⚠️  Very low foreground ratio detected! Consider adjusting mask parameters.")
        
        avg_loss = total_loss / num_batches
        avg_dice = total_dice / num_batches
        avg_fg_ratio = sum(fg_ratios) / len(fg_ratios)
        
        return {
            'train_loss': avg_loss,
            'train_dice': avg_dice,
            'fg_ratio': avg_fg_ratio
        }
    
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        
        total_loss = 0.0
        total_dice = 0.0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for batch in self.val_loader:
                watermarked = batch['watermarked'].to(self.device, non_blocking=True)
                masks = batch['mask'].to(self.device, non_blocking=True)
                
                # Forward pass
                if self.scaler and self.device.type == 'cuda':
                    with torch.cuda.amp.autocast():
                        outputs = self.model(watermarked)
                        loss_dict = self.criterion(outputs, masks)
                else:
                    outputs = self.model(watermarked)
                    loss_dict = self.criterion(outputs, masks)
                
                loss = loss_dict['total_loss']
                
                # Compute Dice coefficient
                pred_masks = torch.sigmoid(outputs) > 0.5
                dice_score = self._calculate_dice_score(pred_masks, masks)
                
                total_loss += loss.item()
                total_dice += dice_score
        
        avg_loss = total_loss / num_batches
        avg_dice = total_dice / num_batches
        
        return {
            'val_loss': avg_loss,
            'val_dice': avg_dice
        }
    
    def _calculate_dice_score(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Compute Dice coefficient"""
        smooth = 1e-6
        pred_flat = pred.view(-1).float()
        target_flat = target.view(-1).float()
        
        intersection = (pred_flat * target_flat).sum()
        dice = (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
        
        return dice.item()
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Save checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config,
            'best_metrics': self.monitor.best_metrics
        }
        
        # Save current checkpoint
        checkpoint_path = self.output_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.output_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            logging.info(f"💾 New best model saved: {metrics}")
        
        # Save latest model
        latest_path = self.output_dir / "latest_model.pth"
        torch.save(checkpoint, latest_path)
    
    def visualize_predictions(self, epoch: int, num_samples: int = 4):
        """Visualize predictions"""
        self.model.eval()
        
        # Create output directory
        vis_dir = self.output_dir / "visualizations" / f"epoch_{epoch}"
        vis_dir.mkdir(parents=True, exist_ok=True)
        
        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                if i >= num_samples:
                    break
                
                watermarked = batch['watermarked'].to(self.device)
                masks_true = batch['mask'].to(self.device)
                
                # Predict
                outputs = self.model(watermarked)
                masks_pred = torch.sigmoid(outputs)
                
                # Save visualizations
                for j in range(min(4, watermarked.size(0))):
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    
                    # Original image
                    img = watermarked[j].cpu()
                    img = (img - img.min()) / (img.max() - img.min())  # Normalize for display
                    axes[0].imshow(img.permute(1, 2, 0))
                    axes[0].set_title("Watermarked Image")
                    axes[0].axis('off')
                    
                    # Ground truth mask
                    mask_true = masks_true[j, 0].cpu().numpy()
                    axes[1].imshow(mask_true, cmap='hot')
                    axes[1].set_title("Ground Truth Mask")
                    axes[1].axis('off')
                    
                    # Predicted mask
                    mask_pred = masks_pred[j, 0].cpu().numpy()
                    axes[2].imshow(mask_pred, cmap='hot')
                    axes[2].set_title(f"Predicted Mask")
                    axes[2].axis('off')
                    
                    plt.tight_layout()
                    plt.savefig(vis_dir / f"sample_{i}_{j}.png", dpi=150, bbox_inches='tight')
                    plt.close()
    
    def train(self):
        """Full training loop"""
        logging.info("🚀 Starting training...")
        logging.info(f"Configuration: {self.config}")
        
        for epoch in range(1, self.config['num_epochs'] + 1):
            epoch_start_time = time.time()
            
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate_epoch(epoch)
            
            # Update learning rate
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Merge metrics
            all_metrics = {**train_metrics, **val_metrics, 'learning_rate': current_lr}
            
            # Update monitor
            self.monitor.update(epoch, all_metrics)
            
            # Check for best model
            is_best = val_metrics['val_loss'] < self.monitor.best_metrics['val_loss']
            
            # Save checkpoint
            if epoch % self.config['save_every'] == 0 or is_best:
                self.save_checkpoint(epoch, all_metrics, is_best)
            
            # Visualize predictions
            if epoch % 10 == 0:
                self.visualize_predictions(epoch)
            
            # Log progress
            epoch_time = time.time() - epoch_start_time
            progress_summary = self.monitor.get_progress_summary()
            logging.info(f"⏱️  Epoch {epoch} completed in {epoch_time:.1f}s")
            logging.info(f"📊 {progress_summary}")
            
            # Early stopping check
            if self.monitor.should_stop_early(self.config['patience']):
                logging.info(f"🛑 Early stopping triggered after {epoch} epochs")
                break
            
            # Foreground ratio check
            if train_metrics['fg_ratio'] < 0.001:
                logging.warning("🔔 Consistently low foreground ratio detected!")
                logging.warning("   Consider: 1) Lower mask_threshold, 2) Increase mask_dilation, 3) Check data quality")
        
        # Training complete
        logging.info("✅ Training completed!")
        logging.info(f"🏆 Best validation loss: {self.monitor.best_metrics['val_loss']:.4f} at epoch {self.monitor.best_metrics['epoch']}")
        
        # Save final training report
        self._save_training_report()
        
        return self.output_dir / "best_model.pth"
    
    def _save_training_report(self):
        """Save training report"""
        report = {
            'config': self.config,
            'best_metrics': self.monitor.best_metrics,
            'final_metrics': {
                'train_loss': self.monitor.metrics_history['train_loss'][-1] if self.monitor.metrics_history['train_loss'] else 0,
                'val_loss': self.monitor.metrics_history['val_loss'][-1] if self.monitor.metrics_history['val_loss'] else 0,
                'train_dice': self.monitor.metrics_history['train_dice'][-1] if self.monitor.metrics_history['train_dice'] else 0,
                'val_dice': self.monitor.metrics_history['val_dice'][-1] if self.monitor.metrics_history['val_dice'] else 0,
            },
            'total_epochs': len(self.monitor.metrics_history['train_loss']),
            'device_info': self.device_manager.device_info
        }
        
        report_path = self.output_dir / "training_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logging.info(f"📋 Training report saved: {report_path}")

# ==================== Main Function and Usage Example ====================

def main():
    """Main entry point"""
    print("=" * 60)
    print("Production-grade Watermark Detection Training System")
    print("Designed to address loss plateaus")
    print("=" * 60)
    
    # Example data paths; adjust to your dataset
    data_sources = [
        "./expanded_dataset",  # Path containing clean/ and watermarked/ folders
        "./rename_output"      # Path containing mixed files
    ]
    
    # Training settings
    config = {
        'device': 'auto',  # Auto-select best device (MPS supported)
        'image_size': 384,  # Larger image size so the watermark covers more pixels
        'batch_size': 10,
        'num_epochs': 10,
        'learning_rate': 5e-4,  # Higher learning rate to improve convergence on MPS/CPU
        'pos_weight': 6.0,  # Higher foreground weight
        'mask_dilation': 9,  # Increase pseudo-label recall
        'mask_threshold': 0.45,  # Lower threshold
        'feather_radius': 7,
        'patience': 15,
        'val_ratio': 0.15
    }
    
    # Create trainer
    trainer = WatermarkDetectionTrainer(
        data_sources=data_sources,
        output_dir="watermark_detection_training",
        config=config
    )
    
    # Start training
    best_model_path = trainer.train()
    
    print(f"\nTraining complete. Best model saved at: {best_model_path}")

# ==================== Inference and Testing Tools ====================

class WatermarkDetectionInference:
    """Watermark detection inference engine"""
    
    def __init__(self, model_path: str, device: str = 'auto'):
        self.device_manager = SmartDeviceManager(device)
        self.device = self.device_manager.device
        
        # Load model
        self.model = WatermarkDetectionModel(pretrained=False).to(self.device)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Preprocessing
        self.transform = A.Compose([
            A.Resize(height=384, width=384),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
        
        print(f"Model loaded on {self.device}")
    
    def detect_watermark(self, image_path: str, output_path: str = None, threshold: float = 0.5) -> Dict[str, Any]:
        """Detect watermark in a single image"""
        # Load images
        image = Image.open(image_path).convert('RGB')
        original_size = image.size
        
        # Preprocessing
        transformed = self.transform(image=np.array(image))
        input_tensor = transformed['image'].unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            if self.device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    output = self.model(input_tensor)
            else:
                output = self.model(input_tensor)
            
            mask_prob = torch.sigmoid(output).cpu().numpy()[0, 0]
            mask_binary = (mask_prob > threshold).astype(np.uint8) * 255
        
        # Resize to original size
        mask_resized = cv2.resize(mask_prob, original_size, interpolation=cv2.INTER_LINEAR)
        mask_binary_resized = cv2.resize(mask_binary, original_size, interpolation=cv2.INTER_NEAREST)
        
        # Compute statistics
        watermark_area = np.sum(mask_binary_resized > 0)
        total_area = mask_binary_resized.shape[0] * mask_binary_resized.shape[1]
        watermark_ratio = watermark_area / total_area
        
        result = {
            'watermark_detected': watermark_area > 100,  # At least 100 pixels
            'watermark_ratio': watermark_ratio,
            'confidence_score': np.mean(mask_prob),
            'max_confidence': np.max(mask_prob),
            'mask_probability': mask_resized,
            'mask_binary': mask_binary_resized,
            'original_size': original_size
        }
        
        # Save results
        if output_path:
            # Save probability map and binary mask
            base_path = Path(output_path)
            prob_path = base_path.with_suffix('.prob.png')
            binary_path = base_path.with_suffix('.mask.png')
            
            # Save probability map (heatmap)
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(image)
            plt.title("Original Image")
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.imshow(mask_resized, cmap='hot')
            plt.title(f"Watermark Detection (Ratio: {watermark_ratio:.3f})")
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(prob_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            # Save binary mask
            cv2.imwrite(str(binary_path), mask_binary_resized)
            
            result['output_files'] = {
                'probability_map': str(prob_path),
                'binary_mask': str(binary_path)
            }
        
        return result
    
    def batch_detect(self, input_dir: str, output_dir: str, threshold: float = 0.5) -> Dict[str, Any]:
        """Batch detection"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Supported formats
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        for ext in extensions:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))
        
        results = {
            'total_files': len(image_files),
            'processed_files': 0,
            'watermark_detected': 0,
            'detection_results': [],
            'summary_stats': {}
        }
        
        watermark_ratios = []
        confidence_scores = []
        
        for img_file in image_files:
            try:
                print(f"Processing: {img_file.name}")
                
                output_file = output_path / img_file.name
                result = self.detect_watermark(str(img_file), str(output_file), threshold)
                
                results['detection_results'].append({
                    'filename': img_file.name,
                    'watermark_detected': result['watermark_detected'],
                    'watermark_ratio': result['watermark_ratio'],
                    'confidence_score': result['confidence_score']
                })
                
                if result['watermark_detected']:
                    results['watermark_detected'] += 1
                
                watermark_ratios.append(result['watermark_ratio'])
                confidence_scores.append(result['confidence_score'])
                results['processed_files'] += 1
                
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
        
        # Compute summary statistics
        if watermark_ratios:
            results['summary_stats'] = {
                'avg_watermark_ratio': np.mean(watermark_ratios),
                'avg_confidence_score': np.mean(confidence_scores),
                'detection_rate': results['watermark_detected'] / results['processed_files']
            }
        
        # Save results report
        report_path = output_path / "detection_report.json"
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nBatch detection complete:")
        print(f"Total files: {results['total_files']}")
        print(f"Processed: {results['processed_files']}")
        print(f"Watermarks detected: {results['watermark_detected']}")
        print(f"Detection rate: {results['summary_stats'].get('detection_rate', 0):.2%}")
        
        return results

# ==================== Parameter Tuning Tools ====================

class ParameterTuner:
    """Parameter tuner that searches for the best configuration"""
    
    def __init__(self, data_sources: List[str]):
        self.data_sources = data_sources
        self.best_config = None
        self.best_score = float('inf')
    
    def tune_parameters(self, output_dir: str = "parameter_tuning"):
        """Automatically tune parameters"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Parameter search space
        param_grid = {
            'mask_dilation': [5, 7, 9, 11],
            'mask_threshold': [0.35, 0.4, 0.45, 0.5],
            'feather_radius': [5, 7, 9],
            'pos_weight': [4.0, 6.0, 8.0],
            'learning_rate': [2e-4, 5e-4, 1e-3]
        }
        
        tuning_results = []
        
        print("Starting parameter tuning...")
        
        # Grid search
        from itertools import product
        
        param_combinations = list(product(
            param_grid['mask_dilation'],
            param_grid['mask_threshold'], 
            param_grid['feather_radius'],
            param_grid['pos_weight'],
            param_grid['learning_rate']
        ))
        
        for i, (dilation, threshold, feather, pos_weight, lr) in enumerate(param_combinations[:10]):  # Limit number of trials
            print(f"\nTrying parameter combination {i+1}/10:")
            print(f"  mask_dilation: {dilation}")
            print(f"  mask_threshold: {threshold}")
            print(f"  feather_radius: {feather}")
            print(f"  pos_weight: {pos_weight}")
            print(f"  learning_rate: {lr}")
            
            config = {
                'device': 'auto',
                'image_size': 384,
                'batch_size': 8,  # Smaller batch size for faster search
                'num_epochs': 10,  # Fewer epochs
                'learning_rate': lr,
                'pos_weight': pos_weight,
                'mask_dilation': dilation,
                'mask_threshold': threshold,
                'feather_radius': feather,
                'patience': 8,
                'val_ratio': 0.15
            }
            
            try:
                # Create experiment subdirectory
                exp_dir = output_path / f"experiment_{i+1}"
                
                trainer = WatermarkDetectionTrainer(
                    data_sources=self.data_sources,
                    output_dir=str(exp_dir),
                    config=config
                )
                
                # Quick training
                best_model_path = trainer.train()
                
                # Get best validation loss
                best_val_loss = trainer.monitor.best_metrics['val_loss']
                best_val_dice = trainer.monitor.best_metrics['val_dice']
                
                print(f"  Result: Val Loss={best_val_loss:.4f}, Val Dice={best_val_dice:.4f}")
                
                # Record result
                result = {
                    'experiment_id': i+1,
                    'config': config,
                    'val_loss': best_val_loss,
                    'val_dice': best_val_dice,
                    'model_path': str(best_model_path)
                }
                tuning_results.append(result)
                
                # Update best configuration
                if best_val_loss < self.best_score:
                    self.best_score = best_val_loss
                    self.best_config = config.copy()
                    print(f"  🎯 New best configuration!")
                
            except Exception as e:
                print(f"  Experiment failed: {e}")
        
        # Save tuning results
        results_path = output_path / "tuning_results.json"
        with open(results_path, 'w') as f:
            json.dump({
                'best_config': self.best_config,
                'best_score': self.best_score,
                'all_results': tuning_results
            }, f, indent=2, default=str)
        
        print(f"\nParameter tuning complete!")
        print(f"Best configuration: {self.best_config}")
        print(f"Best score: {self.best_score:.4f}")
        print(f"Results saved at: {results_path}")
        
        return self.best_config

# ==================== Usage Example ====================

def example_usage():
    """Usage example"""
    
    # 1. Train model
    print("1. Training watermark detection model...")
    data_sources = [
        "./expanded_dataset",
        "./rename_output"
    ]
    
    # Use recommended configuration
    config = {
        'device': 'auto',
        'image_size': 384,
        'batch_size': 16,
        'num_epochs': 10,
        'learning_rate': 5e-4,
        'pos_weight': 6.0,
        'mask_dilation': 9,
        'mask_threshold': 0.45,
        'feather_radius': 7
    }
    
    trainer = WatermarkDetectionTrainer(data_sources, config=config)
    best_model = trainer.train()
    
    # 2. Inference test
    print("\n2. Testing model...")
    inference = WatermarkDetectionInference(str(best_model))
    
    # Single image test
    result = inference.detect_watermark(
        "TEST2.jpg",
        "TEST3.jpg"
    )
    print(f"Detection result: {result['watermark_detected']}, Confidence: {result['confidence_score']:.3f}")
    
    # # Batch test
    # batch_results = inference.batch_detect(
    #     "test_images/",
    #     "detection_results/"
    # )
    # print(f"Batch detection complete: {batch_results['watermark_detected']}/{batch_results['total_files']} images with watermarks detected")

def quick_start_with_parameter_tuning():
    """Quick start with parameter tuning"""
    
    data_sources = [
        "WATERMARK/expanded_dataset",
        "WATERMARK/rename_output"
    ]
    
    # 1. Parameter tuning
    print("1. Starting parameter tuning...")
    tuner = ParameterTuner(data_sources)
    best_config = tuner.tune_parameters()
    
    # 2. Train the full model with the best configuration
    print("\n2. Training full model with best configuration...")
    best_config.update({
        'num_epochs': 10,  # Full training
        'batch_size': 16
    })
    
    trainer = WatermarkDetectionTrainer(
        data_sources,
        output_dir="final_watermark_model",
        config=best_config
    )
    
    final_model = trainer.train()
    print(f"\nTraining complete! Final model: {final_model}")

if __name__ == "__main__":
    # Select the run mode as needed
    
    # Mode 1: Train directly (recommended configuration)
    main()
    
    # Mode 2: Parameter tuning + training
    # quick_start_with_parameter_tuning()
    
    # Mode 3: Inference only
    # inference = WatermarkDetectionInference("path/to/best_model.pth")
    # result = inference.detect_watermark("test.jpg", "result.jpg")
    # print(result)