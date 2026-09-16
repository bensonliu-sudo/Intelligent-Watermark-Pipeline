import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import cv2
import os
import random
import string
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Dict
import json
import hashlib
from datetime import datetime

# ==================== Configuration ====================
class Config:
    """Global configuration"""
    # Dataset configuration
    DATASET_SIZE = 5000
    TRAIN_SPLIT = 0.8
    IMAGE_SIZE = 512
    BATCH_SIZE = 4
    
    # Watermark generation configuration
    WATERMARK_TYPES = ['engineering', 'text', 'logo', 'mixed']
    OPACITY_RANGE = (0.1, 0.9)  # widened opacity range
    COLOR_VARIATIONS = [
        {'text': (255, 255, 255), 'bg': (0, 0, 255)},    # white text on blue
        {'text': (0, 0, 255), 'bg': (255, 255, 255)},    # blue text on white
        {'text': (255, 0, 0), 'bg': (255, 255, 255)},    # red text on white
        {'text': (0, 255, 0), 'bg': (255, 255, 255)},    # green text on white
        {'text': (255, 255, 255), 'bg': (128, 128, 128)}, # white text on gray
    ]
    
    # Training configuration
    EPOCHS = 50
    LEARNING_RATE = 1e-4
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CHECKPOINT_DIR = 'checkpoints'
    LOG_DIR = 'logs'
    
    # Model configuration
    USE_ATTENTION = True
    USE_MULTISCALE = True
    PRETRAINED_BACKBONE = True

# ==================== Dataset Generation ====================
class WatermarkGenerator:
    """Advanced watermark generator"""
    
    def __init__(self, config: Config):
        self.config = config
        self.font_sizes = [20, 30, 40, 50, 60]
        self.texts = self._generate_text_pool()
        
    def _generate_text_pool(self) -> List[str]:
        """Build a diverse text pool"""
        texts = [
            "CONFIDENTIAL", "SAMPLE", "DRAFT", "PREVIEW",
            "DO NOT COPY", "INTERNAL USE", "WATERMARK",
            "© 2024", "PROTECTED", "DEMO VERSION",
            "NOT FOR SALE", "EVALUATION COPY", "TEST",
            "PRELIMINARY", "RESTRICTED", "PRIVATE"
        ]
        # Add random strings
        for _ in range(10):
            random_text = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
            texts.append(random_text)
        return texts
    
    def generate_engineering_watermark(self, size: Tuple[int, int]) -> Tuple[Image.Image, np.ndarray]:
        """Generate an engineering-style watermark (with frame structure)"""
        watermark = Image.new('RGBA', size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(watermark)
        mask = np.zeros((size[1], size[0]), dtype=np.uint8)
        
        # Pick a random color scheme
        color_scheme = random.choice(self.config.COLOR_VARIATIONS)
        opacity = random.uniform(*self.config.OPACITY_RANGE)
        
        # Draw the outer frame
        border_width = random.randint(2, 5)
        bg_color = (*color_scheme['bg'], int(255 * opacity))
        draw.rectangle([0, 0, size[0]-1, size[1]-1], outline=bg_color, width=border_width)
        
        # Draw the internal structure
        sections = random.randint(2, 4)
        section_height = size[1] // sections
        
        for i in range(sections):
            y = i * section_height
            if i > 0:
                draw.line([(0, y), (size[0], y)], fill=bg_color, width=border_width)
            
            # Add text
            text = random.choice(self.texts)
            font_size = random.choice(self.font_sizes)
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
            except:
                font = ImageFont.load_default()
            
            text_color = (*color_scheme['text'], int(255 * opacity * random.uniform(0.7, 1.0)))
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            x = (size[0] - text_width) // 2
            y_text = y + (section_height - text_height) // 2
            draw.text((x, y_text), text, fill=text_color, font=font)
            
            # Update mask
            mask[y:y+section_height, :] = 255
        
        return watermark, mask
    
    def generate_text_watermark(self, size: Tuple[int, int]) -> Tuple[Image.Image, np.ndarray]:
        """Generate a text watermark (varied layouts)"""
        watermark = Image.new('RGBA', size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(watermark)
        mask = np.zeros((size[1], size[0]), dtype=np.uint8)
        
        color_scheme = random.choice(self.config.COLOR_VARIATIONS)
        opacity = random.uniform(*self.config.OPACITY_RANGE)
        
        # Pick a random layout: diagonal, grid, center
        layout = random.choice(['diagonal', 'grid', 'center', 'random'])
        
        if layout == 'diagonal':
            # Diagonal layout
            text = random.choice(self.texts)
            font_size = random.choice(self.font_sizes)
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
            except:
                font = ImageFont.load_default()
            
            angle = random.randint(30, 60)
            rotated = Image.new('RGBA', size, (0, 0, 0, 0))
            rotated_draw = ImageDraw.Draw(rotated)
            
            # Repeated text
            spacing = font_size * 3
            for i in range(-size[0], size[0]*2, spacing):
                for j in range(-size[1], size[1]*2, spacing):
                    color = (*color_scheme['text'], int(255 * opacity * random.uniform(0.5, 1.0)))
                    rotated_draw.text((i, j), text, fill=color, font=font)
            
            watermark = rotated.rotate(angle, expand=False)
            mask[:, :] = int(255 * opacity)
            
        elif layout == 'grid':
            # Grid layout
            text = random.choice(self.texts)
            font_size = random.randint(15, 30)
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
            except:
                font = ImageFont.load_default()
            
            rows = random.randint(3, 6)
            cols = random.randint(3, 6)
            
            for i in range(rows):
                for j in range(cols):
                    x = j * (size[0] // cols)
                    y = i * (size[1] // rows)
                    color = (*color_scheme['text'], int(255 * opacity * random.uniform(0.3, 1.0)))
                    draw.text((x, y), text, fill=color, font=font)
                    
                    # Update mask
                    mask[y:y+font_size*2, x:x+len(text)*font_size] = 255
                    
        elif layout == 'center':
            # Center layout
            text = random.choice(self.texts)
            font_size = random.randint(40, 80)
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
            except:
                font = ImageFont.load_default()
            
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            x = (size[0] - text_width) // 2
            y = (size[1] - text_height) // 2
            
            # Add background
            padding = 20
            bg_color = (*color_scheme['bg'], int(255 * opacity * 0.5))
            draw.rectangle([x-padding, y-padding, x+text_width+padding, y+text_height+padding], 
                          fill=bg_color)
            
            # Add text
            text_color = (*color_scheme['text'], int(255 * opacity))
            draw.text((x, y), text, fill=text_color, font=font)
            
            # Update mask
            mask[y-padding:y+text_height+padding, x-padding:x+text_width+padding] = 255
            
        else:  # random
            # Random-position layout
            num_texts = random.randint(3, 8)
            for _ in range(num_texts):
                text = random.choice(self.texts)
                font_size = random.randint(15, 40)
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
                except:
                    font = ImageFont.load_default()
                
                x = random.randint(0, size[0] - len(text) * font_size)
                y = random.randint(0, size[1] - font_size)
                
                color = (*color_scheme['text'], int(255 * opacity * random.uniform(0.3, 1.0)))
                draw.text((x, y), text, fill=color, font=font)
                
                # Update mask
                mask[y:y+font_size*2, x:x+len(text)*font_size//2] = 255
        
        return watermark, mask
    
    def generate_watermark(self, size: Tuple[int, int]) -> Tuple[Image.Image, np.ndarray]:
        """Generate a watermark of a random type"""
        watermark_type = random.choice(['engineering', 'text'])
        
        if watermark_type == 'engineering':
            return self.generate_engineering_watermark(size)
        else:
            return self.generate_text_watermark(size)
    
    def apply_watermark(self, image: Image.Image, watermark: Image.Image) -> Image.Image:
        """Apply the watermark to an image"""
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        # Composite the images
        result = Image.alpha_composite(image, watermark)
        return result.convert('RGB')

# ==================== Dataset Class ====================
class WatermarkDataset(Dataset):
    """Watermark dataset"""
    
    def __init__(self, data_dir: str, mode: str = 'train', transform=None):
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.transform = transform
        
        # Load the sample list
        self.samples = []
        mode_dir = self.data_dir / mode
        if mode_dir.exists():
            for img_path in mode_dir.glob('watermarked_*.png'):
                img_id = img_path.stem.replace('watermarked_', '')
                clean_path = mode_dir / f'clean_{img_id}.png'
                mask_path = mode_dir / f'mask_{img_id}.png'
                
                if clean_path.exists() and mask_path.exists():
                    self.samples.append({
                        'watermarked': str(img_path),
                        'clean': str(clean_path),
                        'mask': str(mask_path)
                    })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load images
        watermarked = Image.open(sample['watermarked']).convert('RGB')
        clean = Image.open(sample['clean']).convert('RGB')
        mask = Image.open(sample['mask']).convert('L')
        
        if self.transform:
            watermarked = self.transform(watermarked)
            clean = self.transform(clean)
            mask = self.transform(mask)
        
        return {
            'watermarked': watermarked,
            'clean': clean,
            'mask': mask
        }

# ==================== Models ====================
class TransparentWatermarkDetector(nn.Module):
    """Transparent watermark detection model - based on U-Net++ and attention"""
    
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        
        # Encoder - pretrained ResNet backbone
        resnet = models.resnet34(pretrained=True)
        
        # Adapt the first layer to the input
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        
        # ResNet layers
        self.encoder1 = resnet.layer1  # 64
        self.encoder2 = resnet.layer2  # 128
        self.encoder3 = resnet.layer3  # 256
        self.encoder4 = resnet.layer4  # 512
        
        # Attention modules
        self.attention1 = self._make_attention_block(64)
        self.attention2 = self._make_attention_block(128)
        self.attention3 = self._make_attention_block(256)
        self.attention4 = self._make_attention_block(512)
        
        # Decoder - U-Net++ style
        self.decoder4 = self._make_decoder_block(512, 256)
        self.decoder3 = self._make_decoder_block(256 + 256, 128)
        self.decoder2 = self._make_decoder_block(128 + 128, 64)
        self.decoder1 = self._make_decoder_block(64 + 64, 32)
        
        # Output layer
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, out_channels, kernel_size=1),
            nn.Sigmoid()
        )
    
    def _make_attention_block(self, channels):
        """Create an attention block"""
        return nn.Sequential(
            nn.Conv2d(channels, channels // 8, kernel_size=1),
            nn.BatchNorm2d(channels // 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )
    
    def _make_decoder_block(self, in_channels, out_channels):
        """Create a decoder block"""
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoding path
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        
        # Encoder layers + attention
        e1 = self.encoder1(x)
        e1 = e1 * self.attention1(e1)
        
        e2 = self.encoder2(e1)
        e2 = e2 * self.attention2(e2)
        
        e3 = self.encoder3(e2)
        e3 = e3 * self.attention3(e3)
        
        e4 = self.encoder4(e3)
        e4 = e4 * self.attention4(e4)
        
        # Decoding path
        d4 = self.decoder4(e4)
        d3 = self.decoder3(torch.cat([d4, e3], dim=1))
        d2 = self.decoder2(torch.cat([d3, e2], dim=1))
        d1 = self.decoder1(torch.cat([d2, e1], dim=1))
        
        # Output
        out = self.final_conv(d1)
        return out

class WatermarkInpainter(nn.Module):
    """Watermark inpainting model - based on improved PConv"""
    
    def __init__(self):
        super().__init__()
        
        # Encoder
        self.enc1 = self._make_encoder_block(3, 64, 7)
        self.enc2 = self._make_encoder_block(64, 128, 5)
        self.enc3 = self._make_encoder_block(128, 256, 5)
        self.enc4 = self._make_encoder_block(256, 512, 3)
        self.enc5 = self._make_encoder_block(512, 512, 3)
        self.enc6 = self._make_encoder_block(512, 512, 3)
        self.enc7 = self._make_encoder_block(512, 512, 3)
        self.enc8 = self._make_encoder_block(512, 512, 3)
        
        # Decoder
        self.dec8 = self._make_decoder_block(512 + 512, 512, 3)
        self.dec7 = self._make_decoder_block(512 + 512, 512, 3)
        self.dec6 = self._make_decoder_block(512 + 512, 512, 3)
        self.dec5 = self._make_decoder_block(512 + 512, 512, 3)
        self.dec4 = self._make_decoder_block(512 + 256, 256, 3)
        self.dec3 = self._make_decoder_block(256 + 128, 128, 3)
        self.dec2 = self._make_decoder_block(128 + 64, 64, 3)
        self.dec1 = self._make_decoder_block(64 + 3, 3, 3)
        
        # Refinement network
        self.refine = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )
    
    def _make_encoder_block(self, in_channels, out_channels, kernel_size):
        """Create an encoder block"""
        padding = kernel_size // 2
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=2, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _make_decoder_block(self, in_channels, out_channels, kernel_size):
        """Create a decoder block"""
        padding = kernel_size // 2
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x, mask):
        # Encode
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        e7 = self.enc7(e6)
        e8 = self.enc8(e7)
        
        # Decode
        d8 = self.dec8(torch.cat([e8, e7], dim=1))
        d7 = self.dec7(torch.cat([d8, e6], dim=1))
        d6 = self.dec6(torch.cat([d7, e5], dim=1))
        d5 = self.dec5(torch.cat([d6, e4], dim=1))
        d4 = self.dec4(torch.cat([d5, e3], dim=1))
        d3 = self.dec3(torch.cat([d4, e2], dim=1))
        d2 = self.dec2(torch.cat([d3, e1], dim=1))
        d1 = self.dec1(torch.cat([d2, x], dim=1))
        
        # Refine
        refined = self.refine(d1)
        
        # Blend the original image with the inpainted result
        output = x * (1 - mask) + refined * mask
        
        return output

class CombinedWatermarkModel(nn.Module):
    """Combined model: detection + inpainting"""
    
    def __init__(self):
        super().__init__()
        self.detector = TransparentWatermarkDetector()
        self.inpainter = WatermarkInpainter()
    
    def forward(self, x, mode='both'):
        if mode == 'detect':
            return self.detector(x)
        elif mode == 'inpaint':
            mask = self.detector(x)
            return self.inpainter(x, mask)
        else:  # both
            mask = self.detector(x)
            inpainted = self.inpainter(x, mask)
            return mask, inpainted

# ==================== Loss Functions ====================
class CombinedLoss(nn.Module):
    """Combined loss function"""
    
    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.ssim_loss = self._ssim_loss
    
    def _ssim_loss(self, x, y):
        """SSIM loss"""
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        mu_x = F.avg_pool2d(x, 3, 1, padding=1)
        mu_y = F.avg_pool2d(y, 3, 1, padding=1)
        
        sigma_x = F.avg_pool2d(x ** 2, 3, 1, padding=1) - mu_x ** 2
        sigma_y = F.avg_pool2d(y ** 2, 3, 1, padding=1) - mu_y ** 2
        sigma_xy = F.avg_pool2d(x * y, 3, 1, padding=1) - mu_x * mu_y
        
        ssim_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        ssim_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
        
        ssim = ssim_n / ssim_d
        return 1 - ssim.mean()
    
    def forward(self, pred_mask, true_mask, pred_image, true_image):
        """Compute the combined loss"""
        # Detection loss
        detect_loss = self.bce_loss(pred_mask, true_mask)
        
        # Inpainting loss
        l1_loss = self.l1_loss(pred_image, true_image)
        l2_loss = self.l2_loss(pred_image, true_image)
        ssim_loss = self.ssim_loss(pred_image, true_image)
        
        # Perceptual loss (using VGG features)
        perceptual_loss = 0  # VGG perceptual loss could be added here
        
        # Total loss
        total_loss = detect_loss + l1_loss + 0.5 * l2_loss + 0.1 * ssim_loss
        
        return {
            'total': total_loss,
            'detect': detect_loss,
            'l1': l1_loss,
            'l2': l2_loss,
            'ssim': ssim_loss
        }

# ==================== Trainer ====================
class Trainer:
    """Trainer"""
    
    def __init__(self, model, config: Config):
        self.model = model.to(config.DEVICE)
        self.config = config
        self.device = config.DEVICE
        
        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
        
        # Loss function
        self.criterion = CombinedLoss()
        
        # Create directories
        os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(config.LOG_DIR, exist_ok=True)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'best_val_loss': float('inf')
        }
    
    def train_epoch(self, dataloader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(dataloader, desc='Training')
        for batch in pbar:
            watermarked = batch['watermarked'].to(self.device)
            clean = batch['clean'].to(self.device)
            mask = batch['mask'].to(self.device)
            
            # Forward pass
            pred_mask, pred_clean = self.model(watermarked, mode='both')
            
            # Compute loss
            losses = self.criterion(pred_mask, mask, pred_clean, clean)
            loss = losses['total']
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        return total_loss / len(dataloader)
    
    def validate(self, dataloader):
        """Validate"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            pbar = tqdm(dataloader, desc='Validation')
            for batch in pbar:
                watermarked = batch['watermarked'].to(self.device)
                clean = batch['clean'].to(self.device)
                mask = batch['mask'].to(self.device)
                
                # Forward pass
                pred_mask, pred_clean = self.model(watermarked, mode='both')
                
                # Compute loss
                losses = self.criterion(pred_mask, mask, pred_clean, clean)
                loss = losses['total']
                
                total_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
        
        return total_loss / len(dataloader)
    
    def train(self, train_loader, val_loader, epochs):
        """Full training loop"""
        print(f"Starting training on {self.device}")
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # Train
            train_loss = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)
            
            # Validate
            val_loss = self.validate(val_loader)
            self.history['val_loss'].append(val_loss)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Save the best model
            if val_loss < self.history['best_val_loss']:
                self.history['best_val_loss'] = val_loss
                self.save_checkpoint(epoch, val_loss, is_best=True)
            
            # Periodic save
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch, val_loss, is_best=False)
        
        # Save training history
        self.save_history()
        print("Training completed!")
    
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        """Save a checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'history': self.history
        }
        
        if is_best:
            path = os.path.join(self.config.CHECKPOINT_DIR, 'best_model.pth')
        else:
            path = os.path.join(self.config.CHECKPOINT_DIR, f'checkpoint_epoch_{epoch+1}.pth')
        
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path):
        """Load a checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        print(f"Loaded checkpoint from {path}")
        return checkpoint['epoch']
    
    def save_history(self):
        """Save training history"""
        history_path = os.path.join(self.config.LOG_DIR, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=4)
        
        # Plot the loss curve
        plt.figure(figsize=(10, 5))
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['val_loss'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.config.LOG_DIR, 'loss_curve.png'))
        plt.close()

# ==================== Data Generator ====================
class DatasetGenerator:
    """Dataset generator"""
    
    def __init__(self, config: Config):
        self.config = config
        self.generator = WatermarkGenerator(config)
        
    def generate_background_images(self, num_images: int) -> List[Image.Image]:
        """Generate or load background images"""
        backgrounds = []
        
        for i in range(num_images):
            # Generate a random background (can be replaced with real images)
            bg = Image.new('RGB', (self.config.IMAGE_SIZE, self.config.IMAGE_SIZE))
            pixels = np.random.randint(0, 255, 
                (self.config.IMAGE_SIZE, self.config.IMAGE_SIZE, 3), dtype=np.uint8)
            
            # Add some patterns to make the background more realistic
            if i % 3 == 0:
                # Gradient background
                for y in range(self.config.IMAGE_SIZE):
                    color_value = int(255 * (y / self.config.IMAGE_SIZE))
                    pixels[y, :] = [color_value, color_value * 0.8, color_value * 0.6]
            elif i % 3 == 1:
                # Noise background
                noise = np.random.normal(128, 30, 
                    (self.config.IMAGE_SIZE, self.config.IMAGE_SIZE, 3))
                pixels = np.clip(noise, 0, 255).astype(np.uint8)
            
            bg = Image.fromarray(pixels)
            
            # Apply blur to make the background more natural
            bg = bg.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 2)))
            
            backgrounds.append(bg)
        
        return backgrounds
    
    def generate_dataset(self, output_dir: str):
        """Generate the full dataset"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create train and val directories
        train_dir = output_path / 'train'
        val_dir = output_path / 'val'
        train_dir.mkdir(exist_ok=True)
        val_dir.mkdir(exist_ok=True)
        
        # Generate background images
        print("Generating background images...")
        backgrounds = self.generate_background_images(self.config.DATASET_SIZE)
        
        # Split the dataset
        split_idx = int(self.config.DATASET_SIZE * self.config.TRAIN_SPLIT)
        train_backgrounds = backgrounds[:split_idx]
        val_backgrounds = backgrounds[split_idx:]
        
        # Generate training data
        print("Generating training data...")
        self._generate_samples(train_backgrounds, train_dir, "train")
        
        # Generate validation data
        print("Generating validation data...")
        self._generate_samples(val_backgrounds, val_dir, "val")
        
        print(f"Dataset generated successfully in {output_dir}")
        print(f"Training samples: {len(train_backgrounds)}")
        print(f"Validation samples: {len(val_backgrounds)}")
    
    def _generate_samples(self, backgrounds: List[Image.Image], output_dir: Path, prefix: str):
        """Generate samples"""
        for idx, bg in enumerate(tqdm(backgrounds, desc=f"Generating {prefix} samples")):
            # Generate watermark
            watermark, mask = self.generator.generate_watermark(
                (self.config.IMAGE_SIZE, self.config.IMAGE_SIZE)
            )
            
            # Apply watermark
            watermarked = self.generator.apply_watermark(bg, watermark)
            
            # Save images
            sample_id = f"{idx:05d}"
            watermarked.save(output_dir / f"watermarked_{sample_id}.png")
            bg.save(output_dir / f"clean_{sample_id}.png")
            Image.fromarray(mask).save(output_dir / f"mask_{sample_id}.png")

# ==================== Inference and Testing ====================
class Inference:
    """Inference wrapper"""
    
    def __init__(self, model_path: str, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = CombinedWatermarkModel().to(self.device)
        
        # Load model weights
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])
    
    def detect_watermark(self, image_path: str) -> np.ndarray:
        """Detect a watermark"""
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            mask = self.model(image_tensor, mode='detect')
            mask = mask.squeeze().cpu().numpy()
        
        return mask
    
    def remove_watermark(self, image_path: str, save_path: str = None):
        """Remove a watermark"""
        image = Image.open(image_path).convert('RGB')
        original_size = image.size
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            mask, cleaned = self.model(image_tensor, mode='both')
        
        # Convert back to a PIL image
        cleaned = cleaned.squeeze().cpu()
        cleaned = transforms.ToPILImage()(cleaned)
        cleaned = cleaned.resize(original_size, Image.LANCZOS)
        
        if save_path:
            cleaned.save(save_path)
            print(f"Cleaned image saved to {save_path}")
        
        return cleaned, mask.squeeze().cpu().numpy()
    
    def batch_process(self, input_dir: str, output_dir: str):
        """Batch-process images"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        image_files = list(input_path.glob('*.png')) + list(input_path.glob('*.jpg'))
        
        for img_path in tqdm(image_files, desc="Processing images"):
            try:
                cleaned, mask = self.remove_watermark(str(img_path))
                
                # Save the result
                output_name = output_path / f"cleaned_{img_path.name}"
                cleaned.save(output_name)
                
                # Save the mask (optional)
                mask_name = output_path / f"mask_{img_path.stem}.png"
                mask_img = Image.fromarray((mask * 255).astype(np.uint8))
                mask_img.save(mask_name)
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

# ==================== Main Program ====================
def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Transparent Watermark Detection and Removal')
    parser.add_argument('--mode', type=str, choices=['generate', 'train', 'test', 'infer'], 
                       default='train', help='Mode of operation')
    parser.add_argument('--data_dir', type=str, default='./data', 
                       help='Data directory')
    parser.add_argument('--checkpoint', type=str, default=None, 
                       help='Checkpoint path for resume training or inference')
    parser.add_argument('--input', type=str, help='Input image/directory for inference')
    parser.add_argument('--output', type=str, help='Output directory for results')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    
    args = parser.parse_args()
    
    # Create configuration
    config = Config()
    config.EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size
    config.LEARNING_RATE = args.lr
    
    if args.mode == 'generate':
        # Generate dataset
        print("Generating dataset...")
        generator = DatasetGenerator(config)
        generator.generate_dataset(args.data_dir)
        
    elif args.mode == 'train':
        # Train model
        print("Starting training...")
        
        # Check whether the dataset exists
        if not Path(args.data_dir).exists():
            print("Dataset not found. Generating dataset first...")
            generator = DatasetGenerator(config)
            generator.generate_dataset(args.data_dir)
        
        # Prepare data loaders
        transform = transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor()
        ])
        
        train_dataset = WatermarkDataset(args.data_dir, mode='train', transform=transform)
        val_dataset = WatermarkDataset(args.data_dir, mode='val', transform=transform)
        
        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, 
                                shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, 
                              shuffle=False, num_workers=4)
        
        # Create model and trainer
        model = CombinedWatermarkModel()
        trainer = Trainer(model, config)
        
        # Resume training (if a checkpoint is provided)
        start_epoch = 0
        if args.checkpoint:
            start_epoch = trainer.load_checkpoint(args.checkpoint)
        
        # Start training
        trainer.train(train_loader, val_loader, config.EPOCHS - start_epoch)
        
    elif args.mode == 'test':
        # Test model
        if not args.checkpoint:
            args.checkpoint = os.path.join(config.CHECKPOINT_DIR, 'best_model.pth')
        
        print(f"Testing model from {args.checkpoint}")
        inference = Inference(args.checkpoint)
        
        # Test on the validation set
        transform = transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor()
        ])
        
        val_dataset = WatermarkDataset(args.data_dir, mode='val', transform=transform)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
        
        # Evaluate performance
        total_psnr = 0
        total_ssim = 0
        
        for batch in tqdm(val_loader, desc="Testing"):
            watermarked = batch['watermarked']
            clean = batch['clean']
            
            # Inference
            with torch.no_grad():
                model = inference.model
                _, pred_clean = model(watermarked.to(inference.device), mode='both')
            
            # Compute metrics (simplified here; a real evaluation needs more detail)
            # TODO: add PSNR and SSIM computation
            
        print("Testing completed!")
        
    elif args.mode == 'infer':
        # Inference mode
        if not args.checkpoint:
            args.checkpoint = os.path.join(config.CHECKPOINT_DIR, 'best_model.pth')
        
        if not args.input:
            print("Please provide input image or directory with --input")
            return
        
        print(f"Running inference with model from {args.checkpoint}")
        inference = Inference(args.checkpoint)
        
        if os.path.isfile(args.input):
            # Single image
            output_path = args.output or "cleaned_output.png"
            inference.remove_watermark(args.input, output_path)
        else:
            # Batch processing
            output_dir = args.output or "./output"
            inference.batch_process(args.input, output_dir)

if __name__ == "__main__":
    main()