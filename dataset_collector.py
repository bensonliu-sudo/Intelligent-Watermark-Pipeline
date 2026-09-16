"""
Fixed dataset collector
Works around network download issues and offers multiple ways to obtain data
"""

import requests
import os
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw
import random
import json
from urllib.parse import urlparse
import time

class FixedDatasetCollector:
    """Fixed dataset collector"""
    
    def __init__(self, output_folder="extended_dataset"):
        self.output_folder = Path(output_folder)
        self.clean_folder = self.output_folder / "clean"
        self.watermarked_folder = self.output_folder / "watermarked"
        
        # Create folders
        self.clean_folder.mkdir(parents=True, exist_ok=True)
        self.watermarked_folder.mkdir(parents=True, exist_ok=True)
        
        print(f"Dataset will be saved to: {self.output_folder}")
    
    def test_internet_connection(self):
        """Test network connectivity"""
        test_urls = [
            "https://httpbin.org/get",
            "https://www.google.com",
            "https://www.baidu.com"
        ]
        
        for url in test_urls:
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    print(f"Network connection OK: {url}")
                    return True
            except Exception as e:
                print(f"Failed to connect to {url}: {e}")
                continue
        
        print("Network unavailable; cannot download images")
        return False
    
    def download_with_multiple_sources(self, count=100):
        """Download from multiple image sources"""
        print(f"Trying to download {count} images from multiple sources...")
        
        if not self.test_internet_connection():
            print("Skipping network download; consider using the local image expansion option")
            return 0
        
        downloaded = 0
        
        # Method 1: random images from Unsplash
        downloaded += self.download_from_unsplash(count // 3)
        
        # Method 2: Lorem Picsum (another free image service)
        downloaded += self.download_from_picsum(count // 3)
        
        # Method 3: fallback generation (if the above fail)
        if downloaded < count // 2:
            downloaded += self.download_placeholder_images(count - downloaded)
        
        return downloaded
    
    def download_from_unsplash(self, count):
        """Download from Unsplash"""
        print(f"Trying to download {count} images from Unsplash...")
        downloaded = 0
        
        # Several different Unsplash endpoints
        unsplash_urls = [
            "https://source.unsplash.com/{width}x{height}/?nature&{seed}",
            "https://source.unsplash.com/{width}x{height}/?architecture&{seed}",
            "https://source.unsplash.com/{width}x{height}/?people&{seed}",
            "https://source.unsplash.com/{width}x{height}/?technology&{seed}"
        ]
        
        for i in range(count * 2):  # try a few extra
            if downloaded >= count:
                break
            
            try:
                # Pick a random URL template
                url_template = random.choice(unsplash_urls)
                url = url_template.format(width=800, height=600, seed=i)
                
                print(f"Downloading: {url}")
                
                response = requests.get(url, timeout=15, allow_redirects=True)
                if response.status_code == 200 and len(response.content) > 10000:  # make sure it is not an empty file
                    filename = f"unsplash_{i:04d}.jpg"
                    filepath = self.clean_folder / filename
                    
                    with open(filepath, 'wb') as f:
                        f.write(response.content)
                    
                    # Verify the image is valid
                    if self.validate_image(filepath):
                        downloaded += 1
                        print(f"Downloaded: {filename} ({downloaded}/{count})")
                    else:
                        os.remove(filepath)
                        print(f"Invalid image removed: {filename}")
                    
                    time.sleep(1)  # add a delay
                else:
                    print(f"Download failed: HTTP {response.status_code}")
                
            except Exception as e:
                print(f"Download error: {e}")
                time.sleep(0.5)
                continue
        
        print(f"Downloaded {downloaded} images from Unsplash")
        return downloaded
    
    def download_from_picsum(self, count):
        """Download from Lorem Picsum"""
        print(f"Trying to download {count} images from Lorem Picsum...")
        downloaded = 0
        
        for i in range(count * 2):
            if downloaded >= count:
                break
            
            try:
                # Lorem Picsum API
                url = f"https://picsum.photos/800/600?random={i}"
                
                response = requests.get(url, timeout=15)
                if response.status_code == 200:
                    filename = f"picsum_{i:04d}.jpg"
                    filepath = self.clean_folder / filename
                    
                    with open(filepath, 'wb') as f:
                        f.write(response.content)
                    
                    if self.validate_image(filepath):
                        downloaded += 1
                        print(f"Downloaded: {filename} ({downloaded}/{count})")
                    else:
                        os.remove(filepath)
                    
                    time.sleep(0.8)
                
            except Exception as e:
                print(f"Picsum download failed: {e}")
                continue
        
        print(f"Downloaded {downloaded} images from Lorem Picsum")
        return downloaded
    
    def download_placeholder_images(self, count):
        """Generate placeholder images (last-resort fallback)"""
        print(f"Generating {count} placeholder images...")
        generated = 0
        
        for i in range(count):
            try:
                # Generate a colored noise image
                img = self.generate_random_image(800, 600)
                filename = f"generated_{i:04d}.jpg"
                filepath = self.clean_folder / filename
                
                cv2.imwrite(str(filepath), img)
                generated += 1
                
            except Exception as e:
                print(f"Failed to generate placeholder: {e}")
        
        print(f"Generated {generated} placeholder images")
        return generated
    
    def generate_random_image(self, width, height):
        """Generate a random image"""
        # Create a colored gradient background
        img = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Random background colors
        color1 = (random.randint(50, 200), random.randint(50, 200), random.randint(50, 200))
        color2 = (random.randint(50, 200), random.randint(50, 200), random.randint(50, 200))
        
        # Build the gradient
        for y in range(height):
            ratio = y / height
            blended_color = [
                int(color1[i] * (1 - ratio) + color2[i] * ratio) for i in range(3)
            ]
            img[y, :] = blended_color
        
        # Add some random shapes
        num_shapes = random.randint(3, 8)
        for _ in range(num_shapes):
            shape_type = random.choice(['circle', 'rectangle', 'line'])
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            
            if shape_type == 'circle':
                center = (random.randint(0, width), random.randint(0, height))
                radius = random.randint(20, 100)
                cv2.circle(img, center, radius, color, -1)
            elif shape_type == 'rectangle':
                pt1 = (random.randint(0, width), random.randint(0, height))
                pt2 = (random.randint(0, width), random.randint(0, height))
                cv2.rectangle(img, pt1, pt2, color, -1)
            elif shape_type == 'line':
                pt1 = (random.randint(0, width), random.randint(0, height))
                pt2 = (random.randint(0, width), random.randint(0, height))
                cv2.line(img, pt1, pt2, color, random.randint(2, 8))
        
        return img
    
    def validate_image(self, filepath):
        """Check whether an image file is valid"""
        try:
            img = cv2.imread(str(filepath))
            if img is None:
                return False
            
            h, w = img.shape[:2]
            if h < 100 or w < 100:  # image too small
                return False
            
            return True
        except:
            return False
    
    def use_local_images(self, local_folder):
        """Use a local image folder"""
        local_path = Path(local_folder)
        if not local_path.exists():
            print(f"Local folder does not exist: {local_folder}")
            return 0
        
        # Supported image formats
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        
        for ext in extensions:
            image_files.extend(local_path.glob(f"*{ext}"))
            image_files.extend(local_path.glob(f"*{ext.upper()}"))
        
        print(f"Found {len(image_files)} images in {local_folder}")
        
        # Copy into the clean folder
        copied = 0
        for i, img_file in enumerate(image_files):
            try:
                # Validate the image
                if self.validate_image(img_file):
                    # Copy and rename
                    new_name = f"local_{i:04d}{img_file.suffix}"
                    new_path = self.clean_folder / new_name
                    
                    # Re-save with cv2 to normalize the format
                    img = cv2.imread(str(img_file))
                    cv2.imwrite(str(new_path), img)
                    
                    copied += 1
                    print(f"Copied: {img_file.name} -> {new_name}")
                
            except Exception as e:
                print(f"Failed to process {img_file.name}: {e}")
        
        print(f"Processed {copied} local images")
        return copied

class EnhancedWatermarkGenerator:
    """Enhanced watermark generator"""
    
    def __init__(self):
        # More diverse watermark texts
        self.watermark_texts = [
            "SAMPLE", "PREVIEW", "WATERMARK", "COPYRIGHT", "DEMO", "DRAFT",
            "FOR PREVIEW ONLY", "© 2024", "CONFIDENTIAL", "PROTECTED",
            "NOT FOR SALE", "EVALUATION COPY", "PROOF", "RESTRICTED",
            "测试", "样本", "预览", "版权所有", "演示", "草稿"
        ]
        
        # More color choices
        self.colors = [
            (255, 255, 255),  # white
            (0, 0, 0),        # black
            (255, 0, 0),      # blue (BGR)
            (0, 255, 0),      # green
            (0, 0, 255),      # red (BGR)
            (255, 255, 0),    # cyan (BGR)
            (255, 0, 255),    # magenta
            (0, 255, 255),    # yellow (BGR)
            (128, 128, 128),  # gray
            (64, 64, 64),     # dark gray
            (192, 192, 192),  # light gray
        ]
    
    def generate_diverse_watermarks(self, image, count=15):
        """Generate a more diverse set of watermarks"""
        watermarks = []
        h, w = image.shape[:2]
        
        # Text watermarks (40%)
        text_count = int(count * 0.4)
        watermarks.extend(self.generate_text_watermarks(image, text_count))
        
        # Shape watermarks (30%)
        shape_count = int(count * 0.3)
        watermarks.extend(self.generate_shape_watermarks(image, shape_count))
        
        # Pattern watermarks (20%)
        pattern_count = int(count * 0.2)
        watermarks.extend(self.generate_pattern_watermarks(image, pattern_count))
        
        # Combined watermarks (10%)
        combo_count = count - len(watermarks)
        watermarks.extend(self.generate_combo_watermarks(image, combo_count))
        
        return watermarks
    
    def generate_text_watermarks(self, image, count):
        """Generate text watermarks"""
        watermarks = []
        h, w = image.shape[:2]
        
        for _ in range(count):
            img_copy = image.copy()
            
            # Random text and style
            text = random.choice(self.watermark_texts)
            font_scale = random.uniform(0.6, 3.0)
            thickness = random.randint(1, 4)
            color = random.choice(self.colors)
            
            # Random position - smarter placement
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            
            # Keep the text within the image bounds
            max_x = max(1, w - text_size[0])
            max_y = max(text_size[1], h)
            
            x = random.randint(0, max_x)
            y = random.randint(text_size[1], max_y)
            
            # Add text
            cv2.putText(img_copy, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 
                       font_scale, color, thickness)
            
            # Randomly add a background or border
            if random.random() < 0.3:
                # Add a text background
                background_color = tuple(255 - c for c in color)  # inverted color
                cv2.rectangle(img_copy, (x-5, y-text_size[1]-5), 
                             (x+text_size[0]+5, y+5), background_color, -1)
                cv2.putText(img_copy, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 
                           font_scale, color, thickness)
            
            # Random transparency
            if random.random() < 0.5:
                alpha = random.uniform(0.3, 0.8)
                img_copy = cv2.addWeighted(image, alpha, img_copy, 1-alpha, 0)
            
            watermarks.append(img_copy)
        
        return watermarks
    
    def generate_shape_watermarks(self, image, count):
        """Generate shape watermarks"""
        watermarks = []
        h, w = image.shape[:2]
        
        for _ in range(count):
            img_copy = image.copy()
            
            # Random shape parameters
            num_shapes = random.randint(1, 5)
            
            for _ in range(num_shapes):
                color = random.choice(self.colors)
                shape_type = random.choice(['rectangle', 'circle', 'ellipse', 'triangle', 'line'])
                
                if shape_type == 'rectangle':
                    x1, y1 = random.randint(0, w), random.randint(0, h)
                    x2, y2 = random.randint(0, w), random.randint(0, h)
                    cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, random.randint(-1, 5))
                
                elif shape_type == 'circle':
                    center = (random.randint(0, w), random.randint(0, h))
                    radius = random.randint(20, min(w, h) // 8)
                    cv2.circle(img_copy, center, radius, color, random.randint(-1, 5))
                
                elif shape_type == 'ellipse':
                    center = (random.randint(0, w), random.randint(0, h))
                    axes = (random.randint(20, w//8), random.randint(20, h//8))
                    angle = random.randint(0, 360)
                    cv2.ellipse(img_copy, center, axes, angle, 0, 360, color, random.randint(-1, 5))
                
                elif shape_type == 'triangle':
                    pts = np.array([
                        [random.randint(0, w), random.randint(0, h)],
                        [random.randint(0, w), random.randint(0, h)],
                        [random.randint(0, w), random.randint(0, h)]
                    ], np.int32)
                    cv2.fillPoly(img_copy, [pts], color)
                
                elif shape_type == 'line':
                    pt1 = (random.randint(0, w), random.randint(0, h))
                    pt2 = (random.randint(0, w), random.randint(0, h))
                    cv2.line(img_copy, pt1, pt2, color, random.randint(2, 10))
            
            # Apply transparency
            alpha = random.uniform(0.3, 0.7)
            img_copy = cv2.addWeighted(image, alpha, img_copy, 1-alpha, 0)
            
            watermarks.append(img_copy)
        
        return watermarks
    
    def generate_pattern_watermarks(self, image, count):
        """Generate pattern watermarks"""
        watermarks = []
        h, w = image.shape[:2]
        
        for _ in range(count):
            img_copy = image.copy()
            
            pattern_type = random.choice(['grid', 'dots', 'diagonal', 'cross'])
            color = random.choice(self.colors)
            spacing = random.randint(30, 100)
            
            if pattern_type == 'grid':
                # Grid pattern
                for x in range(0, w, spacing):
                    cv2.line(img_copy, (x, 0), (x, h), color, 2)
                for y in range(0, h, spacing):
                    cv2.line(img_copy, (0, y), (w, y), color, 2)
            
            elif pattern_type == 'dots':
                # Dot pattern
                for x in range(spacing//2, w, spacing):
                    for y in range(spacing//2, h, spacing):
                        cv2.circle(img_copy, (x, y), random.randint(3, 8), color, -1)
            
            elif pattern_type == 'diagonal':
                # Diagonal pattern
                for offset in range(-h, w, spacing):
                    cv2.line(img_copy, (offset, 0), (offset + h, h), color, 3)
            
            elif pattern_type == 'cross':
                # Cross pattern
                for x in range(spacing//2, w, spacing):
                    for y in range(spacing//2, h, spacing):
                        size = random.randint(10, 20)
                        cv2.line(img_copy, (x-size, y), (x+size, y), color, 2)
                        cv2.line(img_copy, (x, y-size), (x, y+size), color, 2)
            
            # Apply transparency
            alpha = random.uniform(0.2, 0.6)
            img_copy = cv2.addWeighted(image, alpha, img_copy, 1-alpha, 0)
            
            watermarks.append(img_copy)
        
        return watermarks
    
    def generate_combo_watermarks(self, image, count):
        """Generate combined watermarks (text + shapes)"""
        watermarks = []
        
        for _ in range(count):
            img_copy = image.copy()
            
            # Add shapes first
            shape_watermarks = self.generate_shape_watermarks(img_copy, 1)
            if shape_watermarks:
                img_copy = shape_watermarks[0]
            
            # Then add text
            text_watermarks = self.generate_text_watermarks(img_copy, 1)
            if text_watermarks:
                img_copy = text_watermarks[0]
            
            watermarks.append(img_copy)
        
        return watermarks

def main():
    """Main entry point"""
    print("=== Fixed Watermark Dataset Expansion Tool ===")
    
    # Diagnose network connectivity
    collector = FixedDatasetCollector()
    
    print("\nSelect a data acquisition method:")
    print("1. Try downloading images from the internet (fixed)")
    print("2. Use a local image folder")
    print("3. Generate random placeholder images")
    print("4. Expand existing images (add watermarks)")
    
    choice = input("Enter choice (1-4): ").strip()
    
    if choice == "1":
        count = int(input("Number of images to download: ") or "50")
        downloaded = collector.download_with_multiple_sources(count)
        
        if downloaded > 0:
            print(f"Downloaded {downloaded} images; generating watermarked versions...")
            expander = DatasetExpander(collector.clean_folder)
            pairs = expander.expand_dataset_enhanced(downloaded * 15)
            print(f"Generated {pairs} training pairs in total")
        else:
            print("Download failed; consider using local images or placeholders")
    
    elif choice == "2":
        local_folder = input("Path to local image folder: ")
        copied = collector.use_local_images(local_folder)
        
        if copied > 0:
            print(f"Using {copied} local images; generating watermarked versions...")
            expander = DatasetExpander(collector.clean_folder)
            pairs = expander.expand_dataset_enhanced(copied * 15)
            print(f"Generated {pairs} training pairs in total")
    
    elif choice == "3":
        count = int(input("Number of placeholder images to generate: ") or "100")
        generated = collector.download_placeholder_images(count)
        
        print(f"Generated {generated} placeholder images; generating watermarked versions...")
        expander = DatasetExpander(collector.clean_folder)
        pairs = expander.expand_dataset_enhanced(generated * 15)
        print(f"Generated {pairs} training pairs in total")
    
    elif choice == "4":
        clean_folder = input("Path to existing image folder: ")
        target_pairs = int(input("Target number of data pairs: ") or "500")
        
        expander = DatasetExpander(clean_folder)
        pairs = expander.expand_dataset_enhanced(target_pairs)
        print(f"Generated {pairs} training pairs in total")

class DatasetExpander:
    """Dataset expander (uses the enhanced watermark generator)"""
    
    def __init__(self, clean_folder, output_folder="expanded_dataset"):
        self.clean_folder = Path(clean_folder)
        self.output_folder = Path(output_folder)
        self.watermark_gen = EnhancedWatermarkGenerator()
        
        # Create output folders for clean and watermarked images
        (self.output_folder / "clean").mkdir(parents=True, exist_ok=True)
        (self.output_folder / "watermarked").mkdir(parents=True, exist_ok=True)
    
    def expand_dataset_enhanced(self, target_pairs=500):
        """Expand the dataset using the enhanced watermark generator"""
        # Collect existing clean images
        clean_images = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            clean_images.extend(self.clean_folder.glob(f"*{ext}"))
        
        if not clean_images:
            print("No image files found")
            return 0
        
        print(f"Found {len(clean_images)} clean images")
        
        generated_pairs = 0
        
        for img_idx, clean_img_path in enumerate(clean_images):
            if generated_pairs >= target_pairs:
                break
            
            try:
                # Read the image
                clean_img = cv2.imread(str(clean_img_path))
                if clean_img is None:
                    continue
                
                # Resize (optional)
                h, w = clean_img.shape[:2]
                if max(h, w) > 1000:
                    scale = 1000 / max(h, w)
                    new_w, new_h = int(w * scale), int(h * scale)
                    clean_img = cv2.resize(clean_img, (new_w, new_h))
                
                # Decide how many watermarked versions to generate for this image
                remaining_pairs = target_pairs - generated_pairs
                max_versions = min(20, remaining_pairs)  # at most 20 versions per image
                
                # Generate watermarked versions
                watermarked_versions = self.watermark_gen.generate_diverse_watermarks(
                    clean_img, max_versions
                )
                
                # Save paired data
                for version_idx, watermarked in enumerate(watermarked_versions):
                    if generated_pairs >= target_pairs:
                        break
                    
                    # Uniform naming scheme
                    pair_id = f"{generated_pairs:05d}"
                    clean_name = f"img{pair_id}.jpg"
                    watermarked_name = f"img{pair_id}_wm.jpg"
                    
                    # Save into the corresponding folders
                    clean_path = self.output_folder / "clean" / clean_name
                    watermarked_path = self.output_folder / "watermarked" / watermarked_name
                    
                    cv2.imwrite(str(clean_path), clean_img)
                    cv2.imwrite(str(watermarked_path), watermarked)
                    
                    generated_pairs += 1
                    
                    if generated_pairs % 100 == 0:
                        print(f"Generated {generated_pairs}/{target_pairs} pairs")
                
                print(f"Image {img_idx+1}/{len(clean_images)} done; {generated_pairs} pairs generated so far")
                
            except Exception as e:
                print(f"Failed to process {clean_img_path}: {e}")
                continue
        
        print(f"Dataset expansion complete! Generated {generated_pairs} training pairs")
        print(f"Data saved to:")
        print(f"  Clean images: {self.output_folder}/clean/")
        print(f"  Watermarked images: {self.output_folder}/watermarked/")
        
        return generated_pairs

if __name__ == "__main__":
    main()