#!/usr/bin/env python3
"""
Extract individual images from the 10x10 grid PNG.
Each of the 100 images will be saved as a separate PNG file.
"""

import os
import torch
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np

# Configuration
GRID_IMAGE_PATH = "vis_output/original_100_images.png"
OUTPUT_DIR = "vis_output/extracted_images"
GRID_ROWS = 10
GRID_COLS = 10
IMAGE_SIZE = 28  # Original image size (28x28)
PADDING = 2  # Padding between images in the grid

def extract_images_from_grid():
    """Extract 100 individual images from the grid PNG."""
    
    # Check if grid image exists
    if not os.path.exists(GRID_IMAGE_PATH):
        print(f"Error: Grid image not found at {GRID_IMAGE_PATH}")
        return False
    
    # Load the grid image using torchvision
    print(f"Loading grid image from {GRID_IMAGE_PATH}...")
    transform = transforms.ToTensor()
    
    # Use PIL through torchvision's image loader
    from torchvision.io import read_image
    try:
        # Try reading as image file
        grid_tensor = read_image(GRID_IMAGE_PATH).float() / 255.0
        grid_height, grid_width = grid_tensor.shape[1], grid_tensor.shape[2]
    except:
        # Fallback: use PIL if available
        try:
            from PIL import Image
            grid_img = Image.open(GRID_IMAGE_PATH)
            grid_width, grid_height = grid_img.size
            grid_tensor = transform(grid_img)
        except ImportError:
            print("Error: Need PIL (Pillow) or torchvision.io to load images")
            print("Trying alternative: load from original dataset and extract...")
            return extract_from_original_dataset()
    
    print(f"Grid image size: {grid_width}x{grid_height}")
    
    # Calculate cell dimensions
    # With padding=2, each cell is: image_size + padding on each side (but padding is only between cells)
    # For a 10x10 grid: 10 images * 28px + 9 gaps * 2px = 280 + 18 = 298px
    cell_width = (grid_width - (GRID_COLS - 1) * PADDING) // GRID_COLS
    cell_height = (grid_height - (GRID_ROWS - 1) * PADDING) // GRID_ROWS
    
    print(f"Calculated cell size: {cell_width}x{cell_height}")
    print(f"Expected: {IMAGE_SIZE}x{IMAGE_SIZE}")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Extract each image
    extracted_count = 0
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            # Calculate position in grid
            x_start = col * (cell_width + PADDING)
            y_start = row * (cell_height + PADDING)
            x_end = x_start + cell_width
            y_end = y_start + cell_height
            
            # Extract the image cell from tensor
            img_cell = grid_tensor[:, y_start:y_end, x_start:x_end]
            
            # Save individual image
            image_idx = row * GRID_COLS + col
            output_path = os.path.join(OUTPUT_DIR, f"image_{image_idx:03d}.png")
            save_image(img_cell, output_path)
            extracted_count += 1
            
            if (extracted_count % 10 == 0) or extracted_count == 100:
                print(f"Extracted {extracted_count}/100 images...")
    
    print(f"\nExtraction complete! Extracted {extracted_count} images.")
    
    # Verify: count files in output directory
    files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.png')]
    file_count = len(files)
    
    print(f"\nVerification:")
    print(f"  Files in output directory: {file_count}")
    print(f"  Expected: 100")
    
    if file_count == 100:
        print("  ✓ SUCCESS: All 100 images extracted correctly!")
        return True
    else:
        print(f"  ✗ ERROR: Expected 100 files, found {file_count}")
        return False

def extract_from_original_dataset():
    """Alternative: Extract directly from the original dataset."""
    print("\nAlternative approach: Loading from original dataset...")
    
    if not os.path.exists("natural_images.pt"):
        print("Error: natural_images.pt not found")
        return False
    
    data = torch.load("natural_images.pt", weights_only=False)
    images = data["images"]  # Shape: (100, 3, 28, 28)
    
    print(f"Loaded {len(images)} images from dataset")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Save each image individually
    for i in range(len(images)):
        output_path = os.path.join(OUTPUT_DIR, f"image_{i:03d}.png")
        save_image(images[i:i+1], output_path)
        
        if (i + 1) % 10 == 0 or i == 99:
            print(f"Saved {i+1}/100 images...")
    
    # Verify
    files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.png')]
    file_count = len(files)
    
    print(f"\nVerification:")
    print(f"  Files in output directory: {file_count}")
    print(f"  Expected: 100")
    
    if file_count == 100:
        print("  ✓ SUCCESS: All 100 images extracted correctly!")
        return True
    else:
        print(f"  ✗ ERROR: Expected 100 files, found {file_count}")
        return False

if __name__ == "__main__":
    success = extract_images_from_grid()
    exit(0 if success else 1)

