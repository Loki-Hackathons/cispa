"""
================================================================================
UPGRADED DUAL-STREAM FORENSIC NETWORK FOR WATERMARK DETECTION (V2)
================================================================================
Architecture: ForensicNet V2 with Paper-Based Upgrades

Stream A (Frequency): Polar FFT for Tree-Rings watermark detection
- Converts circular patterns in Fourier space to vertical lines
- Based on "Tree-Rings Watermarks" paper (NeurIPS 2023)

Stream B (Spatial): SRM Residuals for BitMark/Stable Signature detection
- Fixed SRM (Spatial Rich Models) filters for high-frequency noise
- Based on steganalysis research (BitMark, Stable Signature)

Inference: Test-Time Augmentation (TTA)
- Center Crop + Horizontal Flip + Vertical Flip -> Mean Score
- Reduces False Positives and stabilizes predictions

Metric: TPR @ 1% FPR (optimized during training)
Hardware: Multi-GPU with DataParallel
================================================================================
"""

import os
import sys
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from PIL import Image
import numpy as np
import pandas as pd
import requests
from pathlib import Path
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================
DATASET_DIR = Path("dataset")  # Path to unzipped dataset folder
SUBMISSION_FILE = "submission.csv"
BATCH_SIZE = 16  # Reduced for 512x512 images with dual-stream architecture
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 10  # Early stopping patience
GRADIENT_ACCUMULATION_STEPS = 2  # Accumulate gradients over 2 steps (effective batch size = 16 * 2 = 32)

# Device setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
if torch.cuda.is_available():
    print(f"Number of GPUs: {torch.cuda.device_count()}")

# FIX: Option to disable DataParallel if CUDA misaligned address errors occur
# Set USE_DATAPARALLEL = False to use single GPU (more stable with FFT/grid_sample)
USE_DATAPARALLEL = False  # Set to True to use multiple GPUs (may cause CUDA errors)

# Leaderboard submission
SERVER_URL = "http://34.122.51.94:80"
API_KEY = "f62b1499d4e2bf13ae56be5683c974c1"
TASK_ID = "08-watermark-detection"

# ============================================================================
# SRM FILTERS (Spatial Rich Models) for Steganalysis
# ============================================================================
class SRMConv2d(nn.Module):
    """
    Fixed SRM (Spatial Rich Models) filters for detecting high-frequency noise residuals.
    Used for BitMark and Stable Signature watermark detection.
    """
    def __init__(self):
        super(SRMConv2d, self).__init__()
        # FIX: Added groups=3 to handle the [3, 1, 5, 5] weights correctly on RGB input
        self.conv = nn.Conv2d(3, 3, kernel_size=5, stride=1, padding=2, bias=False, groups=3)
        
        # 3 Standard Steganalysis Filters (SRM)
        # Filter 1: KB (3rd order)
        f1 = np.array([[0, 0, 0, 0, 0],
                       [0, -1, 2, -1, 0],
                       [0, 2, -4, 2, 0],
                       [0, -1, 2, -1, 0],
                       [0, 0, 0, 0, 0]]) / 4.0
        
        # Filter 2: KV (Edge detection)
        f2 = np.array([[-1, 2, -2, 2, -1],
                       [2, -6, 8, -6, 2],
                       [-2, 8, -12, 8, -2],
                       [2, -6, 8, -6, 2],
                       [-1, 2, -2, 2, -1]]) / 12.0
        
        # Filter 3: Spam (Square 5x5)
        f3 = np.array([[0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0],
                       [0, 1, -2, 1, 0],
                       [0, -2, 4, -2, 0],
                       [0, 1, -2, 1, 0]]) / 4.0
        
        # Stack into (3, 1, 5, 5) for grouped conv (applied per channel)
        filters = np.stack([f1, f2, f3])
        filters = torch.from_numpy(filters).float().unsqueeze(1)
        
        self.conv.weight.data = filters
        # Important: Fix the weights (non-learnable preprocessing)
        for param in self.conv.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        """
        Apply SRM filters channel-wise
        
        Args:
            x: [Batch, 3, H, W] RGB image tensor
        
        Returns:
            residuals: [Batch, 3, H, W] High-frequency noise residuals
        """
        return self.conv(x.contiguous())

# ============================================================================
# UPGRADED DUAL-STREAM ARCHITECTURE (V2)
# ============================================================================
class ForensicNet(nn.Module):
    """
    Upgraded Dual-Stream Forensic Network for Watermark Detection
    
    Stream A (Frequency): Polar FFT for Tree-Rings watermark detection
    Stream B (Spatial): SRM Residuals for BitMark/Stable Signature detection
    """
    def __init__(self):
        super(ForensicNet, self).__init__()
        
        # Stream A: Frequency (Polar FFT)
        self.freq_backbone = models.efficientnet_v2_s(weights='DEFAULT')
        self.freq_backbone.classifier = nn.Identity()
        
        # Stream B: Spatial (SRM Residuals)
        self.srm_layer = SRMConv2d()
        self.spatial_backbone = models.efficientnet_v2_s(weights='DEFAULT')
        self.spatial_backbone.classifier = nn.Identity()
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(1280 * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),  # High dropout is critical for small data
            nn.Linear(512, 1)
        )
    
    def compute_polar_fft(self, x):
        """
        Compute Polar Log Magnitude Spectrum of FFT (Ultra-Stable Version)
        Converts circular Tree-Rings patterns into vertical lines for easier detection
        
        Args:
            x: [Batch, 3, H, W] RGB image tensor
        
        Returns:
            polar_mag: [Batch, 3, H, W] Normalized polar log magnitude spectrum
        """
        # 1. FFT Calculation
        # x is [B, 3, H, W]
        # Force input to be contiguous
        x = x.contiguous()
        
        # Compute FFT per channel
        fft = torch.fft.fft2(x)
        fft = torch.fft.fftshift(fft)
        
        # Manual Magnitude (Avoids complex number issues on some drivers)
        real = fft.real.contiguous()
        imag = fft.imag.contiguous()
        mag = torch.sqrt(real**2 + imag**2 + 1e-8)
        mag = torch.log(mag + 1e-6)
        
        # 2. Polar Transform setup
        B, C, H, W = mag.shape
        device = mag.device
        
        # Create grid directly on the correct device
        # (Previously this might have caused issues if created on CPU then moved)
        steps_r = torch.linspace(0, np.sqrt(2)/2, H, device=device)
        steps_theta = torch.linspace(0, np.pi, W, device=device)
        
        grid_r, grid_theta = torch.meshgrid(steps_r, steps_theta, indexing='ij')
        
        grid_x = grid_r * torch.cos(grid_theta)
        grid_y = grid_r * torch.sin(grid_theta)
        
        # Scale to [-1, 1] for grid_sample
        grid_x = grid_x * 2.0
        grid_y = grid_y * 2.0
        
        # Stack and expand (more efficient than repeat)
        grid = torch.stack((grid_x, grid_y), dim=-1)
        grid = grid.unsqueeze(0).expand(B, -1, -1, -1)
        
        # CRITICAL: Force contiguity before grid_sample
        mag = mag.contiguous()
        grid = grid.contiguous()
        
        # Sample
        polar_mag = F.grid_sample(mag, grid, align_corners=True, padding_mode='border')
        
        # Normalize
        return (polar_mag - polar_mag.mean(dim=(2, 3), keepdim=True)) / (polar_mag.std(dim=(2, 3), keepdim=True) + 1e-6)
    
    def forward(self, x):
        """
        Forward pass through upgraded dual-stream architecture
        
        Args:
            x: [Batch, 3, H, W] RGB image tensor
        
        Returns:
            logits: [Batch, 1] Binary classification logits
        """
        # FIX: Ensure input is contiguous for CUDA alignment (critical with DataParallel)
        x = x.contiguous()
        
        # Stream A: Polar FFT (Tree-Rings detection)
        x_polar = self.compute_polar_fft(x)
        feat_freq = self.freq_backbone(x_polar)
        
        # Stream B: SRM Residuals (BitMark/Stable Signature detection)
        x_srm = self.srm_layer(x)
        feat_spatial = self.spatial_backbone(x_srm)
        
        # Fusion
        combined = torch.cat([feat_freq, feat_spatial], dim=1)
        return self.fusion(combined)

# ============================================================================
# CRITICAL METRIC: TPR @ 1% FPR
# ============================================================================
def get_tpr_at_fpr(scores, labels, target_fpr=0.01):
    """
    Calculate True Positive Rate at a specific False Positive Rate
    
    Args:
        scores: Array of prediction scores (higher = more likely watermarked)
        labels: Array of true labels (1 = watermarked, 0 = clean)
        target_fpr: Target False Positive Rate (default: 0.01 = 1%)
    
    Returns:
        tpr_at_fpr: Maximum TPR achievable at target FPR
    """
    scores = np.array(scores)
    labels = np.array(labels)
    
    # Sort scores descending (higher scores first)
    desc_indices = np.argsort(scores)[::-1]
    scores = scores[desc_indices]
    labels = labels[desc_indices]
    
    # Cumulative true positives and false positives
    tps = np.cumsum(labels)
    fps = np.cumsum(1 - labels)
    
    # Calculate TPR and FPR
    total_positives = labels.sum()
    total_negatives = (1 - labels).sum()
    
    if total_positives == 0:
        return 0.0
    if total_negatives == 0:
        return 1.0
    
    tpr = tps / total_positives
    fpr = fps / total_negatives
    
    # Find max TPR where FPR <= target
    valid_mask = fpr <= target_fpr
    if valid_mask.any():
        return tpr[valid_mask].max()
    return 0.0

# ============================================================================
# DATA TRANSFORMS (NO RESIZE - CRITICAL!)
# ============================================================================
# Training: RandomCrop 512x512 (preserves watermark signals)
train_transform = transforms.Compose([
    transforms.RandomCrop(512),  # NO RESIZE - preserves invisible watermarks
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Validation/Test: CenterCrop 512x512
val_test_transform = transforms.Compose([
    transforms.CenterCrop(512),  # NO RESIZE - preserves invisible watermarks
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ============================================================================
# DATASET LOADING
# ============================================================================
print("Loading datasets...")
train_dataset = datasets.ImageFolder(root=DATASET_DIR / "train", transform=train_transform)
val_dataset = datasets.ImageFolder(root=DATASET_DIR / "val", transform=val_test_transform)

# Custom dataset for unlabeled test images
class TestDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.files = sorted(list(self.root.glob("*.png")))
        self.transform = transform
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        img_path = self.files[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return {"image": image, "image_name": img_path.name}

test_dataset = TestDataset(DATASET_DIR / "test", transform=val_test_transform)

train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    num_workers=4, 
    pin_memory=True
)
val_loader = DataLoader(
    val_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    num_workers=4, 
    pin_memory=True
)
test_loader = DataLoader(
    test_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    num_workers=4, 
    pin_memory=True
)

print(f"Train size: {len(train_dataset)} | Val size: {len(val_dataset)} | Test size: {len(test_dataset)}")

# ============================================================================
# MODEL INITIALIZATION
# ============================================================================
print("\n" + "=" * 80)
print("Initializing Dual-Stream ForensicNet...")
print("=" * 80)

model = ForensicNet().to(DEVICE)

# Wrap in DataParallel for multi-GPU training (if enabled and multiple GPUs available)
if USE_DATAPARALLEL and torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs with DataParallel!")
    model = nn.DataParallel(model)
else:
    if torch.cuda.device_count() > 1:
        print(f"DataParallel disabled. Using single GPU (GPU 0) for stability.")
        print(f"Note: {torch.cuda.device_count()} GPUs available but not used due to CUDA compatibility issues.")
    else:
        print("Using single GPU")

# ============================================================================
# LOSS, OPTIMIZER, SCHEDULER
# ============================================================================
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=LEARNING_RATE, 
    weight_decay=WEIGHT_DECAY
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='max', 
    factor=0.5, 
    patience=5
)

# ============================================================================
# TRAINING LOOP
# ============================================================================
print("\n" + "=" * 80)
print("Starting Training...")
print("Optimizing for: TPR @ 1% FPR")
print(f"Batch Size: {BATCH_SIZE} | Gradient Accumulation: {GRADIENT_ACCUMULATION_STEPS} steps")
print(f"Effective Batch Size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
print("Using Mixed Precision (AMP) for memory efficiency")
print("=" * 80)

# Initialize Mixed Precision scaler
scaler = torch.cuda.amp.GradScaler()

best_tpr_at_fpr = 0.0
patience_counter = 0
model_save_path = '/p/scratch/training2557/dougnon1/best_model_forensic.pth'

for epoch in range(NUM_EPOCHS):
    # ========================================================================
    # TRAINING PHASE
    # ========================================================================
    model.train()
    running_loss = 0.0
    num_batches = 0
    
    train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")
    optimizer.zero_grad()  # Zero gradients at the start of epoch
    
    for batch_idx, (images, labels) in enumerate(train_pbar):
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.float().to(DEVICE, non_blocking=True)  # Convert to float for BCE
        
        # Mixed Precision forward pass
        with torch.cuda.amp.autocast():
            logits = model(images).squeeze(1)  # [Batch, 1] -> [Batch]
            loss = criterion(logits, labels)
            # Scale loss by accumulation steps
            loss = loss / GRADIENT_ACCUMULATION_STEPS
        
        # Mixed Precision backward pass
        scaler.scale(loss).backward()
        
        # Update weights every GRADIENT_ACCUMULATION_STEPS
        if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        running_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS  # Unscale for logging
        num_batches += 1
        train_pbar.set_postfix({'loss': f'{loss.item() * GRADIENT_ACCUMULATION_STEPS:.4f}'})
    
    # Handle remaining gradients if batch doesn't divide evenly
    if len(train_loader) % GRADIENT_ACCUMULATION_STEPS != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    
    avg_train_loss = running_loss / num_batches
    
    # ========================================================================
    # VALIDATION PHASE
    # ========================================================================
    model.eval()
    all_scores = []
    all_labels = []
    
    val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]")
    with torch.no_grad():
        for images, labels in val_pbar:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            
            logits = model(images).squeeze(1)
            scores = torch.sigmoid(logits).cpu().numpy()  # Convert to probabilities
            labels_np = labels.cpu().numpy()
            
            all_scores.extend(scores)
            all_labels.extend(labels_np)
    
    # Calculate TPR @ 1% FPR
    tpr_at_fpr = get_tpr_at_fpr(all_scores, all_labels, target_fpr=0.01)
    
    # Learning rate scheduling
    scheduler.step(tpr_at_fpr)
    
    # ========================================================================
    # EARLY STOPPING & CHECKPOINTING (FIXED LOGIC)
    # ========================================================================
    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
    print(f"  Train Loss: {avg_train_loss:.4f}")
    print(f"  TPR @ 1% FPR: {tpr_at_fpr:.4f}")
    print(f"  Learning Rate: {scheduler.optimizer.param_groups[0]['lr']:.6f}")
    
    # FIXED: Use >= and reset patience correctly
    if tpr_at_fpr >= best_tpr_at_fpr:
        if tpr_at_fpr > best_tpr_at_fpr:
            best_tpr_at_fpr = tpr_at_fpr
            # Save best model (handle both DataParallel and single GPU)
            if hasattr(model, 'module'):  # DataParallel wraps model in .module
                torch.save(model.module.state_dict(), model_save_path)
            else:
                torch.save(model.state_dict(), model_save_path)
            print(f"  ✓ New best model saved! TPR @ 1% FPR: {best_tpr_at_fpr:.4f}")
        else:
            print(f"  = Same best TPR @ 1% FPR: {best_tpr_at_fpr:.4f} (resetting patience)")
        patience_counter = 0  # Reset patience if we match or beat best
    else:
        patience_counter += 1
        print(f"  No improvement. Patience: {patience_counter}/{PATIENCE}")
    
    if patience_counter >= PATIENCE:
        print(f"\nEarly stopping triggered after {epoch + 1} epochs")
        break
    
    print()

print("=" * 80)
print(f"Training Complete!")
print(f"Best TPR @ 1% FPR: {best_tpr_at_fpr:.4f}")
print("=" * 80)

# ============================================================================
# LOAD BEST MODEL FOR INFERENCE
# ============================================================================
print("\n" + "=" * 80)
print("Loading best model for inference...")
print("=" * 80)

# Load best model (handle both DataParallel and single GPU)
if hasattr(model, 'module'):  # DataParallel wraps model in .module
    model.module.load_state_dict(torch.load(model_save_path))
else:
    model.load_state_dict(torch.load(model_save_path))
model.eval()

# ============================================================================
# INFERENCE ON TEST SET WITH TTA (Test-Time Augmentation)
# ============================================================================
print("\n" + "=" * 80)
print("Running Inference on Test Set with TTA...")
print("=" * 80)
print("TTA: Center Crop + Horizontal Flip + Vertical Flip -> Mean Score")

submission_data = []
submission_path = '/p/scratch/training2557/dougnon1/submission_forensic.csv'

# Create flip transforms for TTA
hflip = transforms.RandomHorizontalFlip(p=1.0)
vflip = transforms.RandomVerticalFlip(p=1.0)

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Inference with TTA"):
        images = batch["image"].to(DEVICE, non_blocking=True)
        image_names = batch["image_name"]
        
        # TTA: Predict on 3 augmentations
        # 1. Center Crop (original)
        logits_orig = model(images).squeeze(1)
        scores_orig = torch.sigmoid(logits_orig)
        
        # 2. Horizontal Flip
        images_hflip = torch.flip(images, dims=[3])  # Flip width dimension
        logits_hflip = model(images_hflip).squeeze(1)
        scores_hflip = torch.sigmoid(logits_hflip)
        
        # 3. Vertical Flip
        images_vflip = torch.flip(images, dims=[2])  # Flip height dimension
        logits_vflip = model(images_vflip).squeeze(1)
        scores_vflip = torch.sigmoid(logits_vflip)
        
        # Average the 3 scores (TTA ensemble)
        scores = (scores_orig + scores_hflip + scores_vflip) / 3.0
        scores = scores.cpu().numpy()
        
        for fname, score in zip(image_names, scores):
            submission_data.append([fname, float(score)])

# ============================================================================
# SAVE SUBMISSION
# ============================================================================
print(f"\nSaving predictions to {submission_path}...")
df = pd.DataFrame(submission_data, columns=["image_name", "score"])
df.to_csv(submission_path, index=False)

print(f"Successfully saved submission file to {submission_path}")
print(f"Total predictions: {len(submission_data)}")
print(f"Score range: [{df['score'].min():.4f}, {df['score'].max():.4f}]")

# ============================================================================
# SUBMIT TO LEADERBOARD
# ============================================================================
if API_KEY:
    print("\n" + "=" * 80)
    print("Submitting to leaderboard server...")
    print("=" * 80)
    
    try:
        with open(submission_path, "rb") as f:
            response = requests.post(
                f"{SERVER_URL}/submit/{TASK_ID}",
                files={"file": f},
                headers={"X-API-Key": API_KEY},
            )
        
        print(f"Response status: {response.status_code}")
        result = response.json()
        print(f"Server response: {result}")
        
        if "score" in result:
            print(f"\n🎉 Score obtained: {result.get('score', 'N/A')}")
    except Exception as e:
        print(f"❌ Error during submission: {e}")
else:
    print("\nNo API key provided. Skipping submission.")

print("\n" + "=" * 80)
print("Done!")
print("=" * 80)
