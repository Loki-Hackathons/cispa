"""
================================================================================
TWO-STAGE CASCADE ARCHITECTURE FOR IMAGE ATTRIBUTION
================================================================================

This script implements a highly efficient two-stage cascade architecture:

STAGE A (VAR Filter):
  - Uses VAR VAE reconstruction loss to identify VAR images with high precision
  - Calibrates tau_recon threshold on validation set to separate VAR from OOD
  - If reconstruction_loss < tau_recon → Predict "VAR" (high confidence)

STAGE B (ConvNeXt Classifier + Outlier Rejection):
  - Uses ConvNeXt Base for remaining classes (RAR, SD, Taming)
  - Applies confidence threshold (tau_conf) for outlier detection
  - If max_softmax_prob < tau_conf → Predict "outlier"

TRAINING:
  - ConvNeXt trains normally on all 4 known classes (VAR, RAR, SD, Taming)
  - VAE is NOT used during training (only for validation/inference)

VALIDATION/INFERENCE:
  - First applies Stage A (VAE filter) to catch VAR images
  - Then applies Stage B (ConvNeXt) for remaining images
  - Optimizes both tau_recon and tau_conf thresholds

================================================================================
IMPORTANT: DATA GENERATION STEP (Run BEFORE this script!)
================================================================================
Before running this training script, you MUST generate additional training data:

1. Run: python VAR.py
   - This will generate 2000 extra VAR images in outputs/ folder
   - Also downloads VAR VAE checkpoints needed for Stage A

2. Run: python RAR.py  
   - This will generate 2000 extra RAR images in outputs/ folder

3. Copy the generated images to the appropriate dataset/train/ folders:
   - Copy VAR outputs to dataset/train/VAR/
   - Copy RAR outputs to dataset/train/RAR/

This data augmentation is CRITICAL because we only have 250 images per class initially.
================================================================================
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import torch.nn.functional as F
from PIL import Image
import numpy as np
from tqdm import tqdm
from pathlib import Path

# Add VAR directory to path for importing VAR models
VAR_DIR = Path("./VAR")
if VAR_DIR.exists():
    sys.path.insert(0, str(VAR_DIR))

# ============================================================================
# CONFIGURATION
# ============================================================================
BATCH_SIZE = 64  # High batch size optimized for A100s
EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
DATASET_DIR = Path("./dataset")  # Path to unzipped dataset folder

# Device setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
if torch.cuda.is_available():
    print(f"Number of GPUs: {torch.cuda.device_count()}")

# Label mappings (4 known classes only - outlier is determined by threshold)
IDX_TO_LABEL = {0: 'RAR', 1: 'SD', 2: 'Taming', 3: 'VAR'}
LABEL_TO_IDX = {'RAR': 0, 'SD': 1, 'Taming': 2, 'VAR': 3}
KNOWN_CLASSES = ['RAR', 'SD', 'Taming', 'VAR']

# VAE paths (assumes VAR.py has been run to download checkpoints)
VAR_VAE_PATH = Path("./VAR/checkpoints/vae/vae_ch160v4096z32.pth")
VAR_CHECKPOINT_PATH = Path("./VAR/checkpoints/var/var_d16.pth")

# ============================================================================
# CUSTOM DATASET CLASS
# ============================================================================
class AttributionDataset(Dataset):
    """
    Custom dataset loader for Image Attribution task.
    
    - TRAIN: Only loads 4 known classes (RAR, SD, Taming, VAR) - NO outliers
    - VAL: Loads 4 known classes + OOD for threshold optimization
    - TEST: Loads unlabeled images
    """
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.data = []
        
        if split == 'test':
            # Test set: unlabeled images
            test_path = self.root_dir / 'test'
            if test_path.exists():
                self.images = sorted([f for f in os.listdir(test_path) if f.endswith('.png')])
                self.data = [(test_path / img, None) for img in self.images]
        else:
            # Train/Val sets: labeled images
            split_path = self.root_dir / split
            
            if split == 'train':
                # TRAIN: Only 4 known classes (exclude OOD)
                classes = KNOWN_CLASSES
            else:  # split == 'val'
                # VAL: All classes including OOD for threshold optimization
                classes = KNOWN_CLASSES + ['OOD']
            
            for cls_name in classes:
                cls_folder = split_path / cls_name
                if cls_folder.exists():
                    for img_file in sorted(os.listdir(cls_folder)):
                        if img_file.endswith('.png'):
                            img_path = cls_folder / img_file
                            # Map OOD to -1, known classes to their index
                            if cls_name == 'OOD':
                                label = -1
                            else:
                                label = LABEL_TO_IDX[cls_name]
                            self.data.append((img_path, label))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        if self.split == 'test':
            return image, os.path.basename(img_path)
        return image, label

# ============================================================================
# VAR VAE SETUP - Two-Stage Cascade Stage A
# ============================================================================
def build_vae_var():
    """
    Build and load the VAR VAE model for reconstruction loss computation.
    This is Stage A of the two-stage cascade: VAR detection via reconstruction loss.
    
    Returns:
        vae_model: The VAE encoder-decoder model
    """
    try:
        from models import build_vae_var as build_var_model
    except ImportError:
        print("WARNING: Could not import VAR models. Make sure VAR.py has been run first.")
        print("Falling back to placeholder VAE (will not filter VAR images).")
        return None
    
    if not VAR_VAE_PATH.exists():
        print(f"WARNING: VAE checkpoint not found at {VAR_VAE_PATH}")
        print("Make sure VAR.py has been executed to download checkpoints.")
        return None
    
    print("Loading VAR VAE model...")
    device = DEVICE
    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    
    # Build VAE model (we only need the VAE, not the full VAR)
    vae, _ = build_var_model(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,
        device=device, patch_nums=patch_nums,
        num_classes=1000, depth=16, shared_aln=False
    )
    
    # Load VAE weights
    vae.load_state_dict(torch.load(VAR_VAE_PATH, map_location=device))
    vae.eval()
    
    # Freeze VAE parameters
    for p in vae.parameters():
        p.requires_grad_(False)
    
    print("VAR VAE loaded successfully!")
    return vae


def get_vae_loss(image, vae_model):
    """
    Compute VAR VAE reconstruction loss for an image.
    Lower loss indicates the image is more likely to be from VAR.
    
    Args:
        image: Tensor [C, H, W] or [B, C, H, W], normalized to [0, 1] or ImageNet normalized
        vae_model: The loaded VAE model
    
    Returns:
        reconstruction_loss: MSE loss between original and reconstructed image
    """
    if vae_model is None:
        # Return high loss if VAE not available (will not filter VAR)
        return torch.tensor(1.0, device=DEVICE)
    
    vae_model.eval()
    
    # Ensure image is in correct format
    if len(image.shape) == 3:
        image = image.unsqueeze(0)  # Add batch dimension
    
    # VAR VAE expects images in [0, 1] range, but our transforms normalize to ImageNet
    # We need to denormalize first, then normalize to [0, 1]
    # ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    mean = torch.tensor([0.485, 0.456, 0.406], device=image.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=image.device).view(1, 3, 1, 1)
    
    # Denormalize to [0, 1] range
    image_denorm = image * std + mean
    image_denorm = torch.clamp(image_denorm, 0, 1)
    
    # Resize to 256x256 if needed (VAR VAE expects 256x256)
    if image_denorm.shape[-1] != 256:
        image_denorm = F.interpolate(image_denorm, size=(256, 256), mode='bilinear', align_corners=False)
    
    with torch.no_grad():
        try:
            # Encode and decode
            # VAR VAE forward pass: encode -> quantize -> decode
            z, _, _ = vae_model.encode(image_denorm)
            recon = vae_model.decode(z)
            
            # Compute MSE loss
            loss = F.mse_loss(recon, image_denorm, reduction='mean')
            
            # Return per-image loss (average over batch if batch > 1)
            if len(loss.shape) > 0:
                loss = loss.mean()
            
            return loss
        except Exception as e:
            print(f"Warning: VAE forward pass failed: {e}")
            # Return high loss on error
            return torch.tensor(1.0, device=image.device)


def calibrate_tau_recon(val_dataset, vae_model, val_transform):
    """
    Calibrate the reconstruction threshold (tau_recon) on validation set.
    Find the threshold that best separates VAR images from OOD images.
    
    Args:
        val_dataset: Validation dataset (with data attribute containing (path, label) tuples)
        vae_model: Loaded VAE model
        val_transform: Transform for validation images
    
    Returns:
        tau_recon: Optimal reconstruction threshold
    """
    print("\n" + "=" * 80)
    print("Calibrating tau_recon (VAR Reconstruction Threshold)...")
    print("=" * 80)
    
    if vae_model is None:
        print("VAE model not available. Using default tau_recon = 0.5")
        return 0.5
    
    # Collect VAR and OOD samples from validation set
    var_losses = []
    ood_losses = []
    
    print("Computing VAE reconstruction losses on validation set...")
    vae_model.eval()
    
    with torch.no_grad():
        for idx in tqdm(range(len(val_dataset)), desc="Calibrating"):
            img_path, label = val_dataset.data[idx]
            
            # Load and transform image
            image = Image.open(img_path).convert("RGB")
            if val_transform:
                image = val_transform(image)
            image = image.to(DEVICE)
            
            # Compute VAE loss
            loss = get_vae_loss(image, vae_model)
            loss_val = loss.item()
            
            # Categorize by label
            if label == LABEL_TO_IDX['VAR']:
                var_losses.append(loss_val)
            elif label == -1:  # OOD
                ood_losses.append(loss_val)
    
    if len(var_losses) == 0 or len(ood_losses) == 0:
        print("WARNING: Not enough VAR or OOD samples for calibration.")
        print("Using default tau_recon = 0.5")
        return 0.5
    
    var_losses = np.array(var_losses)
    ood_losses = np.array(ood_losses)
    
    print(f"\nVAR reconstruction losses: mean={var_losses.mean():.4f}, std={var_losses.std():.4f}")
    print(f"OOD reconstruction losses: mean={ood_losses.mean():.4f}, std={ood_losses.std():.4f}")
    
    # Find optimal threshold by maximizing VAR TPR while minimizing OOD FPR
    # We want: VAR images should have loss < threshold (high TPR)
    #          OOD images should have loss >= threshold (low FPR)
    
    all_losses = np.concatenate([var_losses, ood_losses])
    threshold_candidates = np.linspace(all_losses.min(), all_losses.max(), 100)
    
    best_thresh = 0.5
    best_score = 0.0
    
    for thresh in threshold_candidates:
        # VAR TPR: fraction of VAR images with loss < threshold
        var_tpr = (var_losses < thresh).mean()
        
        # OOD FPR: fraction of OOD images with loss < threshold (wrongly classified as VAR)
        ood_fpr = (ood_losses < thresh).mean()
        
        # Score: maximize TPR while minimizing FPR
        # We want high VAR TPR and low OOD FPR
        score = var_tpr / (1 + 5 * ood_fpr)  # Similar metric to competition
        
        if score > best_score:
            best_score = score
            best_thresh = thresh
    
    print(f"\nOptimal tau_recon: {best_thresh:.4f}")
    print(f"VAR TPR at this threshold: {(var_losses < best_thresh).mean():.4f}")
    print(f"OOD FPR at this threshold: {(ood_losses < best_thresh).mean():.4f}")
    print("=" * 80)
    
    return best_thresh

# ============================================================================
# MODEL SETUP - ConvNeXt Base
# ============================================================================
def get_model():
    """
    Create ConvNeXt Base model with 4-class output.
    Outlier detection is handled via confidence threshold, not a 5th class.
    """
    model = models.convnext_base(weights=models.ConvNeXt_Base_Weights.DEFAULT)
    
    # Replace final classifier for 4 classes (RAR, SD, Taming, VAR)
    # The model outputs 4 logits - outlier is determined by low confidence
    num_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(num_features, 4)
    
    return model

# ============================================================================
# HACKATHON METRIC: TPR / (1 + 5 * FPR)
# ============================================================================
def compute_hackathon_metric(preds, targets, is_ood, threshold=0.8, var_predictions=None):
    """
    Compute the competition metric: TPR / (1 + 5 * FPR)
    
    Args:
        preds: Softmax probabilities [N, 4] for 4 known classes (RAR, SD, Taming, VAR)
        targets: True labels [N] where -1 indicates OOD
        is_ood: Boolean array [N] indicating which samples are actually OOD
        threshold: Confidence threshold for outlier detection (tau_conf)
        var_predictions: Optional [N] boolean array indicating VAR predictions from Stage A
    
    Returns:
        score: Competition metric value
    """
    # Get max probability and predicted class for each sample
    max_probs, predicted_classes = torch.max(preds, dim=1)
    
    # Apply threshold: if confidence < threshold, predict "outlier"
    pred_is_outlier = max_probs < threshold
    
    # If VAR predictions from Stage A are provided, incorporate them
    # VAR images predicted by Stage A should be counted as VAR predictions
    if var_predictions is not None:
        # Override predictions for VAR-detected images
        var_idx = LABEL_TO_IDX['VAR']
        predicted_classes = predicted_classes.clone()
        predicted_classes[var_predictions] = var_idx
        pred_is_outlier[var_predictions] = False  # VAR predictions are not outliers
    
    # Calculate TPR and FPR for each of the 4 known classes
    tprs = []
    fprs = []
    
    for class_idx in range(4):  # For each known class: RAR, SD, Taming, VAR
        # True labels for this class (exclude OOD)
        is_class = (targets == class_idx)
        is_not_class = (targets != class_idx) & (targets != -1)  # Other known classes
        is_ood_sample = (targets == -1)
        
        # Predictions for this class
        pred_class = (predicted_classes == class_idx) & (~pred_is_outlier)
        pred_not_class = (predicted_classes != class_idx) | pred_is_outlier
        
        # True Positives: Actually class_idx AND predicted as class_idx (not outlier)
        tp = (is_class & pred_class).sum().item()
        
        # False Negatives: Actually class_idx BUT predicted as something else or outlier
        fn = (is_class & pred_not_class).sum().item()
        
        # False Positives: NOT class_idx BUT predicted as class_idx
        # This includes:
        #   - Other known classes misclassified as class_idx
        #   - OOD samples misclassified as class_idx
        fp = ((is_not_class | is_ood_sample) & pred_class).sum().item()
        
        # True Negatives: NOT class_idx AND correctly predicted as not class_idx
        tn = ((is_not_class | is_ood_sample) & pred_not_class).sum().item()
        
        # Calculate TPR and FPR
        tpr = tp / (tp + fn + 1e-6)  # Avoid division by zero
        fpr = fp / (fp + tn + 1e-6)
        
        tprs.append(tpr)
        fprs.append(fpr)
    
    # Macro-average across the 4 classes
    avg_tpr = np.mean(tprs)
    avg_fpr = np.mean(fprs)
    
    # Competition formula: TPR / (1 + 5 * FPR)
    score = avg_tpr / (1 + (5 * avg_fpr))
    
    return score

# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================
def main():
    print("=" * 80)
    print("Image Attribution Training - Two-Stage Cascade Architecture")
    print("Stage A: VAR VAE Reconstruction Loss Filter")
    print("Stage B: ConvNeXt Base Classifier + Outlier Rejection")
    print("Running on 4x A100 GPUs")
    print("=" * 80)
    
    # ========================================================================
    # DATA TRANSFORMS
    # ========================================================================
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # ========================================================================
    # LOAD VAR VAE MODEL (Stage A)
    # ========================================================================
    print("\n" + "=" * 80)
    print("Stage A: Loading VAR VAE for Reconstruction Loss Filtering")
    print("=" * 80)
    vae_model = build_vae_var()
    
    # ========================================================================
    # LOAD DATASETS
    # ========================================================================
    print("\nLoading datasets...")
    train_dataset = AttributionDataset(DATASET_DIR, split='train', transform=train_transform)
    val_dataset = AttributionDataset(DATASET_DIR, split='val', transform=val_transform)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # ========================================================================
    # CALIBRATE TAU_RECON (Reconstruction Threshold)
    # ========================================================================
    # Create transform for VAE (needs 256x256, normalized to [0,1] then ImageNet normalized)
    vae_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Use the same validation dataset but with VAE transform for calibration
    tau_recon = calibrate_tau_recon(val_dataset, vae_model, vae_transform)
    
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
    
    # ========================================================================
    # INITIALIZE MODEL WITH DATAPARALLEL
    # ========================================================================
    print("\nInitializing model...")
    model = get_model().to(DEVICE)
    
    # Wrap in DataParallel for multi-GPU training
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel!")
        model = nn.DataParallel(model)
    else:
        print("Using single GPU")
    
    # ========================================================================
    # OPTIMIZER & SCHEDULER
    # ========================================================================
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=LEARNING_RATE, 
        weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # ========================================================================
    # TRAINING LOOP
    # ========================================================================
    best_score = 0.0
    best_threshold = 0.8
    
    print("\n" + "=" * 80)
    print("Starting Training...")
    print("=" * 80)
    
    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for images, labels in train_pbar:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / num_batches
        scheduler.step()
        
        # ====================================================================
        # VALIDATION & THRESHOLD OPTIMIZATION (Two-Stage Cascade)
        # ====================================================================
        model.eval()
        all_probs = []
        all_labels = []
        var_predictions_stage_a = []  # Track VAR predictions from Stage A
        
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]")
        with torch.no_grad():
            for images, labels in val_pbar:
                images = images.to(DEVICE, non_blocking=True)
                
                # Stage A: VAR Filter using VAE reconstruction loss
                batch_var_preds = []
                if vae_model is not None:
                    for img in images:
                        vae_loss = get_vae_loss(img, vae_model)
                        # If reconstruction loss < tau_recon, predict VAR
                        batch_var_preds.append(vae_loss.item() < tau_recon)
                else:
                    batch_var_preds = [False] * len(images)
                
                var_predictions_stage_a.extend(batch_var_preds)
                
                # Stage B: ConvNeXt classification (for non-VAR images)
                # Note: We still run ConvNeXt on all images for metric calculation,
                # but Stage A VAR predictions will override ConvNeXt predictions
                outputs = model(images)
                probs = F.softmax(outputs, dim=1)
                
                all_probs.append(probs.cpu())
                all_labels.append(labels.cpu())
        
        all_probs = torch.cat(all_probs, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        var_predictions_stage_a = torch.tensor(var_predictions_stage_a, dtype=torch.bool)
        
        # Identify OOD samples (label == -1)
        is_ood = (all_labels == -1)
        
        # Scan thresholds from 0.50 to 0.99 (step 0.05) for tau_conf
        current_best_score = 0.0
        current_best_thresh = 0.5
        
        threshold_range = np.arange(0.50, 1.00, 0.05)
        for thresh in threshold_range:
            score = compute_hackathon_metric(
                all_probs, all_labels, is_ood, 
                threshold=thresh,
                var_predictions=var_predictions_stage_a
            )
            if score > current_best_score:
                current_best_score = score
                current_best_thresh = thresh
        
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print(f"  Train Loss: {avg_loss:.4f}")
        print(f"  Best Val Score: {current_best_score:.4f} at Threshold: {current_best_thresh:.2f}")
        print(f"  Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save best model
        if current_best_score > best_score:
            best_score = current_best_score
            best_threshold = current_best_thresh
            # Handle DataParallel state dict
            if isinstance(model, nn.DataParallel):
                torch.save(model.module.state_dict(), "best_model.pth")
            else:
                torch.save(model.state_dict(), "best_model.pth")
            print(f"  ✓ New best model saved! (Score: {best_score:.4f})")
        print()
    
    print("=" * 80)
    print(f"Training Complete!")
    print(f"Best Validation Score: {best_score:.4f}")
    print(f"Best tau_conf (confidence threshold): {best_threshold:.2f}")
    print(f"tau_recon (reconstruction threshold): {tau_recon:.4f}")
    print("=" * 80)
    
    # ========================================================================
    # INFERENCE ON TEST SET
    # ========================================================================
    print("\n" + "=" * 80)
    print("Running Inference on Test Set...")
    print("=" * 80)
    
    test_dataset = AttributionDataset(DATASET_DIR, split='test', transform=val_transform)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Test samples: {len(test_dataset)}")
    
    # Load best model
    print("Loading best model...")
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(torch.load("best_model.pth"))
    else:
        model.load_state_dict(torch.load("best_model.pth"))
    model.eval()
    
    submission_data = []
    
    print(f"Using tau_recon: {tau_recon:.4f} for VAR detection (Stage A)")
    print(f"Using tau_conf: {best_threshold:.2f} for outlier detection (Stage B)")
    
    var_count = 0
    convnext_count = 0
    
    with torch.no_grad():
        for images, filenames in tqdm(test_loader, desc="Inference"):
            images = images.to(DEVICE, non_blocking=True)
            batch_size = len(filenames)
            
            # ============================================================
            # STAGE A: VAR Filter (Reconstruction Loss) - Batch Processing
            # ============================================================
            var_mask = torch.zeros(batch_size, dtype=torch.bool, device=DEVICE)
            
            if vae_model is not None:
                for i in range(batch_size):
                    vae_loss = get_vae_loss(images[i], vae_model)
                    if vae_loss.item() < tau_recon:
                        var_mask[i] = True
                        var_count += 1
                        submission_data.append([filenames[i], "VAR"])
            
            # ============================================================
            # STAGE B: ConvNeXt Classification + Outlier Rejection
            # ============================================================
            # Only process non-VAR images through ConvNeXt
            non_var_indices = ~var_mask
            if non_var_indices.any():
                non_var_images = images[non_var_indices]
                non_var_filenames = [filenames[i] for i in range(batch_size) if non_var_indices[i]]
                
                outputs = model(non_var_images)
                probs = F.softmax(outputs, dim=1)
                
                max_probs, preds = torch.max(probs, dim=1)
                
                for i, filename in enumerate(non_var_filenames):
                    max_prob = max_probs[i].item()
                    pred = preds[i].item()
                    
                    # Apply confidence threshold: if confidence < tau_conf, predict "outlier"
                    if max_prob < best_threshold:
                        label_str = "outlier"
                    else:
                        label_str = IDX_TO_LABEL[pred]
                    
                    convnext_count += 1
                    submission_data.append([filename, label_str])
    
    print(f"\nStage A (VAR filter): {var_count} images classified as VAR")
    print(f"Stage B (ConvNeXt): {convnext_count} images classified by ConvNeXt")
    
    # Save submission file
    df = pd.DataFrame(submission_data, columns=["image_name", "label"])
    df.to_csv("submission.csv", index=False)
    
    print("\n" + "=" * 80)
    print("Submission file saved: submission.csv")
    print(f"Total predictions: {len(submission_data)}")
    print("=" * 80)
    
    # Print label distribution
    label_counts = df['label'].value_counts()
    print("\nPrediction distribution:")
    for label, count in label_counts.items():
        print(f"  {label}: {count}")

if __name__ == "__main__":
    main()

