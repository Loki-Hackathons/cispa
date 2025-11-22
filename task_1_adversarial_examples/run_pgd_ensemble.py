import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import os
import random
import time

# ------------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------------
INPUT_PATH = "natural_images.pt"
OUTPUT_PATH = "submission_pgd.npz"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Attack Parameters
EPSILON = 2.0          # Fixed epsilon for all images
ALPHA = 0.02           # Step size
STEPS = 100            # Iterations per image
USE_INPUT_DIVERSITY = True
ENSEMBLE_MODELS = ['resnet50', 'densenet121', 'vgg16_bn']

print(f"=== Configuration ===", flush=True)
print(f"Device: {DEVICE}", flush=True)
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
print(f"Epsilon: {EPSILON}", flush=True)
print(f"Alpha: {ALPHA}", flush=True)
print(f"Steps: {STEPS}", flush=True)
print(f"Input Diversity: {USE_INPUT_DIVERSITY}", flush=True)
print(f"Ensemble Models: {ENSEMBLE_MODELS}", flush=True)
print(f"====================", flush=True)

# ------------------------------------------------------------------------------
# 1. LOAD ENSEMBLE SURROGATE MODELS
# ------------------------------------------------------------------------------
def get_ensemble_models():
    print("\n=== Loading Models ===")
    models_list = []
    
    if 'resnet50' in ENSEMBLE_MODELS:
        print("Loading ResNet50...")
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        m.eval()
        m.to(DEVICE)
        models_list.append(m)
        print("  ✓ ResNet50 loaded")
    
    if 'densenet121' in ENSEMBLE_MODELS:
        print("Loading DenseNet121...")
        m = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        m.eval()
        m.to(DEVICE)
        models_list.append(m)
        print("  ✓ DenseNet121 loaded")
    
    if 'vgg16_bn' in ENSEMBLE_MODELS:
        print("Loading VGG16_BN...")
        m = models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT)
        m.eval()
        m.to(DEVICE)
        models_list.append(m)
        print("  ✓ VGG16_BN loaded")
    
    print(f"Total models loaded: {len(models_list)}")
    return models_list

# ------------------------------------------------------------------------------
# 2. INPUT DIVERSITY
# ------------------------------------------------------------------------------
def apply_input_diversity(image_batch):
    if not USE_INPUT_DIVERSITY:
        return image_batch
    
    batch_size = image_batch.shape[0]
    diversified = []
    
    for i in range(batch_size):
        img = image_batch[i:i+1]
        scale = random.uniform(0.9, 1.1)
        new_size = int(224 * scale)
        
        upsampler = nn.Upsample(size=(new_size, new_size), mode='bilinear', align_corners=False)
        scaled = upsampler(img)
        
        pad_h = 224 - new_size
        pad_w = 224 - new_size
        pad_top = random.randint(0, pad_h) if pad_h > 0 else 0
        pad_left = random.randint(0, pad_w) if pad_w > 0 else 0
        
        if pad_h > 0 or pad_w > 0:
            scaled = nn.functional.pad(scaled, (pad_left, pad_w - pad_left, pad_top, pad_h - pad_top), mode='reflect')
        
        if scaled.shape[-1] != 224:
            scaled = nn.functional.interpolate(scaled, size=(224, 224), mode='bilinear', align_corners=False)
        
        diversified.append(scaled)
    
    return torch.cat(diversified, dim=0)

# ------------------------------------------------------------------------------
# 3. PGD ATTACK WITH ENSEMBLE (Single Image)
# ------------------------------------------------------------------------------
def pgd_attack_ensemble(models_list, image, label, epsilon):
    adv_image = image.clone().detach().to(DEVICE)
    label = label.clone().detach().to(DEVICE)
    original = image.clone().detach().to(DEVICE)
    
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(DEVICE)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(DEVICE)
    upsampler = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)
    loss_fn = nn.CrossEntropyLoss()

    for step in range(STEPS):
        adv_image.requires_grad = True
        
        resized = upsampler(adv_image)
        diversified = apply_input_diversity(resized)
        normalized = (diversified - mean) / std
        
        # Average loss across all models
        total_loss = 0
        for model in models_list:
            outputs = model(normalized)
            loss = loss_fn(outputs, label)
            total_loss += loss
        
        avg_loss = total_loss / len(models_list)
        
        # Backward pass
        for model in models_list:
            model.zero_grad()
        avg_loss.backward()
        
        grad = adv_image.grad.data
        adv_image = adv_image.detach() + ALPHA * grad.sign()
        
        # L2 Projection
        delta = adv_image - original
        delta_flat = delta.view(-1)
        norm = delta_flat.norm(p=2)
        if norm > epsilon:
            delta = delta * (epsilon / (norm + 1e-6))
        
        adv_image = original + delta
        adv_image = torch.clamp(adv_image, 0, 1).detach()
        
        if step % 25 == 0:
            current_dist = torch.norm((adv_image - original).view(-1), p=2).item()
            print(f"    Step {step:3d}: Loss={avg_loss.item():7.3f}, L2={current_dist:.4f}")

    return adv_image

# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    start_time = time.time()
    
    print("\n=== Loading Dataset ===")
    data = torch.load(INPUT_PATH, weights_only=False)
    images = data["images"]
    image_ids = data["image_ids"]
    labels = data["labels"]
    print(f"Loaded {len(images)} images")
    print(f"Image shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    
    print("\n=== Loading Models ===")
    models_list = get_ensemble_models()
    
    print(f"\n=== Starting Attack (Processing {len(images)} images) ===")
    adv_images_list = []
    
    for i in range(len(images)):
        img_start = time.time()
        img = images[i:i+1]
        label = labels[i:i+1]
        
        print(f"\nImage {i+1:3d}/100 (ID: {image_ids[i].item()}, True Label: {label[0].item()}):")
        
        adv_img = pgd_attack_ensemble(models_list, img, label, EPSILON)
        adv_images_list.append(adv_img)
        
        # Calculate final distance for this image
        diff = (adv_img.cpu() - img).view(-1)
        l2_dist = torch.norm(diff, p=2).item()
        img_time = time.time() - img_start
        
        print(f"  ✓ Completed in {img_time:.2f}s, Final L2: {l2_dist:.4f}")
        
        # Progress update every 10 images
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / (i + 1)
            remaining = avg_time * (100 - i - 1)
            print(f"\n  Progress: {i+1}/100 ({elapsed/60:.1f} min elapsed, ~{remaining/60:.1f} min remaining)")
    
    print(f"\n=== Combining Results ===")
    adv_images = torch.cat(adv_images_list, dim=0)
    
    print("\n=== Calculating Final Statistics ===")
    diff = (adv_images.cpu() - images).view(100, -1)
    l2_dists = torch.norm(diff, p=2, dim=1).numpy()
    
    print(f"Average L2 Distance: {l2_dists.mean():.4f}")
    print(f"Min L2 Distance:    {l2_dists.min():.4f}")
    print(f"Max L2 Distance:    {l2_dists.max():.4f}")
    print(f"Std L2 Distance:    {l2_dists.std():.4f}")
    
    print(f"\n=== Saving Submission ===")
    final_images_np = adv_images.cpu().numpy().astype(np.float32)
    final_ids_np = image_ids.cpu().numpy()
    
    np.savez_compressed(OUTPUT_PATH, images=final_images_np, image_ids=final_ids_np)
    
    total_time = time.time() - start_time
    print(f"✓ Saved to {OUTPUT_PATH}")
    print(f"\n=== Total Time: {total_time/60:.2f} minutes ===")
