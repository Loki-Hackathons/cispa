import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import os
import random
import time
import argparse

# ------------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------------
INPUT_PATH = "natural_images.pt"
OUTPUT_PATH = "submission_pgd.npz"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Default Attack Parameters
DEFAULT_EPSILON = 8.0
DEFAULT_ALPHA = 0.02
DEFAULT_STEPS = 200
DEFAULT_USE_INPUT_DIVERSITY = True
DEFAULT_MOMENTUM = 0.9  # For MI-FGSM

# Available Models
AVAILABLE_MODELS = {
    'resnet18': lambda: models.resnet18(weights=models.ResNet18_Weights.DEFAULT),
    'resnet50': lambda: models.resnet50(weights=models.ResNet50_Weights.DEFAULT),
    'densenet121': lambda: models.densenet121(weights=models.DenseNet121_Weights.DEFAULT),
    'vgg16_bn': lambda: models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT),
    'efficientnet_b0': lambda: models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT),
    'mobilenet_v3_large': lambda: models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT),
}

# Preset Ensembles
PRESET_ENSEMBLES = {
    'small': ['resnet18'],
    'medium': ['resnet50', 'densenet121'],
    'large': ['resnet50', 'densenet121', 'vgg16_bn'],
    'diverse': ['resnet50', 'densenet121', 'efficientnet_b0'],
    'all': ['resnet50', 'densenet121', 'vgg16_bn', 'efficientnet_b0', 'mobilenet_v3_large'],
}

def parse_args():
    parser = argparse.ArgumentParser(description='PGD Attack with Ensemble Surrogate Models')
    
    # Ensemble selection
    parser.add_argument('--ensemble', type=str, default='all',
                        choices=list(PRESET_ENSEMBLES.keys()) + ['custom'],
                        help='Preset ensemble or "custom" for manual specification')
    parser.add_argument('--models', type=str, nargs='+', default=None,
                        help='Custom model list (e.g., --models resnet50 densenet121). Only used if --ensemble custom')
    
    # Attack parameters
    parser.add_argument('--epsilon', type=float, default=DEFAULT_EPSILON,
                        help=f'Max L2 distance (default: {DEFAULT_EPSILON})')
    parser.add_argument('--alpha', type=float, default=DEFAULT_ALPHA,
                        help=f'Step size (default: {DEFAULT_ALPHA})')
    parser.add_argument('--steps', type=int, default=DEFAULT_STEPS,
                        help=f'Number of iterations (default: {DEFAULT_STEPS})')
    parser.add_argument('--no-input-diversity', action='store_true',
                        help='Disable input diversity (random scaling/padding)')
    parser.add_argument('--momentum', type=float, default=DEFAULT_MOMENTUM,
                        help=f'Momentum factor for MI-FGSM (default: {DEFAULT_MOMENTUM}, 0.0 = no momentum)')
    
    # I/O
    parser.add_argument('--input', type=str, default=INPUT_PATH,
                        help=f'Input dataset path (default: {INPUT_PATH})')
    parser.add_argument('--output', type=str, default=OUTPUT_PATH,
                        help=f'Output submission path (default: {OUTPUT_PATH})')
    
    return parser.parse_args()

def get_ensemble_models(model_names):
    """Load ensemble of models."""
    print(f"\n=== Loading {len(model_names)} Models ===")
    models_list = []
    
    for model_name in model_names:
        if model_name not in AVAILABLE_MODELS:
            print(f"  ✗ Warning: Unknown model '{model_name}'. Skipping.")
            print(f"    Available: {list(AVAILABLE_MODELS.keys())}")
            continue
        
        print(f"  Loading {model_name}...", end=' ', flush=True)
        try:
            model = AVAILABLE_MODELS[model_name]()
            model.eval()
            model.to(DEVICE)
            models_list.append(model)
            print("✓")
        except Exception as e:
            print(f"✗ Failed: {e}")
    
    if len(models_list) == 0:
        raise ValueError("No models loaded! Check model names.")
    
    print(f"Total models loaded: {len(models_list)}")
    return models_list

def apply_input_diversity(image_batch, enabled=True):
    """
    Apply random transformations for better transferability.
    
    Enhanced input diversity includes:
    - Wider scaling range (0.8-1.2x instead of 0.9-1.1x)
    - Random translation via padding
    - Random color jitter (brightness/contrast)
    
    Why this helps: If attacks work under random transformations,
    they're more robust and transfer better to different models.
    The black-box model might preprocess images differently (resize, crop, etc.),
    so training attacks to work under various transformations improves transferability.
    """
    if not enabled:
        return image_batch
    
    batch_size = image_batch.shape[0]
    diversified = []
    
    for i in range(batch_size):
        img = image_batch[i:i+1]
        
        # Enhanced scaling: wider range (0.8-1.2x instead of 0.9-1.1x)
        scale = random.uniform(0.8, 1.2)
        new_size = int(224 * scale)
        new_size = max(180, min(260, new_size))  # Clamp to reasonable range
        
        upsampler = nn.Upsample(size=(new_size, new_size), mode='bilinear', align_corners=False)
        scaled = upsampler(img)
        
        # Padding with random translation (creates random shifts)
        pad_h = 224 - new_size
        pad_w = 224 - new_size
        pad_top = random.randint(0, pad_h) if pad_h > 0 else 0
        pad_left = random.randint(0, pad_w) if pad_w > 0 else 0
        
        if pad_h > 0 or pad_w > 0:
            scaled = nn.functional.pad(scaled, (pad_left, pad_w - pad_left, pad_top, pad_h - pad_top), mode='reflect')
        
        # Ensure exactly 224x224
        if scaled.shape[-1] != 224 or scaled.shape[-2] != 224:
            scaled = nn.functional.interpolate(scaled, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Random color jitter (brightness/contrast) - subtle variations
        brightness_factor = random.uniform(0.95, 1.05)
        contrast_factor = random.uniform(0.95, 1.05)
        scaled = scaled * brightness_factor
        mean_val = scaled.mean()
        scaled = (scaled - mean_val) * contrast_factor + mean_val
        scaled = torch.clamp(scaled, 0, 1)
        
        diversified.append(scaled)
    
    return torch.cat(diversified, dim=0)

def pgd_attack_ensemble(models_list, image, label, epsilon, alpha, steps, use_input_diversity, momentum=0.9):
    """
    MI-FGSM (Momentum Iterative Fast Gradient Sign Method) attack using ensemble of models.
    
    MI-FGSM improves transferability by accumulating gradients with momentum,
    making the attack direction more stable and robust.
    """
    adv_image = image.clone().detach().to(DEVICE)
    label = label.clone().detach().to(DEVICE)
    original = image.clone().detach().to(DEVICE)
    
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(DEVICE)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(DEVICE)
    upsampler = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)
    loss_fn = nn.CrossEntropyLoss()
    
    # Initialize momentum accumulator (for MI-FGSM)
    momentum_grad = torch.zeros_like(adv_image)

    # Track the best adversarial example (highest loss) across steps
    # Rationale:
    # - The loss can oscillate over iterations, especially with momentum and projection.
    # - Keeping the image that maximizes the loss against the true label
    #   is often better than just taking the final iterate.
    best_adv = adv_image.clone().detach()
    best_loss = float("-inf")

    for step in range(steps):
        adv_image.requires_grad = True
        
        resized = upsampler(adv_image)
        diversified = apply_input_diversity(resized, enabled=use_input_diversity)
        normalized = (diversified - mean) / std
        
        # Average loss across all models
        total_loss = 0
        for model in models_list:
            outputs = model(normalized)
            loss = loss_fn(outputs, label)
            total_loss += loss
        
        avg_loss = total_loss / len(models_list)

        # Update best adversarial example if current loss is higher
        # (Use the current adv_image before applying the next step.)
        current_loss_val = avg_loss.item()
        if current_loss_val > best_loss:
            best_loss = current_loss_val
            best_adv = adv_image.detach().clone()
        
        # Backward pass
        for model in models_list:
            model.zero_grad()
        avg_loss.backward()
        
        # MI-FGSM: Accumulate gradients with momentum
        grad = adv_image.grad.data
        grad_norm = grad.view(grad.shape[0], -1).norm(p=1, dim=1).view(grad.shape[0], 1, 1, 1)
        grad = grad / (grad_norm + 1e-8)  # Normalize gradient
        
        # Update momentum accumulator
        momentum_grad = momentum * momentum_grad + grad
        
        # Update image using momentum gradient
        adv_image = adv_image.detach() + alpha * momentum_grad.sign()
        
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

    # Return the best adversarial image (highest loss across all steps),
    # not necessarily the final iterate.
    return best_adv

def main():
    args = parse_args()
    
    # Determine model list
    if args.ensemble == 'custom':
        if args.models is None:
            raise ValueError("--models required when --ensemble custom")
        model_names = args.models
    else:
        model_names = PRESET_ENSEMBLES[args.ensemble]
    
    # Print configuration
    print("=" * 60)
    print("PGD Attack Configuration")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Ensemble: {args.ensemble} ({', '.join(model_names)})")
    print(f"Epsilon: {args.epsilon}")
    print(f"Alpha: {args.alpha}")
    print(f"Steps: {args.steps}")
    print(f"Input Diversity: {not args.no_input_diversity}")
    print(f"Momentum (MI-FGSM): {args.momentum}")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print("=" * 60)
    
    start_time = time.time()
    
    # Load dataset
    print("\n=== Loading Dataset ===")
    data = torch.load(args.input, weights_only=False)
    images = data["images"]
    image_ids = data["image_ids"]
    labels = data["labels"]
    print(f"Loaded {len(images)} images")
    print(f"Image shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # Load models
    models_list = get_ensemble_models(model_names)
    
    # Run attack
    print(f"\n=== Starting Attack (Processing {len(images)} images) ===")
    adv_images_list = []
    
    for i in range(len(images)):
        img_start = time.time()
        img = images[i:i+1]
        label = labels[i:i+1]
        
        print(f"\nImage {i+1:3d}/100 (ID: {image_ids[i].item()}, True Label: {label[0].item()}):")
        
        adv_img = pgd_attack_ensemble(
            models_list, img, label, 
            args.epsilon, args.alpha, args.steps,
            use_input_diversity=not args.no_input_diversity,
            momentum=args.momentum
        )
        adv_images_list.append(adv_img)
        
        # Calculate final distance
        diff = (adv_img.cpu() - img).view(-1)
        l2_dist = torch.norm(diff, p=2).item()
        img_time = time.time() - img_start
        
        print(f"  ✓ Completed in {img_time:.2f}s, Final L2: {l2_dist:.4f}")
        
        # Progress update
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / (i + 1)
            remaining = avg_time * (100 - i - 1)
            print(f"\n  Progress: {i+1}/100 ({elapsed/60:.1f} min elapsed, ~{remaining/60:.1f} min remaining)")
    
    # Combine results
    print(f"\n=== Combining Results ===")
    adv_images = torch.cat(adv_images_list, dim=0)
    
    # Statistics
    print("\n=== Final Statistics ===")
    diff = (adv_images.cpu() - images).view(100, -1)
    l2_dists = torch.norm(diff, p=2, dim=1).numpy()
    
    print(f"Average L2 Distance: {l2_dists.mean():.4f}")
    print(f"Min L2 Distance:    {l2_dists.min():.4f}")
    print(f"Max L2 Distance:    {l2_dists.max():.4f}")
    print(f"Std L2 Distance:    {l2_dists.std():.4f}")
    
    # Save
    print(f"\n=== Saving Submission ===")
    final_images_np = adv_images.cpu().numpy().astype(np.float32)
    final_ids_np = image_ids.cpu().numpy()
    
    np.savez_compressed(args.output, images=final_images_np, image_ids=final_ids_np)
    
    total_time = time.time() - start_time
    print(f"✓ Saved to {args.output}")
    print(f"\n=== Total Time: {total_time/60:.2f} minutes ===")

if __name__ == "__main__":
    main()
