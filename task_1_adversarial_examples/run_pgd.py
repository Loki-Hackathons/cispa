
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import os

# ------------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------------
INPUT_PATH = "natural_images.pt"
OUTPUT_PATH = "submission_pgd.npz"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {DEVICE}")

# Attack Parameters
EPSILON = 10.0          # Max L2 distance allowed (Lower = Better Score, Harder to Attack)
ALPHA = 0.05           # Step size (smaller alpha for smaller epsilon)
STEPS = 50             # Iterations

# ------------------------------------------------------------------------------
# 1. LOAD SURROGATE MODEL
# ------------------------------------------------------------------------------
def get_surrogate_model():
    # We use ResNet18. It's fast and standard.
    # Weights=Default loads the best available ImageNet weights.
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.eval()
    model.to(DEVICE)
    return model

# ------------------------------------------------------------------------------
# 2. PGD ATTACK LOGIC
# ------------------------------------------------------------------------------
def pgd_attack(model, images, labels):
    # Clone to avoid modifying original
    adv_images = images.clone().detach().to(DEVICE)
    labels = labels.clone().detach().to(DEVICE)
    originals = images.clone().detach().to(DEVICE)
    
    # Normalization for ResNet (ImageNet standards)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(DEVICE)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(DEVICE)
    
    # Upsampler (28x28 -> 224x224)
    upsampler = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)

    loss_fn = nn.CrossEntropyLoss()

    print(f"Starting PGD Attack (Epsilon={EPSILON}, Steps={STEPS})...")

    for step in range(STEPS):
        adv_images.requires_grad = True
        
        # Resize & Normalize
        resized = upsampler(adv_images)
        normalized = (resized - mean) / std
        
        # Forward Pass
        outputs = model(normalized)
        
        # Calculate Loss (We want to MAXIMIZE this to fool the model)
        loss = loss_fn(outputs, labels)
        
        # Backward Pass
        model.zero_grad()
        loss.backward()
        
        # Gradient Update
        grad = adv_images.grad.data
        
        # Step in direction of gradient (Attack!)
        adv_images = adv_images.detach() + ALPHA * grad.sign()
        
        # PROJECTION (L2 Constraint)
        # We want: ||new - original||_2 <= EPSILON
        delta = adv_images - originals
        
        # Flatten to calculate norm per image
        delta_flat = delta.view(delta.shape[0], -1)
        norm = delta_flat.norm(p=2, dim=1).view(delta.shape[0], 1, 1, 1)
        
        # If norm > epsilon, scale it down. If norm <= epsilon, keep it (factor=1).
        factor = torch.min(torch.ones_like(norm), torch.tensor(EPSILON) / (norm + 1e-6))
        delta = delta * factor
        
        # Apply constrained delta
        adv_images = originals + delta
        
        # Clip to valid image range [0, 1]
        adv_images = torch.clamp(adv_images, 0, 1).detach()
        
        if step % 10 == 0:
            print(f"  Step {step}: Loss = {loss.item():.4f}")

    return adv_images

# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # Load Data
    data = torch.load(INPUT_PATH, weights_only=False)
    images = data["images"]      # (100, 3, 28, 28)
    image_ids = data["image_ids"]
    labels = data["labels"]      # True labels
    
    print(f"Loaded {len(images)} images.")
    
    # Load Model
    model = get_surrogate_model()
    
    # Run Attack
    adv_images = pgd_attack(model, images, labels)
    
    # Calculate final distance
    diff = (adv_images.cpu() - images).view(100, -1)
    l2_dist = torch.norm(diff, p=2, dim=1).mean().item()
    print(f"Final Average L2 Distance: {l2_dist:.4f}")
    
    # Save
    final_images_np = adv_images.cpu().numpy().astype(np.float32)
    final_ids_np = image_ids.cpu().numpy()
    
    np.savez_compressed(OUTPUT_PATH, images=final_images_np, image_ids=final_ids_np)
    print(f"Saved submission to {OUTPUT_PATH}")
