
import torch
import numpy as np
from torchvision.utils import save_image
import os

# 1. Load Data
data = torch.load("natural_images.pt", weights_only=False)
images = data["images"] # Shape: (100, 3, 28, 28)
image_ids = data["image_ids"] # Shape: (100,)

# 2. Basic Attack: Add Random Gaussian Noise
# We want small noise to keep L2 distance low
noise_level = 0.1 
noise = torch.randn_like(images) * noise_level
adv_images = images + noise

# 3. Clip to ensure valid image range [0,1]
adv_images = torch.clamp(adv_images, 0, 1)

# 4. Save Visualization (First 5 images)
os.makedirs("vis_output", exist_ok=True)
comparison = torch.cat([images[:5], adv_images[:5]])
save_image(comparison, "vis_output/simple_attack_sample.png", nrow=5)
print("View images at: task_1_adversarial_examples/vis_output/simple_attack_sample.png")

# 5. Save Submission File
# Convert to numpy float32
final_images = adv_images.cpu().numpy().astype(np.float32)
# Convert IDs to numpy (usually int64)
final_ids = image_ids.cpu().numpy()

# Save BOTH images and image_ids
np.savez_compressed("simple_attack.npz", images=final_images, image_ids=final_ids)
print("Saved submission to: simple_attack.npz")
