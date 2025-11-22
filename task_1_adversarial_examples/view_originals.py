
import torch
from torchvision.utils import save_image
import os

# 1. Load Dataset
if not os.path.exists("natural_images.pt"):
    print("Error: natural_images.pt not found in current directory.")
    exit(1)

data = torch.load("natural_images.pt", weights_only=False)
images = data["images"] # Shape: (100, 3, 28, 28)

# 2. Save as a 10x10 grid
os.makedirs("vis_output", exist_ok=True)
save_image(images, "vis_output/original_100_images.png", nrow=10, padding=2)

print("Success! View the image at: task_1_adversarial_examples/vis_output/original_100_images.png")

