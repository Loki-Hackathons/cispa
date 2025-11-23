import torch
import numpy as np
import os
import json
import time
import argparse
from version2.models import EnsembleModel
from version2.attack import BatchedBSPGD

# Configuration
# Assumes running from task_1_adversarial_examples/ via 'python -m version2.main_solver'
DATA_PATH = "natural_images.pt" 
STATE_FILE = "version2/local_state.json"
OUTPUT_FILE = "version2/submission.npz"

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_state(state):
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=20, help='Number of restarts (parallel batch size)')
    parser.add_argument('--steps', type=int, default=100, help='PGD steps')
    parser.add_argument('--epsilon', type=float, default=8.0, help='Global Max Epsilon (L2)')
    parser.add_argument('--kappa-default', type=float, default=0.0)
    args = parser.parse_args()

    print("=== Phase 1: Local Solver (Doctorant Grade) ===")
    
    # 1. Load Data
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Could not find {DATA_PATH}. Make sure to run from task_1_adversarial_examples/")
        
    data = torch.load(DATA_PATH, weights_only=False)
    images = data["images"]
    image_ids = data["image_ids"]
    labels = data["labels"]
    
    print(f"Loaded {len(images)} images.")
    
    # 2. Load State
    state = load_state()
    
    # 3. Setup Model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ensemble = EnsembleModel(device=device)
    
    # Add models (The Hybrid Strategy)
    # Group A (Upsampled)
    ensemble.add_imagenet_model('resnet50', weight=1.0)
    ensemble.add_imagenet_model('densenet121', weight=1.0)
    ensemble.add_imagenet_model('vgg16_bn', weight=0.8)
    ensemble.add_imagenet_model('efficientnet_b0', weight=0.8)
    
    # Group B (Native) - TODO: Add if weights available
    # ensemble.add_native_model(..., weight=1.5)
    
    solver = BatchedBSPGD(ensemble, device=device)
    
    # 4. Solver Loop
    # We maintain a tensor of best adversarial examples
    # Initialize with original images
    best_adv_images = images.clone()
    
    if os.path.exists(OUTPUT_FILE):
        print(f"Loading previous best images from {OUTPUT_FILE}...")
        try:
            prev = np.load(OUTPUT_FILE)
            best_adv_images = torch.tensor(prev['images'])
        except:
            print("Could not load previous NPZ, starting fresh.")

    start_time = time.time()
    
    for i in range(len(images)):
        img_id = str(image_ids[i].item())
        img = images[i:i+1].to(device)
        label = labels[i:i+1].to(device)
        
        # Get image specific config from state or defaults
        img_state = state.get(img_id, {})
        kappa = img_state.get('kappa', args.kappa_default)
        
        print(f"\n[{i+1}/100] Image {img_id} (True: {label.item()}) | Kappa: {kappa}")
        
        # Run Attack
        adv, l2, success = solver.attack(
            img, label,
            num_restarts=args.batch_size,
            max_steps=args.steps,
            kappa=kappa,
            epsilon_max=args.epsilon 
        )
        
        # Update State
        if success:
            print(f"  -> Success! L2: {l2:.4f}")
            best_adv_images[i] = adv.cpu()
            img_state['best_local_l2'] = l2
            img_state['status'] = 'success'
        else:
            print("  -> Failed to find adversarial example within limits.")
            img_state['status'] = 'failed'
        
        img_state['last_updated'] = time.time()
        state[img_id] = img_state
        
        # Save periodically
        if i % 5 == 0:
            save_state(state)
            final_images_np = best_adv_images.numpy().astype(np.float32)
            final_ids_np = image_ids.numpy()
            np.savez_compressed(OUTPUT_FILE, images=final_images_np, image_ids=final_ids_np)
            
    # Final Save
    final_images_np = best_adv_images.numpy().astype(np.float32)
    final_ids_np = image_ids.numpy()
    np.savez_compressed(OUTPUT_FILE, images=final_images_np, image_ids=final_ids_np)
    save_state(state)
    
    print(f"\nDone in {(time.time() - start_time)/60:.1f} mins.")

if __name__ == "__main__":
    main()
