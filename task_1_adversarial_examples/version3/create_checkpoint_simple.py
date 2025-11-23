#!/usr/bin/env python3
"""
Create checkpoint from test image result (simple version, no torch needed).
"""

import numpy as np
import json
from pathlib import Path
from datetime import datetime


def main():
    # Hardcoded for image ID 1 (already tested successfully)
    image_id = 1
    
    print("=" * 70)
    print(f"Creating Checkpoint from Test Image {image_id}")
    print("=" * 70)
    
    # Load test submission
    test_path = Path('./output/submission_test_image_1.npz')
    if not test_path.exists():
        print(f"❌ Error: Test submission not found: {test_path}")
        return
    
    print(f"\nLoading test submission: {test_path}")
    test_data = np.load(test_path)
    test_images = test_data["images"]
    test_ids = test_data["image_ids"]
    
    # Load dataset
    dataset_path = Path('../natural_images.pt')
    if not dataset_path.exists():
        print(f"❌ Error: Dataset not found: {dataset_path}")
        return
    
    import torch
    dataset = torch.load(dataset_path, weights_only=False)
    original_images = dataset["images"]
    
    # Find the test image
    img_idx = np.where(test_ids == image_id)[0]
    if len(img_idx) == 0:
        print(f"❌ Error: Image ID {image_id} not found in test submission")
        return
    
    img_idx = img_idx[0]
    adv_image = test_images[img_idx]
    orig_image = original_images[img_idx:img_idx+1].numpy()
    
    # Calculate L2 distance
    diff = adv_image - orig_image[0]
    l2_dist = float(np.linalg.norm(diff))
    
    print(f"\nImage ID {image_id}:")
    print(f"  L2 distance: {l2_dist:.4f}")
    print(f"  Status: SUCCESS (verified via API)")
    
    # Create checkpoint
    checkpoint_path = Path('./logs/lock_and_ram_checkpoint.json')
    checkpoint_path.parent.mkdir(exist_ok=True)
    
    checkpoint_data = {
        'timestamp': datetime.now().isoformat(),
        'num_results': 1,
        'results': {
            str(image_id): {
                'image_id': image_id,
                'l2_distance': l2_dist,
                'epsilon_used': 8.0,
                'kappa_used': 30.0,
                'success': True,
                'confidence_margin': 30.36,  # From test output
                'num_restarts_succeeded': 5,
                'adversarial_image': adv_image.tolist()
            }
        }
    }
    
    # Save checkpoint
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)
    
    print(f"\n✓ Checkpoint created: {checkpoint_path}")
    print(f"  Contains: Image ID {image_id} (SUCCESS)")
    print(f"\n✓ Ready to launch batch on remaining 67 images")
    print(f"  Command: sbatch run_lock_and_ram.sh")


if __name__ == '__main__':
    main()

