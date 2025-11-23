#!/usr/bin/env python3
"""
Create checkpoint from test_single_image result.

This allows you to start the batch from image 1 (already tested successfully).
"""

import torch
import numpy as np
import json
import argparse
from pathlib import Path
from datetime import datetime

from lock_and_ram import LockAndRamSolver
from attack import AttackResult


def main():
    parser = argparse.ArgumentParser(
        description='Create checkpoint from test image result'
    )
    
    parser.add_argument('--test-submission', type=str, required=True,
                       help='Path to test submission (e.g., output/submission_test_image_1.npz)')
    parser.add_argument('--image-id', type=int, required=True,
                       help='Image ID that was tested')
    parser.add_argument('--dataset', type=str, default='../natural_images.pt',
                       help='Path to natural_images.pt')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print(f"Creating Checkpoint from Test Image {args.image_id}")
    print("=" * 70)
    
    # Load test submission
    print(f"\nLoading test submission: {args.test_submission}")
    test_data = np.load(args.test_submission)
    test_images = test_data["images"]
    test_ids = test_data["image_ids"]
    
    # Load dataset
    dataset = torch.load(args.dataset, weights_only=False)
    original_images = dataset["images"]
    
    # Find the test image in the submission
    img_idx = np.where(test_ids == args.image_id)[0]
    if len(img_idx) == 0:
        print(f"❌ Error: Image ID {args.image_id} not found in test submission")
        return
    
    img_idx = img_idx[0]
    adv_image = test_images[img_idx]
    orig_image = original_images[img_idx:img_idx+1]
    
    # Calculate L2 distance
    diff = adv_image - orig_image.numpy()
    l2_dist = float(np.linalg.norm(diff))
    
    # Create AttackResult (we don't have all the details, but we know it was SUCCESS)
    adv_tensor = torch.tensor(adv_image, dtype=torch.float32).unsqueeze(0)
    
    # We'll use placeholder values for fields we don't know
    result = AttackResult(
        image_id=args.image_id,
        adversarial=adv_tensor,
        l2_distance=l2_dist,
        epsilon_used=8.0,  # From test parameters
        kappa_used=30.0,    # From test parameters
        success=True,       # We know it was SUCCESS from API test
        confidence_margin=30.0,  # Approximate (was > 30.0)
        num_restarts_succeeded=5,  # From test parameters
        binary_search_path=[]
    )
    
    # Create checkpoint
    checkpoint_path = Path('./logs/lock_and_ram_checkpoint.json')
    checkpoint_path.parent.mkdir(exist_ok=True)
    
    checkpoint_data = {
        'timestamp': datetime.now().isoformat(),
        'num_results': 1,
        'results': {}
    }
    
    adv_np = result.adversarial.cpu().numpy()[0]
    checkpoint_data['results'][str(args.image_id)] = {
        'image_id': result.image_id,
        'l2_distance': result.l2_distance,
        'epsilon_used': result.epsilon_used,
        'kappa_used': result.kappa_used,
        'success': result.success,
        'confidence_margin': result.confidence_margin,
        'num_restarts_succeeded': result.num_restarts_succeeded,
        'adversarial_image': adv_np.tolist()
    }
    
    # Save checkpoint
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)
    
    print(f"\n✓ Checkpoint created: {checkpoint_path}")
    print(f"  Image ID {args.image_id}: SUCCESS (L2={l2_dist:.4f})")
    print(f"\nNext step: Relaunch batch to process remaining 67 images")
    print(f"  sbatch run_lock_and_ram.sh")


if __name__ == '__main__':
    main()

