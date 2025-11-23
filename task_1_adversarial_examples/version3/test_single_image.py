#!/usr/bin/env python3
"""
Test Lock & Ram strategy on a SINGLE image.

This allows quick testing of the freeze & boost strategy on one image
to verify it works before running on all 68 failed images.
"""

import torch
import numpy as np
import argparse
from pathlib import Path

from lock_and_ram import LockAndRamSolver
from models import HybridEnsemble
from attack import BSPGD, AttackConfig


def main():
    parser = argparse.ArgumentParser(
        description='Test Lock & Ram on a single image',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--image-id', type=int, required=True,
                       help='Image ID to test (must be in FAILED list)')
    parser.add_argument('--previous-submission', type=str, required=True,
                       help='Path to previous submission file')
    parser.add_argument('--logits-json', type=str, default=None,
                       help='Path to logits JSON (auto-detected if not provided)')
    parser.add_argument('--dataset', type=str, default='../natural_images.pt',
                       help='Path to natural_images.pt')
    parser.add_argument('--epsilon', type=float, default=8.0,
                       help='Fixed epsilon')
    parser.add_argument('--kappa', type=float, default=30.0,
                       help='Kappa threshold')
    parser.add_argument('--pgd-steps', type=int, default=150,
                       help='PGD steps')
    parser.add_argument('--restarts', type=int, default=5,
                       help='Number of restarts')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print(f"Test Lock & Ram on SINGLE Image ID {args.image_id}")
    print("=" * 70)
    
    # Initialize solver
    solver = LockAndRamSolver(
        dataset_path=args.dataset,
        output_dir='./output',
        log_dir='./logs',
        device=args.device
    )
    
    # Load previous submission
    print(f"\nLoading previous submission: {args.previous_submission}")
    previous_adv_images, previous_adv_ids = solver.load_previous_submission(args.previous_submission)
    
    # Find logits JSON if not provided
    if args.logits_json is None:
        submission_name = Path(args.previous_submission).stem
        try:
            args.logits_json = solver.find_logits_json(submission_name)
            print(f"  Auto-detected logits JSON: {args.logits_json}")
        except FileNotFoundError as e:
            print(f"\n❌ Error: {e}")
            return
    
    # Identify SUCCESS/FAILED
    success_ids, failed_ids = solver.identify_success_failed(
        args.logits_json,
        previous_adv_images,
        previous_adv_ids
    )
    
    # Check if image_id is in FAILED list
    if args.image_id not in failed_ids:
        print(f"\n❌ Error: Image ID {args.image_id} is not in FAILED list!")
        print(f"  SUCCESS images: {sorted(list(success_ids))[:10]}...")
        print(f"  FAILED images: {sorted(list(failed_ids))[:10]}...")
        return
    
    print(f"\n✓ Image ID {args.image_id} is in FAILED list - proceeding with RAM attack")
    
    # Find image in dataset
    img_idx = np.where(solver.image_ids == args.image_id)[0]
    if len(img_idx) == 0:
        print(f"❌ Error: Image ID {args.image_id} not found in dataset")
        return
    
    img_idx = img_idx[0]
    img = solver.images[img_idx:img_idx+1]
    label = solver.labels[img_idx:img_idx+1]
    
    print(f"\nImage details:")
    print(f"  Index in dataset: {img_idx}")
    print(f"  Label: {label[0].item()}")
    print(f"  Shape: {img.shape}")
    
    # Create attack config
    config = AttackConfig(
        epsilon_min=args.epsilon,
        epsilon_max=args.epsilon,
        binary_search_steps=1,
        pgd_steps=args.pgd_steps,
        num_restarts=args.restarts,
        alpha_factor=2.5,
        kappa=args.kappa,
        use_input_diversity=True,
        momentum=0.9,
    )
    
    # Initialize attacker
    attacker = BSPGD(solver.ensemble, config, device=args.device)
    
    # Run attack
    print(f"\n" + "=" * 70)
    print(f"RAM Attack on Image ID {args.image_id}")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Epsilon: {args.epsilon}")
    print(f"  Kappa: {args.kappa}")
    print(f"  PGD steps: {args.pgd_steps}")
    print(f"  Restarts: {args.restarts}")
    
    import time
    start_time = time.time()
    
    result = attacker.attack_single_epsilon(
        img.to(args.device),
        label.to(args.device),
        args.image_id,
        args.epsilon,
        args.kappa
    )
    
    duration = time.time() - start_time
    
    # Calculate normalized L2
    C, H, W = solver.images.shape[1], solver.images.shape[2], solver.images.shape[3]
    l2_norm_factor = np.sqrt(C * H * W)
    l2_normalized = result.l2_distance / l2_norm_factor
    
    # Print result
    print(f"\n" + "=" * 70)
    print("RESULT")
    print("=" * 70)
    status = "✓ SUCCESS" if result.success else "✗ FAILED"
    print(f"  Status: {status}")
    print(f"  L2 (raw): {result.l2_distance:.4f}")
    print(f"  L2 (normalized): {l2_normalized:.6f}")
    print(f"  Epsilon used: {result.epsilon_used:.3f}")
    print(f"  Margin: {result.confidence_margin:+.2f}")
    print(f"  Kappa threshold: {args.kappa}")
    print(f"  Duration: {duration:.1f}s")
    
    if result.success:
        print(f"\n🎉 SUCCESS! Margin ({result.confidence_margin:.2f}) > Kappa ({args.kappa})")
        print(f"   This image should transfer well to the black-box model.")
    else:
        print(f"\n⚠ FAILED: Margin ({result.confidence_margin:.2f}) < Kappa ({args.kappa})")
        print(f"   Consider:")
        print(f"   - Increasing PGD steps (current: {args.pgd_steps})")
        print(f"   - Increasing restarts (current: {args.restarts})")
        print(f"   - Reducing kappa (current: {args.kappa})")
    
    # Save single image result for testing
    print(f"\n" + "=" * 70)
    print("Creating test submission with this single image")
    print("=" * 70)
    
    # Create a minimal submission with:
    # - All 32 SUCCESS from previous (frozen)
    # - This single new RAM result
    # - Original images for the rest
    
    id_to_idx = {img_id: idx for idx, img_id in enumerate(previous_adv_ids)}
    final_images = np.zeros_like(solver.images.numpy(), dtype=np.float32)
    
    ram_results = {args.image_id: result}
    
    frozen_count = 0
    ram_count = 0
    
    for i, img_id in enumerate(solver.image_ids):
        img_id_int = int(img_id)
        
        if img_id_int in success_ids:
            # Frozen from previous
            if img_id_int in id_to_idx:
                final_images[i] = previous_adv_images[id_to_idx[img_id_int]]
                frozen_count += 1
            else:
                final_images[i] = solver.images[i].numpy()
        elif img_id_int == args.image_id:
            # Our test image
            final_images[i] = result.adversarial.cpu().numpy()[0]
            ram_count += 1
        else:
            # Original image
            final_images[i] = solver.images[i].numpy()
    
    output_path = Path('./output') / f'submission_test_image_{args.image_id}.npz'
    np.savez_compressed(
        output_path,
        images=final_images,
        image_ids=solver.image_ids
    )
    
    print(f"\n✓ Test submission saved: {output_path}")
    print(f"  Frozen (SUCCESS): {frozen_count} images")
    print(f"  RAM (test image): {ram_count} image")
    print(f"  Original (rest): {100 - frozen_count - ram_count} images")
    print(f"\nNext step: Test transfer with API")
    print(f"  python analyze.py {output_path} --mode api")


if __name__ == '__main__':
    main()

