#!/usr/bin/env python3
"""
Utility to create a partial submission file from Lock & Ram V2 checkpoint.

This allows testing transfer of current SUCCESS images without waiting
for the full batch to complete.
"""

import torch
import numpy as np
import json
import argparse
from pathlib import Path

from lock_and_ram_v2 import LockAndRamSolverV2


def main():
    parser = argparse.ArgumentParser(
        description='Create partial submission from Lock & Ram V2 checkpoint',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--previous-submission', type=str, required=True,
                       help='Path to previous submission file (e.g., output/submission_ram_test.npz)')
    parser.add_argument('--logits-json', type=str, default=None,
                       help='Path to logits JSON (auto-detected if not provided)')
    parser.add_argument('--checkpoint', type=str, default='./logs/lock_and_ram_v2_checkpoint.json',
                       help='Path to checkpoint file')
    parser.add_argument('--dataset', type=str, default='../natural_images.pt',
                       help='Path to natural_images.pt')
    parser.add_argument('--output-dir', type=str, default='./output',
                       help='Output directory')
    parser.add_argument('--output-name', type=str, default='submission_ram_v2_partial.npz',
                       help='Output filename')
    parser.add_argument('--only-success', action='store_true',
                       help='Only include SUCCESS images from RAM (skip FAILED)')
    parser.add_argument('--num-gpus', type=int, default=1,
                       help='Number of GPUs (for solver initialization, not used here)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Create Partial Submission from Checkpoint (V2)")
    print("=" * 70)
    
    # Initialize solver
    solver = LockAndRamSolverV2(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        log_dir=str(Path(args.checkpoint).parent),
        num_gpus=args.num_gpus
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
    
    # Identify SUCCESS/FAILED from API results
    print(f"\nIdentifying SUCCESS/FAILED from API results: {args.logits_json}")
    success_ids, failed_ids = solver.identify_success_failed_from_api(
        args.logits_json
    )
    
    # Load checkpoint
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"\n❌ Checkpoint not found: {checkpoint_path}")
        print(f"   Run lock_and_ram_v2.py first to generate checkpoint.")
        return
    
    print(f"\nLoading checkpoint: {checkpoint_path}")
    ram_results = solver._load_checkpoint()
    
    if not ram_results:
        print("❌ Could not load checkpoint results")
        return
    
    print(f"✓ Loaded {len(ram_results)} RAM attack results from checkpoint")
    
    # Count SUCCESS in RAM results
    ram_success = [r for r in ram_results.values() if r.success]
    ram_failed = [r for r in ram_results.values() if not r.success]
    
    print(f"  RAM SUCCESS: {len(ram_success)} images")
    print(f"  RAM FAILED:  {len(ram_failed)} images")
    
    # Create partial submission
    print(f"\nCreating partial submission (only_success={args.only_success})...")
    output_path = solver.create_partial_submission(
        previous_adv_images,
        previous_adv_ids,
        success_ids,
        ram_results,
        output_filename=args.output_name,
        only_success=args.only_success
    )
    
    print("\n" + "=" * 70)
    print("Partial Submission Created!")
    print("=" * 70)
    print(f"\nFile: {output_path}")
    print(f"\nNext steps:")
    print(f"1. Test transfer: python analyze.py {output_path} --mode api")
    print(f"2. Continue RAM attack: Run lock_and_ram_v2.py (will resume from checkpoint)")


if __name__ == '__main__':
    main()

