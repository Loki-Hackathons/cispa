#!/usr/bin/env python3
"""
Create submission file from Lock & Ram V4 checkpoint.

This script loads the checkpoint and creates a submission file,
using previous submission for missing images.
"""

import torch
import numpy as np
import json
import argparse
from pathlib import Path

from lock_and_ram_v4 import LockAndRamSolverV4


def main():
    parser = argparse.ArgumentParser(
        description='Create submission from Lock & Ram V4 checkpoint',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--previous-submission', type=str, required=True,
                       help='Path to previous submission file')
    parser.add_argument('--checkpoint', type=str, default='./logs/lock_and_ram_v4_checkpoint.json',
                       help='Path to checkpoint file')
    parser.add_argument('--dataset', type=str, default='../natural_images.pt',
                       help='Path to natural_images.pt')
    parser.add_argument('--output-dir', type=str, default='./output',
                       help='Output directory')
    parser.add_argument('--output-name', type=str, default='submission_ram_v4.npz',
                       help='Output filename')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Create Submission from Checkpoint V4")
    print("=" * 70)
    
    # Initialize solver
    solver = LockAndRamSolverV4(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        log_dir=str(Path(args.checkpoint).parent),
        num_gpus=1  # Not used for this operation
    )
    
    # Load previous submission
    print(f"\nLoading previous submission: {args.previous_submission}")
    previous_adv_images, previous_adv_ids = solver.load_previous_submission(args.previous_submission)
    
    # Load checkpoint
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"\n❌ Checkpoint not found: {checkpoint_path}")
        return
    
    print(f"\nLoading checkpoint: {checkpoint_path}")
    ram_results = solver._load_checkpoint()
    
    if not ram_results:
        print("❌ Could not load checkpoint results")
        return
    
    print(f"✓ Loaded {len(ram_results)} RAM attack results from checkpoint")
    
    # Create submission using merge_and_save
    print(f"\nCreating submission file...")
    output_path = solver.merge_and_save(
        previous_adv_images,
        previous_adv_ids,
        ram_results,
        output_filename=args.output_name
    )
    
    print("\n" + "=" * 70)
    print("Submission Created!")
    print("=" * 70)
    print(f"\nFile: {output_path}")
    print(f"\nNext steps:")
    print(f"1. Analyze with API: python3 analyze.py {output_path} --mode api")
    print(f"2. Submit to leaderboard: python3 submit.py {output_path} --action submit")


if __name__ == '__main__':
    main()

