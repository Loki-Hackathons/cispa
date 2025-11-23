"""
Lock & Ram Strategy: Freeze & Boost

Strategy:
1. Lock: Load submission_fast.npz (32% success) and identify SUCCESS images
2. Freeze: Copy SUCCESS images directly to final output (L2 ~ 0.06)
3. Ram: Attack only FAILED images with aggressive parameters:
   - No Binary Search (fixed epsilon = 8.0)
   - High kappa = 50.0 (force huge margin)
   - 50 PGD steps
   - 5 restarts
   - Input Diversity enabled
4. Merge: Combine frozen SUCCESS + new RAM attacks

This is ultra-fast because:
- Only 68 images to attack (not 100)
- No Binary Search loop (5x speedup)
- Fixed epsilon (no search overhead)
"""

import torch
import numpy as np
import json
import os
import time
import argparse
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional

from models import HybridEnsemble
from attack import BSPGD, AttackConfig, AttackResult, compute_success_stats


class LockAndRamSolver:
    """
    Lock & Ram Solver: Freeze successful images, boost failed ones.
    
    This strategy maximizes speed by only attacking images that failed
    in the previous run, while preserving successful attacks.
    """
    
    def __init__(
        self,
        dataset_path: str = "../natural_images.pt",
        output_dir: str = "./output",
        log_dir: str = "./logs",
        device: str = 'cuda'
    ):
        self.dataset_path = dataset_path
        self.output_dir = Path(output_dir)
        self.log_dir = Path(log_dir)
        self.device = device
        
        # Create directories
        self.output_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)
        (self.log_dir / "api").mkdir(exist_ok=True)
        
        # Load dataset
        print("=" * 70)
        print("Lock & Ram Solver Initialization")
        print("=" * 70)
        print(f"Device: {device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        print(f"\nLoading dataset: {dataset_path}")
        data = torch.load(dataset_path, weights_only=False)
        self.images = data["images"]
        self.labels = data["labels"]
        self.image_ids = data["image_ids"].numpy()
        
        print(f"✓ Loaded {len(self.images)} images")
        print(f"  Shape: {self.images.shape}")
        print(f"  Labels: {self.labels.shape}")
        
        # Initialize ensemble (fast mode for speed)
        print(f"\nInitializing Hybrid Ensemble (fast_mode=True)...")
        self.ensemble = HybridEnsemble(device=self.device, fast_mode=True)
        
        # Checkpoint file for partial results
        self.checkpoint_path = self.log_dir / "lock_and_ram_checkpoint.json"
        
        # Signal handler for graceful interruption
        self.interrupted = False
        
        print("=" * 70)
    
    def _signal_handler(self, signum, frame):
        """Handle interruption gracefully."""
        print("\n\n⚠ Interruption signal received. Saving checkpoint...")
        self.interrupted = True
    
    def load_previous_submission(self, submission_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load previous submission file.
        
        Returns:
            (adversarial_images, image_ids) where images are numpy arrays
        """
        print(f"\nLoading previous submission: {submission_path}")
        if not os.path.exists(submission_path):
            raise FileNotFoundError(f"Submission file not found: {submission_path}")
        
        data = np.load(submission_path)
        adv_images = data["images"]
        adv_ids = data["image_ids"]
        
        print(f"✓ Loaded {len(adv_images)} adversarial images")
        print(f"  Shape: {adv_images.shape}")
        print(f"  Image IDs: {adv_ids.shape}")
        
        return adv_images, adv_ids
    
    def identify_success_failed(
        self,
        logits_json_path: str,
        adv_images: np.ndarray,
        adv_ids: np.ndarray
    ) -> Tuple[Set[int], Set[int]]:
        """
        Identify SUCCESS and FAILED image IDs from API logits JSON.
        
        Args:
            logits_json_path: Path to JSON file with API logits results
            adv_images: Adversarial images from previous submission
            adv_ids: Image IDs from previous submission
        
        Returns:
            (success_ids, failed_ids) as sets of image IDs
        """
        print(f"\nIdentifying SUCCESS/FAILED from: {logits_json_path}")
        
        if not os.path.exists(logits_json_path):
            raise FileNotFoundError(f"Logits JSON not found: {logits_json_path}")
        
        # Load logits JSON
        with open(logits_json_path, 'r') as f:
            logits_data = json.load(f)
        
        # Load true labels
        dataset = torch.load(self.dataset_path, weights_only=False)
        true_labels = dataset["labels"].numpy()
        
        # Parse results
        results = logits_data["results"]
        results.sort(key=lambda x: x["image_id"])
        
        success_ids = set()
        failed_ids = set()
        
        for res in results:
            img_id = res["image_id"]
            logits = np.array(res["logits"])
            predicted_class = int(np.argmax(logits))
            true_class = int(true_labels[img_id])
            
            is_misclassified = (predicted_class != true_class)
            
            if is_misclassified:
                success_ids.add(img_id)
            else:
                failed_ids.add(img_id)
        
        print(f"✓ Identified:")
        print(f"  SUCCESS: {len(success_ids)} images")
        print(f"  FAILED:  {len(failed_ids)} images")
        
        if len(success_ids) + len(failed_ids) != len(self.images):
            print(f"⚠ Warning: Total ({len(success_ids) + len(failed_ids)}) != dataset size ({len(self.images)})")
        
        return success_ids, failed_ids
    
    def find_logits_json(self, submission_name: str) -> str:
        """
        Find the corresponding logits JSON file for a submission.
        
        Looks for files matching: logs/api/logits_<submission_name>*.json
        """
        api_log_dir = self.log_dir / "api"
        
        # Try exact match first
        exact_match = api_log_dir / f"logits_{submission_name}.json"
        if exact_match.exists():
            return str(exact_match)
        
        # Try pattern match
        pattern = f"logits_{submission_name}_*.json"
        matches = list(api_log_dir.glob(pattern))
        
        if matches:
            # Return most recent
            matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return str(matches[0])
        
        # Try to find any logits file with submission name in it
        all_logits = list(api_log_dir.glob("logits_*.json"))
        for log_file in sorted(all_logits, key=lambda p: p.stat().st_mtime, reverse=True):
            if submission_name in log_file.name:
                return str(log_file)
        
        raise FileNotFoundError(
            f"Could not find logits JSON for {submission_name}. "
            f"Looked in: {api_log_dir}. "
            f"Please run: python analyze.py output/{submission_name}.npz --mode api"
        )
    
    def ram_attack_failed(
        self,
        failed_ids: Set[int],
        epsilon: float = 8.0,
        kappa: float = 50.0,
        pgd_steps: int = 50,
        num_restarts: int = 5,
        alpha_factor: float = 2.5,
        momentum: float = 0.9
    ) -> Dict[int, AttackResult]:
        """
        RAM attack: Aggressive attack on failed images only.
        
        Args:
            failed_ids: Set of image IDs to attack
            epsilon: Fixed epsilon (no binary search)
            kappa: High confidence margin requirement
            pgd_steps: Number of PGD steps
            num_restarts: Number of random restarts
            alpha_factor: Step size factor
            momentum: MI-FGSM momentum
        
        Returns:
            Dictionary mapping image_id -> AttackResult
        """
        print("\n" + "=" * 70)
        print("RAM ATTACK: Aggressive Attack on Failed Images")
        print("=" * 70)
        print(f"\nConfiguration:")
        print(f"  Target images: {len(failed_ids)}")
        print(f"  Epsilon (fixed): {epsilon}")
        print(f"  Kappa: {kappa}")
        print(f"  PGD steps: {pgd_steps}")
        print(f"  Restarts: {num_restarts}")
        print(f"  Alpha factor: {alpha_factor}")
        print(f"  Momentum: {momentum}")
        print(f"  Input Diversity: ENABLED")
        print(f"  Binary Search: DISABLED (fixed epsilon)")
        
        # Create attack config (no binary search - single epsilon)
        config = AttackConfig(
            epsilon_min=epsilon,
            epsilon_max=epsilon,
            binary_search_steps=1,  # Single step = no search
            pgd_steps=pgd_steps,
            num_restarts=num_restarts,
            alpha_factor=alpha_factor,
            kappa=kappa,
            use_input_diversity=True,
            momentum=momentum,
        )
        
        # Initialize attacker
        attacker = BSPGD(self.ensemble, config, device=self.device)
        
        # Load checkpoint if exists
        checkpoint_results = self._load_checkpoint()
        if checkpoint_results:
            print(f"✓ Loaded {len(checkpoint_results)} results from checkpoint")
            # Filter out already processed images
            failed_list = [img_id for img_id in sorted(failed_ids) if img_id not in checkpoint_results]
            results = checkpoint_results
        else:
            results = {}
            failed_list = sorted(failed_ids)
        
        start_time = time.time()
        
        try:
            for idx, img_id in enumerate(failed_list):
                # Check for interruption
                if self.interrupted:
                    print(f"\n⚠ Interrupted at image {idx+1}/{len(failed_list)}. Saving checkpoint...")
                    self._save_checkpoint(results)
                    print(f"✓ Checkpoint saved. Resume by running the same command.")
                    return results
                
                # Find image index in dataset
                img_idx = np.where(self.image_ids == img_id)[0]
                if len(img_idx) == 0:
                    print(f"⚠ Warning: Image ID {img_id} not found in dataset, skipping")
                    continue
                
                img_idx = img_idx[0]
                
                img = self.images[img_idx:img_idx+1]
                label = self.labels[img_idx:img_idx+1]
                
                img_start = time.time()
                
                print(f"\n[{idx+1:3d}/{len(failed_list)}] RAM Attack on Image ID {img_id:3d} (Label: {label[0].item():3d})")
                
                # Run attack with single fixed epsilon (no binary search)
                result = attacker.attack_single_epsilon(
                    img.to(self.device),
                    label.to(self.device),
                    img_id,
                    epsilon,
                    kappa
                )
                
                results[img_id] = result
                    
                # Save checkpoint after EACH image - CRITICAL for testing and recovery
                try:
                    self._save_checkpoint(results)
                    print(f"  💾 Checkpoint saved (image {img_id})")
                except Exception as e:
                    print(f"  ⚠ ERROR saving checkpoint: {e}", file=sys.stderr)
                    import traceback
                    traceback.print_exc()
                
                img_duration = time.time() - img_start
                
                # Calculate normalized L2
                C, H, W = self.images.shape[1], self.images.shape[2], self.images.shape[3]
                l2_norm_factor = np.sqrt(C * H * W)
                l2_normalized = result.l2_distance / l2_norm_factor
                
                # Print result
                status = "✓ SUCCESS" if result.success else "✗ FAILED"
                print(f"  {status}")
                print(f"  L2: {result.l2_distance:.4f} (norm: {l2_normalized:.4f}) | " +
                      f"ε: {result.epsilon_used:.3f} | Margin: {result.confidence_margin:+.2f} | " +
                      f"Time: {img_duration:.1f}s")
                
                # Progress estimate
                if (idx + 1) % 10 == 0:
                    elapsed = time.time() - start_time
                    avg_time = elapsed / (idx + 1)
                    remaining = avg_time * (len(failed_list) - idx - 1)
                    print(f"\n  Progress: {idx+1}/{len(failed_list)} | " +
                          f"Elapsed: {elapsed/60:.1f}min | " +
                          f"Remaining: ~{remaining/60:.1f}min")
        except KeyboardInterrupt:
            print(f"\n\n⚠ KeyboardInterrupt caught at image {idx+1}/{len(failed_list)}. Saving checkpoint...")
            self._save_checkpoint(results)
            print(f"✓ Checkpoint saved to: {self.checkpoint_path}")
            print(f"  Resume by running the same command.")
            raise  # Re-raise to exit properly
        except Exception as e:
            print(f"\n\n⚠ Exception occurred: {e}")
            print(f"Saving checkpoint before exit...")
            self._save_checkpoint(results)
            print(f"✓ Checkpoint saved to: {self.checkpoint_path}")
            raise
        
        total_duration = time.time() - start_time
        
        print("\n" + "=" * 70)
        print("RAM Attack Complete")
        print("=" * 70)
        print(f"\nTotal duration: {total_duration/60:.2f} minutes")
        print(f"Average time per image: {total_duration/len(failed_list):.1f} seconds")
        
        # Compute stats
        ram_results_list = list(results.values())
        stats = compute_success_stats(ram_results_list)
        
        print(f"\nRAM Attack Results:")
        print(f"  Successful: {stats['successful']} ({stats['success_rate']:.1f}%)")
        print(f"  Failed:     {stats['failed']}")
        
        C, H, W = self.images.shape[1], self.images.shape[2], self.images.shape[3]
        l2_norm_factor = np.sqrt(C * H * W)
        
        print(f"\nL2 Distances:")
        print(f"  Average (all, norm):    {stats['avg_l2_all']/l2_norm_factor:.6f}")
        print(f"  Average (success, norm): {stats['avg_l2_success']/l2_norm_factor:.6f}")
        print(f"\nConfidence Margins:")
        print(f"  Average (all):     {stats['avg_margin_all']:+.3f}")
        print(f"  Average (success): {stats['avg_margin_success']:+.3f}")
        
        # Clear checkpoint on successful completion
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()
            print(f"✓ Cleared checkpoint (run completed)")
        
        return results
    
    def _save_checkpoint(self, results: Dict[int, AttackResult]):
        """Save partial results to checkpoint file."""
        try:
            checkpoint_data = {
                'timestamp': datetime.now().isoformat(),
                'results': {}
            }
            
            for img_id, result in results.items():
                # Save adversarial image as numpy array
                adv_np = result.adversarial.cpu().numpy()[0]
                checkpoint_data['results'][str(img_id)] = {
                    'image_id': result.image_id,
                    'l2_distance': result.l2_distance,
                    'epsilon_used': result.epsilon_used,
                    'kappa_used': result.kappa_used,
                    'success': result.success,
                    'confidence_margin': result.confidence_margin,
                    'num_restarts_succeeded': result.num_restarts_succeeded,
                    'adversarial_image': adv_np.tolist()  # Convert to list for JSON
                }
            
            # Use atomic write (write to temp file then rename)
            temp_path = self.checkpoint_path.with_suffix('.tmp')
            with open(temp_path, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            temp_path.replace(self.checkpoint_path)
            
        except Exception as e:
            print(f"  ⚠ ERROR saving checkpoint: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
    
    def _load_checkpoint(self) -> Optional[Dict[int, AttackResult]]:
        """Load partial results from checkpoint file."""
        if not self.checkpoint_path.exists():
            return None
        
        try:
            with open(self.checkpoint_path, 'r') as f:
                checkpoint_data = json.load(f)
            
            results = {}
            for img_id_str, data in checkpoint_data['results'].items():
                img_id = int(img_id_str)
                # Reconstruct AttackResult
                adv_tensor = torch.tensor(data['adversarial_image'], dtype=torch.float32).unsqueeze(0)
                
                result = AttackResult(
                    image_id=data['image_id'],
                    adversarial=adv_tensor,
                    l2_distance=data['l2_distance'],
                    epsilon_used=data['epsilon_used'],
                    kappa_used=data['kappa_used'],
                    success=data['success'],
                    confidence_margin=data['confidence_margin'],
                    num_restarts_succeeded=data['num_restarts_succeeded'],
                    binary_search_path=[]  # Not saved in checkpoint
                )
                results[img_id] = result
            
            return results
        except Exception as e:
            print(f"⚠ Warning: Could not load checkpoint: {e}")
            return None
    
    def create_partial_submission(
        self,
        previous_adv_images: np.ndarray,
        previous_adv_ids: np.ndarray,
        success_ids: Set[int],
        ram_results: Dict[int, AttackResult],
        output_filename: str = "submission_ram_partial.npz",
        only_success: bool = False
    ) -> Path:
        """
        Create a partial submission file with current results.
        
        Args:
            previous_adv_images: Adversarial images from previous submission
            previous_adv_ids: Image IDs from previous submission
            success_ids: Set of image IDs that succeeded (to freeze)
            ram_results: Dictionary of new RAM attack results
            output_filename: Output filename
            only_success: If True, only include SUCCESS images from RAM (for testing transfer)
        
        Returns:
            Path to saved file
        """
        print("\n" + "=" * 70)
        print("Creating Partial Submission")
        print("=" * 70)
        
        # Create mapping from image_id to index in previous submission
        id_to_idx = {img_id: idx for idx, img_id in enumerate(previous_adv_ids)}
        
        # Initialize final images array (same shape as dataset)
        final_images = np.zeros_like(self.images.numpy(), dtype=np.float32)
        
        # Process all images in dataset order
        frozen_count = 0
        ram_success_count = 0
        ram_failed_count = 0
        missing_count = 0
        
        for i, img_id in enumerate(self.image_ids):
            img_id_int = int(img_id)
            
            if img_id_int in success_ids:
                # LOCK: Copy from previous submission
                if img_id_int in id_to_idx:
                    prev_idx = id_to_idx[img_id_int]
                    final_images[i] = previous_adv_images[prev_idx]
                    frozen_count += 1
                else:
                    print(f"⚠ Warning: SUCCESS image ID {img_id_int} not found in previous submission")
                    final_images[i] = self.images[i].numpy()
                    missing_count += 1
            elif img_id_int in ram_results:
                result = ram_results[img_id_int]
                if only_success and not result.success:
                    # Skip failed RAM attacks if only_success=True
                    final_images[i] = self.images[i].numpy()  # Use original
                    ram_failed_count += 1
                else:
                    # RAM: Use new attack result (success or failed)
                    final_images[i] = result.adversarial.cpu().numpy()[0]
                    if result.success:
                        ram_success_count += 1
                    else:
                        ram_failed_count += 1
            else:
                # Missing: use original image
                final_images[i] = self.images[i].numpy()
                missing_count += 1
        
        print(f"\nPartial Submission Summary:")
        print(f"  Frozen (SUCCESS from previous): {frozen_count} images")
        if only_success:
            print(f"  RAM SUCCESS (new attacks):     {ram_success_count} images")
            print(f"  RAM FAILED (skipped):          {ram_failed_count} images")
        else:
            print(f"  RAM SUCCESS (new attacks):     {ram_success_count} images")
            print(f"  RAM FAILED (included):        {ram_failed_count} images")
        print(f"  Missing (using original):      {missing_count} images")
        print(f"  Total:                          {len(final_images)} images")
        
        # Save submission
        output_path = self.output_dir / output_filename
        np.savez_compressed(
            output_path,
            images=final_images,
            image_ids=self.image_ids
        )
        
        print(f"\n✓ Saved partial submission to: {output_path}")
        print(f"  Shape: {final_images.shape}")
        print(f"  Dtype: {final_images.dtype}")
        
        return output_path
    
    def merge_and_save(
        self,
        previous_adv_images: np.ndarray,
        previous_adv_ids: np.ndarray,
        success_ids: Set[int],
        ram_results: Dict[int, AttackResult],
        output_filename: str = "submission_ram_test.npz"
    ):
        """
        Merge frozen SUCCESS images with new RAM attack results.
        
        Args:
            previous_adv_images: Adversarial images from previous submission
            previous_adv_ids: Image IDs from previous submission
            success_ids: Set of image IDs that succeeded (to freeze)
            ram_results: Dictionary of new RAM attack results
            output_filename: Output filename
        """
        print("\n" + "=" * 70)
        print("Merging Results: Lock (Frozen) + Ram (New)")
        print("=" * 70)
        
        # Create mapping from image_id to index in previous submission
        id_to_idx = {img_id: idx for idx, img_id in enumerate(previous_adv_ids)}
        
        # Initialize final images array (same shape as dataset)
        final_images = np.zeros_like(self.images.numpy(), dtype=np.float32)
        
        # Process all images in dataset order
        frozen_count = 0
        ram_count = 0
        missing_count = 0
        
        for i, img_id in enumerate(self.image_ids):
            img_id_int = int(img_id)
            
            if img_id_int in success_ids:
                # LOCK: Copy from previous submission
                if img_id_int in id_to_idx:
                    prev_idx = id_to_idx[img_id_int]
                    final_images[i] = previous_adv_images[prev_idx]
                    frozen_count += 1
                else:
                    print(f"⚠ Warning: SUCCESS image ID {img_id_int} not found in previous submission")
                    # Fallback: use original image
                    final_images[i] = self.images[i].numpy()
                    missing_count += 1
            elif img_id_int in ram_results:
                # RAM: Use new attack result
                result = ram_results[img_id_int]
                final_images[i] = result.adversarial.cpu().numpy()[0]
                ram_count += 1
            else:
                # Missing: use original image (shouldn't happen)
                print(f"⚠ Warning: Image ID {img_id_int} not in SUCCESS or RAM results, using original")
                final_images[i] = self.images[i].numpy()
                missing_count += 1
        
        print(f"\nMerge Summary:")
        print(f"  Frozen (SUCCESS): {frozen_count} images")
        print(f"  RAM (new attack): {ram_count} images")
        print(f"  Missing:          {missing_count} images")
        print(f"  Total:            {len(final_images)} images")
        
        # Save submission
        output_path = self.output_dir / output_filename
        np.savez_compressed(
            output_path,
            images=final_images,
            image_ids=self.image_ids
        )
        
        print(f"\n✓ Saved merged submission to: {output_path}")
        print(f"  Shape: {final_images.shape}")
        print(f"  Dtype: {final_images.dtype}")
        
        return output_path
    
    def run(
        self,
        previous_submission: str,
        logits_json: str = None,
        epsilon: float = 8.0,
        kappa: float = 50.0,
        pgd_steps: int = 50,
        num_restarts: int = 5,
        output_filename: str = "submission_ram_test.npz"
    ):
        """
        Run Lock & Ram strategy.
        
        Args:
            previous_submission: Path to previous submission file (e.g., submission_fast.npz)
            logits_json: Path to logits JSON (auto-detected if None)
            epsilon: Fixed epsilon for RAM attack
            kappa: High kappa for RAM attack
            pgd_steps: PGD steps for RAM attack
            num_restarts: Restarts for RAM attack
            output_filename: Output filename
        """
        print("\n" + "=" * 70)
        print("LOCK & RAM STRATEGY")
        print("=" * 70)
        print(f"\nStep 1: LOCK - Loading previous submission")
        print(f"  File: {previous_submission}")
        
        # Load previous submission
        previous_adv_images, previous_adv_ids = self.load_previous_submission(previous_submission)
        
        # Find logits JSON if not provided
        if logits_json is None:
            submission_name = Path(previous_submission).stem
            try:
                logits_json = self.find_logits_json(submission_name)
                print(f"  Auto-detected logits JSON: {logits_json}")
            except FileNotFoundError as e:
                print(f"\n❌ Error: {e}")
                print(f"\nTo generate logits JSON, run:")
                print(f"  python analyze.py {previous_submission} --mode api")
                return
        
        # Identify SUCCESS and FAILED
        print(f"\nStep 2: LOCK - Identifying SUCCESS/FAILED images")
        success_ids, failed_ids = self.identify_success_failed(
            logits_json,
            previous_adv_images,
            previous_adv_ids
        )
        
        print(f"\n✓ Locked {len(success_ids)} SUCCESS images (will be frozen)")
        print(f"✓ Targeting {len(failed_ids)} FAILED images for RAM attack")
        
        # Set up signal handler for this run
        def signal_handler(signum, frame):
            self._signal_handler(signum, frame)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # RAM attack on failed images
        print(f"\nStep 3: RAM - Aggressive attack on failed images")
        print(f"  Note: Results are saved to checkpoint after each image.")
        print(f"  Press Ctrl+C to interrupt and create partial submission.")
        ram_results = self.ram_attack_failed(
            failed_ids,
            epsilon=epsilon,
            kappa=kappa,
            pgd_steps=pgd_steps,
            num_restarts=num_restarts
        )
        
        # Merge and save
        print(f"\nStep 4: MERGE - Combining frozen + RAM results")
        output_path = self.merge_and_save(
            previous_adv_images,
            previous_adv_ids,
            success_ids,
            ram_results,
            output_filename
        )
        
        print("\n" + "=" * 70)
        print("LOCK & RAM COMPLETE!")
        print("=" * 70)
        print(f"\nOutput saved to: {output_path}")
        print(f"\nNext steps:")
        print(f"1. Review results: python analyze.py {output_path} --mode local")
        print(f"2. Submit to API: python submit.py {output_path} --action submit")
        print(f"3. Check leaderboard: http://34.122.51.94:80/leaderboard_page")


def main():
    parser = argparse.ArgumentParser(
        description='Lock & Ram Strategy: Freeze successful images, boost failed ones',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # I/O paths
    parser.add_argument('--previous-submission', type=str, required=True,
                       help='Path to previous submission file (e.g., output/submission_fast.npz)')
    parser.add_argument('--logits-json', type=str, default=None,
                       help='Path to logits JSON (auto-detected if not provided)')
    parser.add_argument('--dataset', type=str, default='../natural_images.pt',
                       help='Path to natural_images.pt')
    parser.add_argument('--output-dir', type=str, default='./output',
                       help='Output directory for submissions')
    parser.add_argument('--log-dir', type=str, default='./logs',
                       help='Log directory')
    parser.add_argument('--output-name', type=str, default='submission_ram_test.npz',
                       help='Output filename')
    
    # RAM attack config
    parser.add_argument('--epsilon', type=float, default=8.0,
                       help='Fixed epsilon for RAM attack (no binary search)')
    parser.add_argument('--kappa', type=float, default=50.0,
                       help='High kappa for RAM attack (force huge margin)')
    parser.add_argument('--pgd-steps', type=int, default=50,
                       help='PGD steps for RAM attack')
    parser.add_argument('--restarts', type=int, default=5,
                       help='Number of random restarts')
    parser.add_argument('--alpha-factor', type=float, default=2.5,
                       help='Alpha factor for step size')
    parser.add_argument('--momentum', type=float, default=0.9,
                       help='MI-FGSM momentum')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Initialize solver
    solver = LockAndRamSolver(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        log_dir=args.log_dir,
        device=args.device
    )
    
    # Run Lock & Ram
    solver.run(
        previous_submission=args.previous_submission,
        logits_json=args.logits_json,
        epsilon=args.epsilon,
        kappa=args.kappa,
        pgd_steps=args.pgd_steps,
        num_restarts=args.restarts,
        output_filename=args.output_name
    )


if __name__ == '__main__':
    main()

