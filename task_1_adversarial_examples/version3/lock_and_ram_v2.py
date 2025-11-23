"""
Lock & Ram Strategy V2: Freeze & Boost with Multi-GPU Support

Improvements over V1:
- Uses API results (not local) to identify failures
- Parallelizes attacks across 2 GPUs
- Aggressive parameters: kappa=50, epsilon=10, steps=250, restarts=10
- Only attacks images that failed on the black-box API
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
from threading import Thread
from queue import Queue

from models import HybridEnsemble
from attack import BSPGD, AttackConfig, AttackResult, compute_success_stats


class LockAndRamSolverV2:
    """
    Lock & Ram Solver V2: Multi-GPU parallelized version.
    
    Uses API results to identify failures and attacks only those.
    Parallelizes across 2 GPUs for speed.
    """
    
    def __init__(
        self,
        dataset_path: str = "../natural_images.pt",
        output_dir: str = "./output",
        log_dir: str = "./logs",
        num_gpus: int = 2
    ):
        self.dataset_path = dataset_path
        self.output_dir = Path(output_dir)
        self.log_dir = Path(log_dir)
        self.num_gpus = num_gpus
        
        # Create directories
        self.output_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)
        (self.log_dir / "api").mkdir(exist_ok=True)
        
        # Load dataset
        print("=" * 70)
        print("Lock & Ram Solver V2 - Multi-GPU Initialization")
        print("=" * 70)
        print(f"GPUs: {num_gpus}")
        if torch.cuda.is_available():
            for i in range(min(num_gpus, torch.cuda.device_count())):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
        
        print(f"\nLoading dataset: {dataset_path}")
        data = torch.load(dataset_path, weights_only=False)
        self.images = data["images"]
        self.labels = data["labels"]
        self.image_ids = data["image_ids"].numpy()
        
        print(f"✓ Loaded {len(self.images)} images")
        print(f"  Shape: {self.images.shape}")
        print(f"  Labels: {self.labels.shape}")
        
        # Checkpoint file for partial results
        self.checkpoint_path = self.log_dir / "lock_and_ram_v2_checkpoint.json"
        
        # Signal handler for graceful interruption
        self.interrupted = False
        
        print("=" * 70)
    
    def _signal_handler(self, signum, frame):
        """Handle interruption gracefully."""
        print("\n\n⚠ Interruption signal received. Saving checkpoint...")
        self.interrupted = True
    
    def load_previous_submission(self, submission_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load previous submission file."""
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
    
    def find_logits_json(self, submission_name: str) -> str:
        """
        Find the most recent logits JSON file for a submission.
        
        Args:
            submission_name: Name of submission (e.g., "submission_ram_test")
        
        Returns:
            Path to logits JSON file
        
        Raises:
            FileNotFoundError: If no logits JSON found
        """
        logits_dir = self.log_dir / "api"
        pattern = f"logits_{submission_name}_*.json"
        
        import glob
        matches = list(logits_dir.glob(pattern))
        
        if not matches:
            raise FileNotFoundError(
                f"No logits JSON found for submission '{submission_name}'\n"
                f"  Searched in: {logits_dir}\n"
                f"  Pattern: {pattern}\n"
                f"  Generate it with: python analyze.py output/{submission_name}.npz --mode api"
            )
        
        # Return most recent (by modification time)
        latest = max(matches, key=lambda p: p.stat().st_mtime)
        return str(latest)
    
    def identify_success_failed_from_api(
        self,
        api_logits_json_path: str
    ) -> Tuple[Set[int], Set[int]]:
        """
        Identify SUCCESS and FAILED image IDs from API logits JSON.
        
        This uses the REAL black-box API results, not local surrogate results.
        
        Args:
            api_logits_json_path: Path to JSON file with API logits results
        
        Returns:
            (success_ids, failed_ids) as sets of image IDs
        """
        print(f"\nIdentifying SUCCESS/FAILED from API results: {api_logits_json_path}")
        
        if not os.path.exists(api_logits_json_path):
            raise FileNotFoundError(f"API logits JSON not found: {api_logits_json_path}")
        
        # Load logits JSON
        with open(api_logits_json_path, 'r') as f:
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
        
        print(f"✓ Identified from API results:")
        print(f"  SUCCESS (misclassified): {len(success_ids)} images")
        print(f"  FAILED (correctly classified): {len(failed_ids)} images")
        
        if len(success_ids) + len(failed_ids) != len(self.images):
            print(f"⚠ Warning: Total ({len(success_ids) + len(failed_ids)}) != dataset size ({len(self.images)})")
        
        return success_ids, failed_ids
    
    def _worker_thread(
        self,
        gpu_id: int,
        image_list: List[Tuple[int, int]],
        result_dict: Dict[int, AttackResult],
        config_dict: dict,
        lock
    ):
        """Worker thread for GPU parallelization."""
        # Set GPU
        torch.cuda.set_device(gpu_id)
        device = f'cuda:{gpu_id}'
        
        # Load dataset
        data = torch.load(self.dataset_path, weights_only=False)
        images = data["images"]
        labels = data["labels"]
        
        # Initialize ensemble on this GPU
        ensemble = HybridEnsemble(device=device, fast_mode=True)
        
        # Create attack config (remove 'epsilon' key as it's not a valid AttackConfig parameter)
        config_dict_clean = {k: v for k, v in config_dict.items() if k != 'epsilon'}
        config = AttackConfig(**config_dict_clean)
        attacker = BSPGD(ensemble, config, device=device)
        
        print(f"[GPU {gpu_id}] Worker started - {len(image_list)} images")
        
        for img_id, img_idx in image_list:
            try:
                img = images[img_idx:img_idx+1].to(device)
                label = labels[img_idx:img_idx+1].to(device)
                
                # Run attack
                epsilon = config_dict['epsilon_min']  # Use epsilon_min (same as epsilon_max for fixed epsilon)
                kappa = config_dict['kappa']
                result = attacker.attack_single_epsilon(
                    img, label, img_id,
                    epsilon,
                    kappa
                )
                
                # Move result to CPU
                result.adversarial = result.adversarial.cpu()
                
                # Store result (thread-safe)
                with lock:
                    result_dict[img_id] = result
                
                # Calculate normalized L2
                C, H, W = self.images.shape[1], self.images.shape[2], self.images.shape[3]
                l2_norm_factor = np.sqrt(C * H * W)
                l2_normalized = result.l2_distance / l2_norm_factor
                
                # Print result
                status = "✓ SUCCESS" if result.success else "✗ FAILED"
                print(f"[GPU {gpu_id}] Image ID {img_id:3d}: {status} | " +
                      f"L2: {result.l2_distance:.4f} (norm: {l2_normalized:.4f}) | " +
                      f"Margin: {result.confidence_margin:+.2f}")
                
            except Exception as e:
                print(f"[GPU {gpu_id}] Error processing image {img_id}: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"[GPU {gpu_id}] Worker finished")
    
    def ram_attack_failed_parallel(
        self,
        failed_ids: Set[int],
        epsilon: float = 10.0,
        kappa: float = 50.0,
        pgd_steps: int = 250,
        num_restarts: int = 10,
        alpha_factor: float = 2.5,
        momentum: float = 0.9
    ) -> Dict[int, AttackResult]:
        """
        RAM attack: Aggressive attack on failed images with multi-GPU parallelization.
        
        Args:
            failed_ids: Set of image IDs to attack (from API results)
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
        print("RAM ATTACK V2: Multi-GPU Parallelized Attack on API Failures")
        print("=" * 70)
        print(f"\nConfiguration:")
        print(f"  Target images: {len(failed_ids)} (from API failures)")
        print(f"  GPUs: {self.num_gpus}")
        print(f"  Epsilon (fixed): {epsilon}")
        print(f"  Kappa: {kappa}")
        print(f"  PGD steps: {pgd_steps}")
        print(f"  Restarts: {num_restarts}")
        print(f"  Alpha factor: {alpha_factor}")
        print(f"  Momentum: {momentum}")
        print(f"  Input Diversity: ENABLED")
        print(f"  Binary Search: DISABLED (fixed epsilon)")
        
        # Load checkpoint if exists
        checkpoint_results = self._load_checkpoint()
        if checkpoint_results is None:
            checkpoint_results = {}
        
        if checkpoint_results:
            print(f"✓ Loaded {len(checkpoint_results)} results from checkpoint")
            # Filter out already processed images
            failed_list = [img_id for img_id in sorted(failed_ids) if img_id not in checkpoint_results]
            results = checkpoint_results.copy()
        else:
            results = {}
            failed_list = sorted(failed_ids)
        
        if not failed_list:
            print("✓ All images already processed!")
            return results
        
        print(f"  Remaining to process: {len(failed_list)} images")
        
        # Create attack config dict (include epsilon for worker access)
        config_dict = {
            'epsilon': epsilon,  # For worker access
            'epsilon_min': epsilon,
            'epsilon_max': epsilon,
            'binary_search_steps': 1,
            'pgd_steps': pgd_steps,
            'num_restarts': num_restarts,
            'alpha_factor': alpha_factor,
            'kappa': kappa,
            'use_input_diversity': True,
            'momentum': momentum,
        }
        
        # Prepare image lists for each GPU (divide work)
        image_tasks = []
        for img_id in failed_list:
            img_idx = np.where(self.image_ids == img_id)[0]
            if len(img_idx) > 0:
                image_tasks.append((img_id, img_idx[0]))
        
        # Divide tasks between GPUs
        tasks_per_gpu = len(image_tasks) // self.num_gpus
        gpu_tasks = []
        for gpu_id in range(self.num_gpus):
            start_idx = gpu_id * tasks_per_gpu
            if gpu_id == self.num_gpus - 1:
                # Last GPU gets remaining tasks
                end_idx = len(image_tasks)
            else:
                end_idx = (gpu_id + 1) * tasks_per_gpu
            gpu_tasks.append(image_tasks[start_idx:end_idx])
            print(f"  GPU {gpu_id}: {len(gpu_tasks[gpu_id])} images")
        
        # Shared result dictionary (thread-safe with lock)
        import threading
        result_lock = threading.Lock()
        worker_results = {}
        
        # Start worker threads
        threads = []
        start_time = time.time()
        
        for gpu_id in range(self.num_gpus):
            if len(gpu_tasks[gpu_id]) > 0:
                t = Thread(
                    target=self._worker_thread,
                    args=(gpu_id, gpu_tasks[gpu_id], worker_results, config_dict, result_lock)
                )
                t.start()
                threads.append(t)
        
        # Monitor progress and save checkpoint periodically
        try:
            while any(t.is_alive() for t in threads):
                if self.interrupted:
                    print(f"\n⚠ Interrupted. Saving checkpoint...")
                    with result_lock:
                        results.update(worker_results)
                    self._save_checkpoint(results)
                    break
                
                time.sleep(5)  # Check every 5 seconds
                
                # Update results and save checkpoint
                with result_lock:
                    results.update(worker_results)
                    if len(results) > len(checkpoint_results):
                        self._save_checkpoint(results)
                        completed = len(results) - len(checkpoint_results)
                        print(f"\n💾 Progress: {completed}/{len(failed_list)} images processed")
        
        except KeyboardInterrupt:
            print(f"\n\n⚠ KeyboardInterrupt caught. Saving checkpoint...")
            with result_lock:
                results.update(worker_results)
            self._save_checkpoint(results)
            raise
        
        # Wait for all threads to finish
        for t in threads:
            t.join()
        
        # Final update
        with result_lock:
            results.update(worker_results)
        
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
        if self.checkpoint_path.exists() and completed == len(failed_list):
            self.checkpoint_path.unlink()
            print(f"✓ Cleared checkpoint (run completed)")
        
        return results
    
    def _save_checkpoint(self, results: Dict[int, AttackResult]):
        """Save partial results to checkpoint file."""
        try:
            checkpoint_data = {
                'timestamp': datetime.now().isoformat(),
                'num_results': len(results),
                'results': {}
            }
            
            for img_id, result in results.items():
                adv_np = result.adversarial.cpu().numpy()[0]
                checkpoint_data['results'][str(img_id)] = {
                    'image_id': result.image_id,
                    'l2_distance': result.l2_distance,
                    'epsilon_used': result.epsilon_used,
                    'kappa_used': result.kappa_used,
                    'success': result.success,
                    'confidence_margin': result.confidence_margin,
                    'num_restarts_succeeded': result.num_restarts_succeeded,
                    'adversarial_image': adv_np.tolist()
                }
            
            temp_path = self.checkpoint_path.with_suffix('.tmp')
            with open(temp_path, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            temp_path.replace(self.checkpoint_path)
            
        except Exception as e:
            print(f"  ⚠ ERROR saving checkpoint: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            raise
    
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
                    binary_search_path=[]
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
        output_filename: str = "submission_ram_v2_partial.npz",
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
        print(f"  Frozen (SUCCESS from API): {frozen_count} images")
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
        output_filename: str = "submission_ram_v2.npz"
    ) -> Path:
        """Merge frozen SUCCESS images with new RAM attack results."""
        print("\n" + "=" * 70)
        print("Merging Results: Lock (Frozen) + Ram (New)")
        print("=" * 70)
        
        id_to_idx = {img_id: idx for idx, img_id in enumerate(previous_adv_ids)}
        final_images = np.zeros_like(self.images.numpy(), dtype=np.float32)
        
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
                    final_images[i] = self.images[i].numpy()
                    missing_count += 1
            elif img_id_int in ram_results:
                # RAM: Use new attack result
                result = ram_results[img_id_int]
                final_images[i] = result.adversarial.cpu().numpy()[0]
                ram_count += 1
            else:
                # Missing: use original image
                final_images[i] = self.images[i].numpy()
                missing_count += 1
        
        print(f"\nMerge Summary:")
        print(f"  Frozen (SUCCESS from API): {frozen_count} images")
        print(f"  RAM (new attack): {ram_count} images")
        print(f"  Missing (using original): {missing_count} images")
        print(f"  Total: {len(final_images)} images")
        
        output_path = self.output_dir / output_filename
        np.savez_compressed(
            output_path,
            images=final_images,
            image_ids=self.image_ids
        )
        
        print(f"\n✓ Saved merged submission to: {output_path}")
        return output_path
    
    def run(
        self,
        previous_submission: str,
        api_logits_json: str,
        epsilon: float = 10.0,
        kappa: float = 50.0,
        pgd_steps: int = 250,
        num_restarts: int = 10,
        output_filename: str = "submission_ram_v2.npz"
    ):
        """
        Run Lock & Ram V2 strategy.
        
        Args:
            previous_submission: Path to previous submission file
            api_logits_json: Path to API logits JSON (REQUIRED - uses API results, not local)
            epsilon: Fixed epsilon for RAM attack
            kappa: High kappa for RAM attack
            pgd_steps: PGD steps for RAM attack
            num_restarts: Restarts for RAM attack
            output_filename: Output filename
        """
        print("\n" + "=" * 70)
        print("LOCK & RAM STRATEGY V2")
        print("=" * 70)
        print(f"\nStep 1: LOCK - Loading previous submission")
        print(f"  File: {previous_submission}")
        
        # Load previous submission
        previous_adv_images, previous_adv_ids = self.load_previous_submission(previous_submission)
        
        # Identify SUCCESS/FAILED from API results (REQUIRED)
        print(f"\nStep 2: LOCK - Identifying SUCCESS/FAILED from API results")
        print(f"  API logits JSON: {api_logits_json}")
        success_ids, failed_ids = self.identify_success_failed_from_api(api_logits_json)
        
        print(f"\n✓ Locked {len(success_ids)} SUCCESS images (will be frozen)")
        print(f"✓ Targeting {len(failed_ids)} FAILED images for RAM attack")
        
        # Set up signal handler
        def signal_handler(signum, frame):
            self._signal_handler(signum, frame)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # RAM attack on failed images (multi-GPU parallelized)
        print(f"\nStep 3: RAM - Multi-GPU parallelized attack on API failures")
        ram_results = self.ram_attack_failed_parallel(
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
        print("LOCK & RAM V2 COMPLETE!")
        print("=" * 70)
        print(f"\nOutput saved to: {output_path}")
        print(f"\nNext steps:")
        print(f"1. Review results: python analyze.py {output_path} --mode local")
        print(f"2. Submit to API: python submit.py {output_path} --action submit")


def main():
    parser = argparse.ArgumentParser(
        description='Lock & Ram Strategy V2: Multi-GPU parallelized',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # I/O paths
    parser.add_argument('--previous-submission', type=str, required=True,
                       help='Path to previous submission file')
    parser.add_argument('--api-logits-json', type=str, required=True,
                       help='Path to API logits JSON (REQUIRED - uses API results to identify failures)')
    parser.add_argument('--dataset', type=str, default='../natural_images.pt',
                       help='Path to natural_images.pt')
    parser.add_argument('--output-dir', type=str, default='./output',
                       help='Output directory')
    parser.add_argument('--log-dir', type=str, default='./logs',
                       help='Log directory')
    parser.add_argument('--output-name', type=str, default='submission_ram_v2.npz',
                       help='Output filename')
    
    # RAM attack config (aggressive)
    parser.add_argument('--epsilon', type=float, default=10.0,
                       help='Fixed epsilon for RAM attack')
    parser.add_argument('--kappa', type=float, default=50.0,
                       help='High kappa for RAM attack')
    parser.add_argument('--pgd-steps', type=int, default=250,
                       help='PGD steps for RAM attack')
    parser.add_argument('--restarts', type=int, default=10,
                       help='Number of random restarts')
    parser.add_argument('--alpha-factor', type=float, default=2.5,
                       help='Alpha factor')
    parser.add_argument('--momentum', type=float, default=0.9,
                       help='MI-FGSM momentum')
    
    # Multi-GPU
    parser.add_argument('--num-gpus', type=int, default=2,
                       help='Number of GPUs to use')
    
    args = parser.parse_args()
    
    # Initialize solver
    solver = LockAndRamSolverV2(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        log_dir=args.log_dir,
        num_gpus=args.num_gpus
    )
    
    # Run Lock & Ram V2
    solver.run(
        previous_submission=args.previous_submission,
        api_logits_json=args.api_logits_json,
        epsilon=args.epsilon,
        kappa=args.kappa,
        pgd_steps=args.pgd_steps,
        num_restarts=args.restarts,
        output_filename=args.output_name
    )


if __name__ == '__main__':
    main()

