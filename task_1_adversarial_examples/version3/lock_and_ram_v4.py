"""
Lock & Ram Strategy V4: Adaptive Epsilon per Image

Improvements over V3:
- Adapts epsilon per image based on previous API results
- Iteratively refines epsilon to minimize L2 distance
- Supports multiple API iterations with intelligent epsilon adjustment
- Logging tracks epsilon changes and L2 improvements per image
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


class LockAndRamSolverV4:
    """
    Lock & Ram Solver V4: Adaptive epsilon per image.
    
    Uses API results to adapt epsilon per image for optimal L2 distance.
    Supports iterative refinement across multiple API calls.
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
        print("Lock & Ram Solver V4 - Adaptive Epsilon Initialization")
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
        self.checkpoint_path = self.log_dir / "lock_and_ram_v4_checkpoint.json"
        self.epsilon_mapping_path = self.log_dir / "lock_and_ram_v4_epsilon_mapping.json"
        
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
    
    def compute_adaptive_epsilons(
        self,
        analysis_json_path: str,
        previous_epsilon_mapping: Optional[Dict[int, float]] = None
    ) -> Dict[int, float]:
        """
        Compute adaptive epsilon per image based on API analysis results.
        
        Strategy:
        - Images with L2 ~0.06 (epsilon ~1) → epsilon = 0.8
        - Images with L2 ~0.16 (epsilon ~8) → epsilon = 6
        - Images with L2 ~0.20 (epsilon ~10) → epsilon = 8
        - Images with L2 ~0.47 (epsilon ~30) → epsilon = 20
        
        If previous epsilon mapping exists, adjust based on results:
        - If image worsened (L2 increased or became FAILED), increase epsilon slightly
        - If image improved (L2 decreased), keep or slightly decrease epsilon
        
        Args:
            analysis_json_path: Path to analysis_api JSON file
            previous_epsilon_mapping: Previous epsilon mapping (for iteration > 1)
        
        Returns:
            Dictionary mapping image_id -> epsilon
        """
        print(f"\nComputing adaptive epsilons from: {analysis_json_path}")
        
        if not os.path.exists(analysis_json_path):
            raise FileNotFoundError(f"Analysis JSON not found: {analysis_json_path}")
        
        with open(analysis_json_path, 'r') as f:
            analysis_data = json.load(f)
        
        per_image = analysis_data.get('per_image', [])
        if not per_image:
            raise ValueError("No 'per_image' data found in analysis JSON")
        
        # Compute L2 norm factor
        C, H, W = self.images.shape[1], self.images.shape[2], self.images.shape[3]
        l2_norm_factor = np.sqrt(C * H * W)
        
        epsilon_mapping = {}
        epsilon_stats = {
            '0.8': 0, '6': 0, '8': 0, '20': 0
        }
        
        print("\nAdaptive Epsilon Mapping:")
        print(f"{'ID':>3} | {'L2 Norm':>8} | {'Status':>7} | {'Old Eps':>8} | {'New Eps':>8} | {'Change':>8}")
        print("-" * 70)
        
        for item in per_image:
            img_id = item['image_id']
            l2_norm = item['l2_normalized']
            is_success = item.get('misclassified', True)
            
            # Get previous epsilon if exists
            prev_epsilon = previous_epsilon_mapping.get(img_id) if previous_epsilon_mapping else None
            
            # Determine base epsilon based on L2 distance
            if l2_norm < 0.08:  # ~epsilon 1
                base_epsilon = 0.8
            elif l2_norm < 0.18:  # ~epsilon 8
                base_epsilon = 6.0
            elif l2_norm < 0.25:  # ~epsilon 10
                base_epsilon = 8.0
            else:  # ~epsilon 30
                base_epsilon = 20.0
            
            # If previous epsilon exists, adjust based on results
            if prev_epsilon is not None:
                if not is_success:
                    # Image became FAILED - increase epsilon significantly
                    new_epsilon = min(prev_epsilon * 1.5, 30.0)
                    change = f"+{new_epsilon - prev_epsilon:.1f}"
                else:
                    # Image is SUCCESS - try to optimize epsilon
                    # If current epsilon was too high (L2 is high), try lower
                    # If current epsilon was too low (might be borderline), keep similar
                    if l2_norm > 0.4:  # Very high L2 - can try lower epsilon
                        new_epsilon = max(prev_epsilon * 0.9, base_epsilon * 0.8)
                        change = f"{new_epsilon - prev_epsilon:.1f}"
                    elif l2_norm < 0.1:  # Very low L2 - can try even lower
                        new_epsilon = max(prev_epsilon * 0.95, base_epsilon * 0.9)
                        change = f"{new_epsilon - prev_epsilon:.1f}"
                    else:
                        # Moderate L2 - fine-tune toward base
                        new_epsilon = (prev_epsilon * 0.7 + base_epsilon * 0.3)
                        change = f"{new_epsilon - prev_epsilon:+.1f}"
            else:
                # First iteration - use base epsilon
                new_epsilon = base_epsilon
                change = "new"
            
            epsilon_mapping[img_id] = float(new_epsilon)
            
            # Track statistics
            eps_key = str(int(new_epsilon)) if new_epsilon >= 1 else f"{new_epsilon:.1f}"
            if eps_key in epsilon_stats:
                epsilon_stats[eps_key] += 1
            
            status_str = "SUCCESS" if is_success else "FAILED"
            prev_str = f"{prev_epsilon:.1f}" if prev_epsilon else "N/A"
            print(f"{img_id:3d} | {l2_norm:8.4f} | {status_str:>7} | {prev_str:>8} | {new_epsilon:8.1f} | {change:>8}")
        
        print("-" * 70)
        print(f"\nEpsilon Distribution:")
        for eps, count in epsilon_stats.items():
            if count > 0:
                print(f"  Epsilon {eps}: {count} images")
        
        return epsilon_mapping
    
    def load_epsilon_mapping(self) -> Optional[Dict[int, float]]:
        """Load previous epsilon mapping if exists."""
        if not self.epsilon_mapping_path.exists():
            return None
        
        try:
            with open(self.epsilon_mapping_path, 'r') as f:
                data = json.load(f)
                return {int(k): float(v) for k, v in data.items()}
        except Exception as e:
            print(f"⚠ Warning: Could not load epsilon mapping: {e}")
            return None
    
    def save_epsilon_mapping(self, epsilon_mapping: Dict[int, float]):
        """Save epsilon mapping for next iteration."""
        with open(self.epsilon_mapping_path, 'w') as f:
            json.dump(epsilon_mapping, f, indent=2)
        print(f"\n✓ Saved epsilon mapping to: {self.epsilon_mapping_path}")
    
    def identify_success_failed_from_api(
        self,
        api_logits_json_path: str
    ) -> Tuple[Set[int], Set[int]]:
        """Identify SUCCESS and FAILED image IDs from API logits JSON."""
        print(f"\nIdentifying SUCCESS/FAILED from API results: {api_logits_json_path}")
        
        if not os.path.exists(api_logits_json_path):
            raise FileNotFoundError(f"API logits JSON not found: {api_logits_json_path}")
        
        with open(api_logits_json_path, 'r') as f:
            logits_data = json.load(f)
        
        dataset = torch.load(self.dataset_path, weights_only=False)
        true_labels = dataset["labels"].numpy()
        
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
        
        return success_ids, failed_ids
    
    def _worker_thread(
        self,
        gpu_id: int,
        image_list: List[Tuple[int, int]],
        result_dict: Dict[int, AttackResult],
        epsilon_mapping: Dict[int, float],
        config_dict: dict,
        lock
    ):
        """Worker thread for GPU parallelization with per-image epsilon."""
        torch.cuda.set_device(gpu_id)
        device = f'cuda:{gpu_id}'
        
        data = torch.load(self.dataset_path, weights_only=False)
        images = data["images"]
        labels = data["labels"]
        
        ensemble = HybridEnsemble(device=device, fast_mode=True)
        config_dict_clean = {k: v for k, v in config_dict.items() if k != 'epsilon'}
        config = AttackConfig(**config_dict_clean)
        attacker = BSPGD(ensemble, config, device=device)
        
        print(f"[GPU {gpu_id}] Worker started - {len(image_list)} images")
        
        for img_id, img_idx in image_list:
            try:
                img = images[img_idx:img_idx+1].to(device)
                label = labels[img_idx:img_idx+1].to(device)
                
                # Get epsilon for this specific image
                epsilon = epsilon_mapping.get(img_id, config_dict['epsilon_min'])
                kappa = config_dict['kappa']
                
                result = attacker.attack_single_epsilon(
                    img, label, img_id,
                    epsilon,
                    kappa
                )
                
                result.adversarial = result.adversarial.cpu()
                
                with lock:
                    result_dict[img_id] = result
                
                C, H, W = self.images.shape[1], self.images.shape[2], self.images.shape[3]
                l2_norm_factor = np.sqrt(C * H * W)
                l2_normalized = result.l2_distance / l2_norm_factor
                
                status = "✓ SUCCESS" if result.success else "✗ FAILED"
                print(f"[GPU {gpu_id}] Image ID {img_id:3d}: {status} | " +
                      f"Eps: {epsilon:.1f} | " +
                      f"L2: {result.l2_distance:.4f} (norm: {l2_normalized:.4f}) | " +
                      f"Margin: {result.confidence_margin:+.2f}")
                
            except Exception as e:
                print(f"[GPU {gpu_id}] Error processing image {img_id}: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"[GPU {gpu_id}] Worker finished")
    
    def ram_attack_all_parallel(
        self,
        target_ids: Set[int],
        epsilon_mapping: Dict[int, float],
        kappa: float = 50.0,
        pgd_steps: int = 80,
        num_restarts: int = 5,
        alpha_factor: float = 2.5,
        momentum: float = 0.9
    ) -> Dict[int, AttackResult]:
        """RAM attack with adaptive epsilon per image - attacks ALL target images."""
        print("\n" + "=" * 70)
        print("RAM ATTACK V4: Adaptive Epsilon per Image (ALL Images)")
        print("=" * 70)
        print(f"\nConfiguration:")
        print(f"  Target images: {len(target_ids)} (ALL images for L2 optimization)")
        print(f"  GPUs: {self.num_gpus}")
        print(f"  Epsilon: Adaptive per image (see mapping above)")
        print(f"  Kappa: {kappa}")
        print(f"  PGD steps: {pgd_steps}")
        print(f"  Restarts: {num_restarts}")
        print(f"  Alpha factor: {alpha_factor}")
        print(f"  Momentum: {momentum}")
        print(f"  Input Diversity: ENABLED")
        
        checkpoint_results = self._load_checkpoint()
        if checkpoint_results is None:
            checkpoint_results = {}
        
        if checkpoint_results:
            print(f"✓ Loaded {len(checkpoint_results)} results from checkpoint")
            target_list = [img_id for img_id in sorted(target_ids) if img_id not in checkpoint_results]
            results = checkpoint_results.copy()
        else:
            results = {}
            target_list = sorted(target_ids)
        
        if not target_list:
            print("✓ All images already processed!")
            return results
        
        print(f"  Remaining to process: {len(target_list)} images")
        
        # Filter epsilon mapping to target images
        target_epsilon_mapping = {img_id: epsilon_mapping.get(img_id, 10.0) 
                                  for img_id in target_list}
        
        # Create attack config dict
        config_dict = {
            'epsilon_min': min(target_epsilon_mapping.values()) if target_epsilon_mapping else 10.0,
            'epsilon_max': max(target_epsilon_mapping.values()) if target_epsilon_mapping else 10.0,
            'binary_search_steps': 1,
            'pgd_steps': pgd_steps,
            'num_restarts': num_restarts,
            'alpha_factor': alpha_factor,
            'kappa': kappa,
            'use_input_diversity': True,
            'momentum': momentum,
        }
        
        # Prepare image lists for each GPU
        image_tasks = []
        for img_id in target_list:
            img_idx = np.where(self.image_ids == img_id)[0]
            if len(img_idx) > 0:
                image_tasks.append((img_id, img_idx[0]))
        
        tasks_per_gpu = len(image_tasks) // self.num_gpus
        gpu_tasks = []
        for gpu_id in range(self.num_gpus):
            start_idx = gpu_id * tasks_per_gpu
            if gpu_id == self.num_gpus - 1:
                end_idx = len(image_tasks)
            else:
                end_idx = (gpu_id + 1) * tasks_per_gpu
            gpu_tasks.append(image_tasks[start_idx:end_idx])
            print(f"  GPU {gpu_id}: {len(gpu_tasks[gpu_id])} images")
        
        import threading
        result_lock = threading.Lock()
        worker_results = {}
        
        threads = []
        start_time = time.time()
        
        for gpu_id in range(self.num_gpus):
            if len(gpu_tasks[gpu_id]) > 0:
                t = Thread(
                    target=self._worker_thread,
                    args=(gpu_id, gpu_tasks[gpu_id], worker_results, target_epsilon_mapping, config_dict, result_lock)
                )
                t.start()
                threads.append(t)
        
        try:
            while any(t.is_alive() for t in threads):
                if self.interrupted:
                    print(f"\n⚠ Interrupted. Saving checkpoint...")
                    with result_lock:
                        results.update(worker_results)
                    self._save_checkpoint(results)
                    break
                
                time.sleep(5)
                
                with result_lock:
                    results.update(worker_results)
                    if len(results) > len(checkpoint_results):
                        self._save_checkpoint(results)
                        completed = len(results) - len(checkpoint_results)
                        print(f"\n💾 Progress: {completed}/{len(target_list)} images processed")
        
        except KeyboardInterrupt:
            print(f"\n\n⚠ KeyboardInterrupt caught. Saving checkpoint...")
            with result_lock:
                results.update(worker_results)
            self._save_checkpoint(results)
            raise
        
        for t in threads:
            t.join()
        
        # Final update
        with result_lock:
            results.update(worker_results)
        
        elapsed = time.time() - start_time
        completed = len(results) - len(checkpoint_results) if checkpoint_results else len(results)
        
        print(f"\n✓ Completed {completed} images in {elapsed:.1f}s")
        
        if self.checkpoint_path.exists() and completed == len(target_list):
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
    
    def merge_and_save(
        self,
        previous_adv_images: np.ndarray,
        previous_adv_ids: np.ndarray,
        ram_results: Dict[int, AttackResult],
        output_filename: str = "submission_ram_v4.npz"
    ) -> Path:
        """Merge new RAM attack results (all images are re-attacked for L2 optimization)."""
        print("\n" + "=" * 70)
        print("Merging Results: All Images Re-attacked")
        print("=" * 70)
        
        id_to_idx = {img_id: idx for idx, img_id in enumerate(previous_adv_ids)}
        final_images = np.zeros_like(self.images.numpy(), dtype=np.float32)
        
        ram_count = 0
        missing_count = 0
        
        for i, img_id in enumerate(self.image_ids):
            img_id_int = int(img_id)
            
            if img_id_int in ram_results:
                # Use new attack result
                result = ram_results[img_id_int]
                final_images[i] = result.adversarial.cpu().numpy()[0]
                ram_count += 1
            elif img_id_int in id_to_idx:
                # Fallback: use previous submission if no new result
                prev_idx = id_to_idx[img_id_int]
                final_images[i] = previous_adv_images[prev_idx]
                missing_count += 1
            else:
                # Last resort: use original image
                final_images[i] = self.images[i].numpy()
                missing_count += 1
        
        print(f"\nMerge Summary:")
        print(f"  RAM (new attack): {ram_count} images")
        print(f"  Fallback (previous): {missing_count} images")
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
        analysis_json: str,
        kappa: float = 50.0,
        pgd_steps: int = 80,
        num_restarts: int = 5,
        output_filename: str = "submission_ram_v4.npz"
    ):
        """Run Lock & Ram V4 strategy with adaptive epsilon - attacks ALL images."""
        print("\n" + "=" * 70)
        print("LOCK & RAM STRATEGY V4: Adaptive Epsilon (ALL Images)")
        print("=" * 70)
        
        # Load previous submission
        print(f"\nStep 1: Loading previous submission")
        print(f"  File: {previous_submission}")
        previous_adv_images, previous_adv_ids = self.load_previous_submission(previous_submission)
        
        # Load previous epsilon mapping if exists
        previous_epsilon_mapping = self.load_epsilon_mapping()
        
        # Compute adaptive epsilons for ALL images
        print(f"\nStep 2: ADAPTIVE EPSILON - Computing per-image epsilon")
        print(f"  Analysis JSON: {analysis_json}")
        epsilon_mapping = self.compute_adaptive_epsilons(analysis_json, previous_epsilon_mapping)
        
        # Save epsilon mapping for next iteration
        self.save_epsilon_mapping(epsilon_mapping)
        
        # Target ALL images for L2 optimization (all are SUCCESS, we just want lower L2)
        all_image_ids = set(int(img_id) for img_id in self.image_ids)
        print(f"\n✓ Targeting ALL {len(all_image_ids)} images for L2 optimization")
        
        # Set up signal handler
        def signal_handler(signum, frame):
            self._signal_handler(signum, frame)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # RAM attack on ALL images with adaptive epsilon
        print(f"\nStep 3: RAM - Multi-GPU parallelized attack with adaptive epsilon (ALL images)")
        ram_results = self.ram_attack_all_parallel(
            all_image_ids,
            epsilon_mapping,
            kappa=kappa,
            pgd_steps=pgd_steps,
            num_restarts=num_restarts
        )
        
        # Merge and save
        print(f"\nStep 4: MERGE - Saving all new attack results")
        output_path = self.merge_and_save(
            previous_adv_images,
            previous_adv_ids,
            ram_results,
            output_filename
        )
        
        print("\n" + "=" * 70)
        print("LOCK & RAM V4 COMPLETE!")
        print("=" * 70)
        print(f"\nOutput saved to: {output_path}")
        print(f"\nNext steps:")
        print(f"1. Analyze with API: python analyze.py {output_path} --mode api")
        print(f"2. Review epsilon mapping: cat {self.epsilon_mapping_path}")
        print(f"3. For next iteration, use the new analysis JSON")


def main():
    parser = argparse.ArgumentParser(
        description='Lock & Ram Strategy V4: Adaptive Epsilon per Image',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--previous-submission', type=str, required=True,
                       help='Path to previous submission file')
    parser.add_argument('--analysis-json', type=str, required=True,
                       help='Path to analysis_api JSON (contains L2 distances)')
    parser.add_argument('--dataset', type=str, default='../natural_images.pt',
                       help='Path to natural_images.pt')
    parser.add_argument('--output-dir', type=str, default='./output',
                       help='Output directory')
    parser.add_argument('--log-dir', type=str, default='./logs',
                       help='Log directory')
    parser.add_argument('--output-name', type=str, default='submission_ram_v4.npz',
                       help='Output filename')
    parser.add_argument('--kappa', type=float, default=50.0,
                       help='Kappa threshold')
    parser.add_argument('--pgd-steps', type=int, default=80,
                       help='PGD steps')
    parser.add_argument('--restarts', type=int, default=5,
                       help='Number of restarts')
    parser.add_argument('--num-gpus', type=int, default=2,
                       help='Number of GPUs')
    
    args = parser.parse_args()
    
    solver = LockAndRamSolverV4(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        log_dir=args.log_dir,
        num_gpus=args.num_gpus
    )
    
    solver.run(
        previous_submission=args.previous_submission,
        analysis_json=args.analysis_json,
        kappa=args.kappa,
        pgd_steps=args.pgd_steps,
        num_restarts=args.restarts,
        output_filename=args.output_name
    )


if __name__ == '__main__':
    main()

