"""
Binary Search PGD (BS-PGD) with Random Restarts

Core Algorithm:
1. Binary search on epsilon per image to find minimal perturbation
2. For each epsilon candidate, run multiple random restarts in parallel
3. Track best adversarial example that satisfies success condition with minimum L2
4. Adaptive step size: alpha = 2.5 * epsilon / num_steps

Success Condition:
    logit_max_wrong - logit_true > kappa_i

Where kappa_i is a per-image confidence margin that will be calibrated 
in Phase 2 based on API feedback.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class AttackConfig:
    """Configuration for BS-PGD attack."""
    epsilon_min: float = 0.5
    epsilon_max: float = 12.0
    binary_search_steps: int = 8
    pgd_steps: int = 150
    num_restarts: int = 15
    alpha_factor: float = 2.5  # alpha = alpha_factor * epsilon / pgd_steps
    kappa: float = 0.0  # Confidence margin (will be per-image in practice)
    use_input_diversity: bool = True
    momentum: float = 0.9  # MI-FGSM momentum
    
    def get_alpha(self, epsilon: float) -> float:
        """Compute adaptive step size."""
        return self.alpha_factor * epsilon / self.pgd_steps


@dataclass
class AttackResult:
    """Result of attacking a single image."""
    image_id: int
    adversarial: torch.Tensor  # Best adversarial example found
    l2_distance: float
    epsilon_used: float
    kappa_used: float
    success: bool  # Local success (against surrogate ensemble)
    confidence_margin: float  # logit_max_wrong - logit_true
    num_restarts_succeeded: int  # How many restarts found valid attacks
    binary_search_path: list  # [(epsilon, success, l2)] for debugging


class BSPGD:
    """
    Binary Search PGD Attack with Random Restarts.
    
    This is the core of Phase 1: finding minimal perturbations that fool
    the local surrogate ensemble.
    """
    
    def __init__(self, ensemble, config: AttackConfig, device='cuda'):
        self.ensemble = ensemble
        self.config = config
        self.device = device
    
    def check_success(self, x: torch.Tensor, label: int, kappa: float) -> Tuple[bool, float]:
        """
        Check if adversarial example satisfies success condition.
        
        Success: logit_max_wrong - logit_true > kappa
        
        Returns:
            (success, margin) where margin = logit_max_wrong - logit_true
        """
        with torch.no_grad():
            logits = self.ensemble.predict(x)  # (1, num_classes)
            logits = logits[0]  # (num_classes,)
            
            true_logit = logits[label].item()
            
            # Mask out true class
            logits_masked = logits.clone()
            logits_masked[label] = -float('inf')
            max_wrong_logit = logits_masked.max().item()
            
            margin = max_wrong_logit - true_logit
            success = margin > kappa
            
            return success, margin
    
    def pgd_single_restart(
        self, 
        x_orig: torch.Tensor, 
        label: torch.Tensor,
        epsilon: float,
        alpha: float,
        random_start: bool = True
    ) -> Tuple[torch.Tensor, float, bool, float]:
        """
        Single PGD run with momentum.
        
        Returns:
            (best_adv, best_l2, success, best_margin)
        """
        # Random initialization
        if random_start:
            delta = torch.randn_like(x_orig) * 0.001
            delta = self._project_l2(delta, epsilon)
            x_adv = torch.clamp(x_orig + delta, 0, 1).detach()
        else:
            x_adv = x_orig.clone().detach()
        
        momentum_grad = torch.zeros_like(x_adv)
        
        # Track best candidate during optimization
        best_adv = x_adv.clone()
        best_l2 = float('inf')
        best_margin = -float('inf')
        best_success = False
        
        for step in range(self.config.pgd_steps):
            x_adv.requires_grad = True
            
            # Forward pass with ensemble
            loss = self.ensemble.forward(
                x_adv, 
                label, 
                use_diversity=self.config.use_input_diversity
            )
            
            # Backward pass
            self.ensemble.zero_grad()
            loss.backward()
            
            # MI-FGSM: Momentum gradient
            grad = x_adv.grad.data
            grad_norm = grad.view(grad.shape[0], -1).norm(p=1, dim=1, keepdim=True)
            grad = grad / (grad_norm.view(-1, 1, 1, 1) + 1e-8)
            
            momentum_grad = self.config.momentum * momentum_grad + grad
            
            # Update
            x_adv = x_adv.detach() + alpha * momentum_grad.sign()
            
            # Project to L2 ball
            delta = x_adv - x_orig
            delta = self._project_l2(delta, epsilon)
            x_adv = torch.clamp(x_orig + delta, 0, 1).detach()
            
            # Check if current iterate is best
            label_int = label[0].item()
            success, margin = self.check_success(x_adv, label_int, self.config.kappa)
            current_l2 = torch.norm((x_adv - x_orig).view(-1), p=2).item()
            
            # Update best if:
            # 1. Current is successful and has lower L2 than previous best
            # 2. No previous success but current has better margin
            if success:
                if not best_success or current_l2 < best_l2:
                    best_adv = x_adv.clone()
                    best_l2 = current_l2
                    best_margin = margin
                    best_success = True
            elif not best_success and margin > best_margin:
                best_adv = x_adv.clone()
                best_l2 = current_l2
                best_margin = margin
        
        return best_adv, best_l2, best_success, best_margin
    
    def pgd_multi_restart(
        self,
        x_orig: torch.Tensor,
        label: torch.Tensor,
        epsilon: float,
        alpha: float
    ) -> Tuple[torch.Tensor, float, bool, float, int]:
        """
        Run PGD with multiple random restarts and return the best result.
        
        Returns:
            (best_adv, best_l2, success, best_margin, num_succeeded)
        """
        best_adv = x_orig.clone()
        best_l2 = float('inf')
        best_margin = -float('inf')
        global_success = False
        num_succeeded = 0
        
        for restart_idx in range(self.config.num_restarts):
            adv, l2, success, margin = self.pgd_single_restart(
                x_orig, label, epsilon, alpha, random_start=(restart_idx > 0)
            )
            
            if success:
                num_succeeded += 1
                # Keep best successful attack (lowest L2)
                if not global_success or l2 < best_l2:
                    best_adv = adv
                    best_l2 = l2
                    best_margin = margin
                    global_success = True
            elif not global_success:
                # Keep best unsuccessful attempt (highest margin)
                if margin > best_margin:
                    best_adv = adv
                    best_l2 = l2
                    best_margin = margin
        
        return best_adv, best_l2, global_success, best_margin, num_succeeded
    
    def binary_search_epsilon(
        self,
        x_orig: torch.Tensor,
        label: torch.Tensor,
        image_id: int,
        kappa: float
    ) -> AttackResult:
        """
        Binary search on epsilon to find minimal perturbation.
        
        This is the main attack loop for a single image.
        """
        # Update config kappa for this image
        old_kappa = self.config.kappa
        self.config.kappa = kappa
        
        eps_min = self.config.epsilon_min
        eps_max = self.config.epsilon_max
        
        best_adv = x_orig.clone()
        best_l2 = float('inf')
        best_epsilon = eps_max
        best_margin = -float('inf')
        global_success = False
        
        bs_path = []  # Track binary search path
        
        for bs_step in range(self.config.binary_search_steps):
            eps_curr = (eps_min + eps_max) / 2.0
            alpha = self.config.get_alpha(eps_curr)
            
            # Run multi-restart PGD at this epsilon
            adv, l2, success, margin, num_succeeded = self.pgd_multi_restart(
                x_orig, label, eps_curr, alpha
            )
            
            bs_path.append((eps_curr, success, l2, margin))
            
            if success:
                # Found successful attack at this epsilon
                if l2 < best_l2:
                    best_adv = adv
                    best_l2 = l2
                    best_epsilon = eps_curr
                    best_margin = margin
                    global_success = True
                
                # Try smaller epsilon
                eps_max = eps_curr
            else:
                # Failed at this epsilon, try larger
                if not global_success and margin > best_margin:
                    best_adv = adv
                    best_l2 = l2
                    best_margin = margin
                
                eps_min = eps_curr
        
        # Final refinement: If we found success, try one more time at best_epsilon
        # with more steps for polish
        if global_success:
            alpha = self.config.get_alpha(best_epsilon)
            old_steps = self.config.pgd_steps
            self.config.pgd_steps = int(old_steps * 1.5)  # 50% more steps for refinement
            
            adv, l2, success, margin, _ = self.pgd_multi_restart(
                x_orig, label, best_epsilon, alpha
            )
            
            if success and l2 < best_l2:
                best_adv = adv
                best_l2 = l2
                best_margin = margin
            
            self.config.pgd_steps = old_steps  # Restore
        
        # Restore config
        self.config.kappa = old_kappa
        
        # Compute final success check
        final_success, final_margin = self.check_success(
            best_adv, label[0].item(), kappa
        )
        
        # Count how many restarts succeeded in the search
        total_succeeded = sum(1 for _, s, _, _ in bs_path if s)
        
        return AttackResult(
            image_id=image_id,
            adversarial=best_adv,
            l2_distance=best_l2,
            epsilon_used=best_epsilon,
            kappa_used=kappa,
            success=final_success,
            confidence_margin=final_margin,
            num_restarts_succeeded=total_succeeded,
            binary_search_path=bs_path
        )
    
    def attack_single_epsilon(
        self,
        x_orig: torch.Tensor,
        label: torch.Tensor,
        image_id: int,
        epsilon: float,
        kappa: float
    ) -> AttackResult:
        """
        Attack with a single fixed epsilon (no binary search).
        
        This is optimized for speed when epsilon is already fixed.
        Used by Lock & Ram strategy.
        
        Args:
            x_orig: Original image (1, 3, 28, 28)
            label: True label (1,)
            image_id: Image ID
            epsilon: Fixed epsilon value
            kappa: Confidence margin requirement
        
        Returns:
            AttackResult
        """
        # Update config kappa for this image
        old_kappa = self.config.kappa
        self.config.kappa = kappa
        
        alpha = self.config.get_alpha(epsilon)
        
        # Run multi-restart PGD at fixed epsilon
        best_adv, best_l2, success, best_margin, num_succeeded = self.pgd_multi_restart(
            x_orig, label, epsilon, alpha
        )
        
        # Restore config
        self.config.kappa = old_kappa
        
        return AttackResult(
            image_id=image_id,
            adversarial=best_adv,
            l2_distance=best_l2,
            epsilon_used=epsilon,
            kappa_used=kappa,
            success=success,
            confidence_margin=best_margin,
            num_restarts_succeeded=num_succeeded,
            binary_search_path=[(epsilon, success, best_l2, best_margin)]
        )
    
    def attack_batch(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        image_ids: np.ndarray,
        kappas: Optional[Dict[int, float]] = None
    ) -> list[AttackResult]:
        """
        Attack a batch of images.
        
        Args:
            images: (N, 3, 28, 28)
            labels: (N,)
            image_ids: (N,)
            kappas: Per-image confidence margins {image_id: kappa}
        
        Returns:
            List of AttackResult objects
        """
        if kappas is None:
            kappas = {img_id: 0.0 for img_id in image_ids}
        
        results = []
        
        for i in range(len(images)):
            img = images[i:i+1].to(self.device)
            label = labels[i:i+1].to(self.device)
            img_id = int(image_ids[i])
            kappa = kappas.get(img_id, 0.0)
            
            result = self.binary_search_epsilon(img, label, img_id, kappa)
            results.append(result)
        
        return results
    
    @staticmethod
    def _project_l2(delta: torch.Tensor, epsilon: float) -> torch.Tensor:
        """Project perturbation to L2 ball."""
        batch_size = delta.shape[0]
        delta_flat = delta.view(batch_size, -1)
        norm = delta_flat.norm(p=2, dim=1, keepdim=True)
        factor = torch.clamp(epsilon / (norm + 1e-8), max=1.0)
        delta_flat = delta_flat * factor
        return delta_flat.view_as(delta)


def compute_success_stats(results: list[AttackResult]) -> dict:
    """Compute statistics from attack results."""
    total = len(results)
    successes = [r for r in results if r.success]
    failures = [r for r in results if not r.success]
    
    stats = {
        'total_images': total,
        'successful': len(successes),
        'failed': len(failures),
        'success_rate': len(successes) / total * 100 if total > 0 else 0,
        'avg_l2_all': np.mean([r.l2_distance for r in results]),
        'avg_l2_success': np.mean([r.l2_distance for r in successes]) if successes else 0,
        'avg_l2_failed': np.mean([r.l2_distance for r in failures]) if failures else 0,
        'avg_margin_all': np.mean([r.confidence_margin for r in results]),
        'avg_margin_success': np.mean([r.confidence_margin for r in successes]) if successes else 0,
        'avg_margin_failed': np.mean([r.confidence_margin for r in failures]) if failures else 0,
        'avg_epsilon': np.mean([r.epsilon_used for r in results]),
        'min_l2': min([r.l2_distance for r in results]),
        'max_l2': max([r.l2_distance for r in results]),
    }
    
    return stats

