import torch
import torch.nn as nn
import torch.nn.functional as F
import random

def input_diversity_28(x):
    """
    Input Diversity tailored for 28x28 images.
    1. Pad to 32x32
    2. Randomly crop back to 28x28
    3. Add slight noise
    """
    # x is (B, 3, 28, 28)
    B, C, H, W = x.shape
    pad_size = 4 # 28 -> 32
    
    # Pad
    x_padded = F.pad(x, (2, 2, 2, 2), mode='reflect') # 32x32
    
    # Random Crop
    padded_size = 32
    h_start = random.randint(0, pad_size)
    w_start = random.randint(0, pad_size)
    
    x_div = x_padded[:, :, h_start:h_start+28, w_start:w_start+28]
    
    # Slight Gaussian noise (to smooth gradients)
    noise = torch.randn_like(x_div) * 0.002
    
    return torch.clamp(x_div + noise, 0, 1)

def cw_loss(logits, label, kappa=0, target=None):
    """
    Carlini-Wagner Loss (Margin Loss).
    Minimizes: max(0, logits[real] - max(logits[other]) + kappa)
    """
    # logits: (B, Classes)
    B = logits.shape[0]
    
    real_logits = torch.gather(logits, 1, label.view(-1, 1)).view(-1)
    
    logits_others = logits.clone()
    logits_others.scatter_(1, label.view(-1, 1), -1e9)
    max_other_logits, _ = torch.max(logits_others, dim=1)
    
    loss = torch.clamp(real_logits + kappa - max_other_logits, min=0)
    
    return loss.mean()

class BatchedBSPGD:
    """
    Optimized PGD with Parallel Restarts and Internal Best Candidate Selection.
    (Removed external Binary Search for speed).
    """
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.min_pixel = 0.0
        self.max_pixel = 1.0

    def attack(self, image, label, 
               num_restarts=20, 
               max_steps=100, 
               kappa=0, 
               epsilon_max=8.0): 
        """
        Runs PGD on a single image (replicated num_restarts times).
        Instead of BS, we run with a large epsilon and track the minimal successful perturbation
        found along the trajectory.
        """
        
        # 1. Expand inputs for batch processing (Restarts)
        x_batch = image.repeat(num_restarts, 1, 1, 1).to(self.device)
        y_batch = label.repeat(num_restarts).to(self.device)
        
        # 2. Run Single PGD with high epsilon
        # We set epsilon high enough to guarantee success, then minimize L2 internally.
        
        adv_batch, success_mask, l2_dists = self._run_pgd_batch(
            x_batch, y_batch, 
            epsilon=epsilon_max, 
            steps=max_steps, 
            kappa=kappa
        )
        
        # 3. Select Best Global Candidate
        if success_mask.any():
            valid_l2 = l2_dists[success_mask]
            min_l2_idx = torch.argmin(valid_l2)
            min_l2_val = valid_l2[min_l2_idx].item()
            
            successful_advs = adv_batch[success_mask]
            best_adv = successful_advs[min_l2_idx:min_l2_idx+1].detach().clone()
            
            return best_adv, min_l2_val, True
            
        else:
            # If failed, return original or best failed attempt (not implemented, return original)
            return image, float('inf'), False

    def _run_pgd_batch(self, x_batch, y_batch, epsilon, steps, kappa):
        """
        Standard PGD on a batch. Tracks BEST candidate found during steps.
        """
        x_adv = x_batch.clone().detach()
        x_adv.requires_grad = True
        
        # Initialize BEST tracking for this batch
        best_x_adv_batch = x_batch.clone().detach()
        best_l2_batch = torch.full((x_batch.shape[0],), float('inf'), device=self.device)
        success_mask_batch = torch.zeros(x_batch.shape[0], dtype=torch.bool, device=self.device)
        
        # Random start
        # We use full range random start to explore the space
        if epsilon > 0:
            noise = torch.zeros_like(x_adv).uniform_(-epsilon, epsilon) 
            # Renorm noise to be within L2 ball? No, PGD usually uniform in Linf or L2 ball.
            # Let's keep it simple: uniform per pixel, then clamped.
            x_adv = torch.clamp(x_adv + noise, self.min_pixel, self.max_pixel).detach()
            x_adv.requires_grad = True

        # Adapt Alpha
        alpha = (2.5 * epsilon) / steps
        
        for i in range(steps):
            x_div = input_diversity_28(x_adv)
            
            logits = self.model(x_div)
            loss = cw_loss(logits, y_batch, kappa=kappa)
            
            self.model.zero_grad()
            loss.backward()
            
            # 1. CHECK SUCCESS (Best Candidate Selection)
            with torch.no_grad():
                logits_check = self.model(x_adv)
                
                real_logits = torch.gather(logits_check, 1, y_batch.view(-1, 1)).view(-1)
                logits_others = logits_check.clone()
                logits_others.scatter_(1, y_batch.view(-1, 1), -1e9)
                max_other_logits, _ = torch.max(logits_others, dim=1)
                
                # Success condition
                is_success = (max_other_logits > (real_logits + kappa))
                
                # Calculate L2
                diff = (x_adv - x_batch).view(x_batch.shape[0], -1)
                current_l2 = torch.norm(diff, p=2, dim=1)
                
                # Update Best if: Success AND (Lower L2 OR previously failed)
                update_mask = is_success & (current_l2 < best_l2_batch)
                
                if update_mask.any():
                    best_l2_batch[update_mask] = current_l2[update_mask]
                    mask_broad = update_mask.view(-1, 1, 1, 1)
                    best_x_adv_batch = torch.where(mask_broad, x_adv.detach(), best_x_adv_batch)
                    success_mask_batch = success_mask_batch | update_mask 
            
            # 2. UPDATE STEP
            with torch.no_grad():
                grad = x_adv.grad
                # Normalize gradient for L2 attack? 
                # C&W / PGD-L2 usually normalizes grad by L2 norm.
                # Let's try standard Sign method first as it's robust, but L2 update is better for L2 min.
                # Update: x = x - alpha * grad / ||grad||_2
                
                # Let's stick to Sign for now (L_inf steep descent) but project to L2.
                # It's a heuristic that often works well for finding *some* adversarial example.
                x_adv.data += alpha * grad.sign()
                
                # Projection
                delta = x_adv.data - x_batch
                delta = delta.renorm(p=2, dim=0, maxnorm=epsilon)
                x_adv.data = torch.clamp(x_batch + delta, self.min_pixel, self.max_pixel)
                x_adv.grad.zero_()
                
        return best_x_adv_batch, success_mask_batch, best_l2_batch
