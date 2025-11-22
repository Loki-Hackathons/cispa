"""
Helper functions for generating adversarial examples.
These are example implementations - you should modify and improve them.
"""

import torch
import numpy as np
from typing import Tuple, Optional


def query_api_logits(images: np.ndarray, api_client) -> np.ndarray:
    """
    Query the API to get logits for given images.
    
    Args:
        images: numpy array of shape (N, 3, 28, 28) with values in [0, 1]
        api_client: object with query_logits method
        
    Returns:
        logits: numpy array of shape (N, num_classes)
    """
    # Save images temporarily
    temp_path = "temp_query.npz"
    np.savez_compressed(temp_path, images=images.astype(np.float32))
    
    # Query API (you'll need to implement this based on your API client)
    logits = api_client.query_logits(temp_path)
    
    return logits


def fgsm_attack(
    image: torch.Tensor,
    epsilon: float = 0.1,
    logits_fn=None,
    target_label: Optional[int] = None
) -> torch.Tensor:
    """
    Fast Gradient Sign Method (FGSM) attack.
    Single-step attack - fast but not optimal.
    
    Args:
        image: original image tensor, shape (3, 28, 28), values in [0, 1]
        epsilon: perturbation magnitude (typically 0.01 to 0.3)
        logits_fn: function that takes image tensor and returns logits
        target_label: if None, untargeted attack (any misclassification)
        
    Returns:
        adversarial_image: perturbed image tensor
    """
    image = image.clone().detach().requires_grad_(True)
    
    # Get logits
    logits = logits_fn(image.unsqueeze(0))
    
    if target_label is not None:
        # Targeted attack: maximize target class probability
        loss = -logits[0, target_label]
    else:
        # Untargeted attack: minimize true class probability
        true_label = logits.argmax(dim=1)[0]
        loss = -logits[0, true_label]
    
    # Compute gradient
    loss.backward()
    gradient = image.grad.data
    
    # Apply perturbation
    perturbation = epsilon * gradient.sign()
    adversarial = image + perturbation
    
    # Clip to valid range [0, 1]
    adversarial = torch.clamp(adversarial, 0, 1)
    
    return adversarial.detach()


def pgd_attack(
    image: torch.Tensor,
    epsilon: float = 0.1,
    alpha: float = 0.01,
    num_iterations: int = 40,
    logits_fn=None,
    target_label: Optional[int] = None
) -> torch.Tensor:
    """
    Projected Gradient Descent (PGD) attack.
    Iterative attack - better results than FGSM.
    
    Args:
        image: original image tensor, shape (3, 28, 28), values in [0, 1]
        epsilon: maximum perturbation magnitude (L∞ bound)
        alpha: step size per iteration
        num_iterations: number of PGD steps
        logits_fn: function that takes image tensor and returns logits
        target_label: if None, untargeted attack
        
    Returns:
        adversarial_image: perturbed image tensor
    """
    original = image.clone().detach()
    adversarial = image.clone().detach().requires_grad_(True)
    
    for _ in range(num_iterations):
        # Get logits
        logits = logits_fn(adversarial.unsqueeze(0))
        
        if target_label is not None:
            loss = -logits[0, target_label]
        else:
            true_label = logits.argmax(dim=1)[0]
            loss = -logits[0, true_label]
        
        # Compute gradient
        loss.backward()
        gradient = adversarial.grad.data
        
        # Update adversarial example
        adversarial = adversarial + alpha * gradient.sign()
        
        # Project back to epsilon-ball around original
        perturbation = adversarial - original
        perturbation = torch.clamp(perturbation, -epsilon, epsilon)
        adversarial = original + perturbation
        
        # Clip to valid range [0, 1]
        adversarial = torch.clamp(adversarial, 0, 1)
        
        # Reset gradients for next iteration
        adversarial = adversarial.detach().requires_grad_(True)
    
    return adversarial.detach()


def compute_l2_distance(original: torch.Tensor, adversarial: torch.Tensor) -> float:
    """
    Compute L2 distance between original and adversarial image.
    
    Args:
        original: original image tensor
        adversarial: adversarial image tensor
        
    Returns:
        l2_distance: scalar L2 distance
    """
    diff = adversarial - original
    l2_norm = torch.norm(diff, p=2)
    return l2_norm.item()


def is_misclassified(logits: np.ndarray, true_label: int) -> bool:
    """
    Check if the predicted label differs from true label.
    
    Args:
        logits: logits array of shape (num_classes,)
        true_label: true label index
        
    Returns:
        True if misclassified, False otherwise
    """
    predicted_label = np.argmax(logits)
    return predicted_label != true_label


def generate_adversarial_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    attack_fn,
    attack_kwargs: dict,
    device: str = "cuda"
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Generate adversarial examples for a batch of images.
    
    Args:
        images: batch of images, shape (N, 3, 28, 28)
        labels: true labels, shape (N,)
        attack_fn: attack function to use (e.g., pgd_attack)
        attack_kwargs: keyword arguments for attack function
        device: device to run on ('cuda' or 'cpu')
        
    Returns:
        adversarial_images: tensor of adversarial examples
        success_mask: boolean array indicating which attacks succeeded
    """
    images = images.to(device)
    adversarial_images = []
    success_mask = []
    
    for i in range(len(images)):
        image = images[i]
        label = labels[i].item()
        
        # Generate adversarial example
        adversarial = attack_fn(image, **attack_kwargs)
        adversarial_images.append(adversarial)
        
        # Check if successful (you'll need to query API for this)
        # For now, we'll assume all succeed - you need to verify via API
        success_mask.append(True)
    
    adversarial_batch = torch.stack(adversarial_images)
    return adversarial_batch, np.array(success_mask)


def save_submission(
    adversarial_images: torch.Tensor,
    image_ids: torch.Tensor,
    filepath: str = "submission.npz"
):
    """
    Save adversarial examples in the correct submission format.
    
    Args:
        adversarial_images: tensor of shape (100, 3, 28, 28), values in [0, 1]
        image_ids: tensor of shape (100,) with image IDs
        filepath: path to save .npz file
    """
    # Convert to numpy
    images_np = adversarial_images.detach().cpu().numpy()
    
    # Ensure float32 dtype
    images_np = images_np.astype(np.float32)
    
    # Ensure values in [0, 1]
    images_np = np.clip(images_np, 0, 1)
    
    # Check for NaN/Inf
    if not np.isfinite(images_np).all():
        raise ValueError("Images contain NaN or Inf values!")
    
    # Save (note: submission only needs 'images', not 'image_ids')
    np.savez_compressed(filepath, images=images_np)
    print(f"Saved submission to {filepath}")
    print(f"Shape: {images_np.shape}, Dtype: {images_np.dtype}")
    print(f"Min: {images_np.min():.4f}, Max: {images_np.max():.4f}")

