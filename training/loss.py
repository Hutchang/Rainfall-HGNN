import torch
import torch.nn.functional as F
from typing import Protocol, Any

def binary_rainfall_loss(logits, targets, threshold=0.5, pos_weight_val=None):
    """
    Computes Binary Cross-Entropy with Logits for rainfall classification. (BCE)
    
    Args:
        logits: The raw, unactivated outputs from the HGNNClassifier.
        targets: The continuous ground-truth rainfall values (mm/h).
        threshold: The mm/h cutoff defining a positive rain event.
        pos_weight_val: Float multiplier to penalize missed rain events harder.
    """
    # 1. Binarize the continuous targets on the fly
    binary_targets = (targets >= threshold).float()
    
    # 2. Format the positive weight for PyTorch
    pos_weight = None
    if pos_weight_val is not None:
        pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32, device=logits.device)
        
    # 3. Calculate BCE with Logits
    return F.binary_cross_entropy_with_logits(logits, binary_targets, pos_weight=pos_weight)


class LossFunction(Protocol):
    """A loss function.

    This is a protocol so it's fine to use a plain function which 'quacks like'
    this. This is just to document the interface.
    """
