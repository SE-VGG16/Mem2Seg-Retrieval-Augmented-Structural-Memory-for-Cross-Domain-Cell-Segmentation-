import torch
import torch.nn.functional as F

def dice_loss(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    num = 2 * (probs * targets).sum(dim=(2,3))
    den = (probs + targets).sum(dim=(2,3)) + eps
    dice = 1 - (num / den)
    return dice.mean()

def bce_loss(logits, targets):
    return F.binary_cross_entropy_with_logits(logits, targets)

def seg_loss(logits, targets):
    return dice_loss(logits, targets) + bce_loss(logits, targets)

def info_nce_loss(z1, z2, temperature=0.2):
    """
    Simple InfoNCE for batch-wise contrastive learning (SimCLR style).
    z1,z2: [B,D] normalized or not.
    """
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    B = z1.size(0)
    logits = (z1 @ z2.t()) / temperature  # [B,B]
    labels = torch.arange(B, device=z1.device)
    return F.cross_entropy(logits, labels)