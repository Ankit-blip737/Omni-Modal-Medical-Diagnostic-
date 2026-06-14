"""
Training Losses for Omni-Modal Diagnostic Framework
====================================================
Combines multiple loss functions for the 3-phase training strategy:
    - Focal Loss (handles class imbalance in medical datasets)
    - InfoNCE (contrastive visual-semantic alignment)
    - Combined multi-task loss with dynamic weighting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing severe class imbalance in medical datasets.
    
    Reduces the loss contribution from easy/well-classified examples and
    focuses training on hard, misclassified cases.
    
    FL(p_t) = -α_t (1 - p_t)^γ log(p_t)
    
    Args:
        alpha: Per-class weighting factor (tensor of shape [num_classes])
               If None, equal weighting is used.
        gamma: Focusing parameter (default: 2.0)
               γ=0 is equivalent to standard cross-entropy
               γ=2 works well in practice for medical imaging
        reduction: 'mean', 'sum', or 'none'
        multi_label: If True, treats each class independently (sigmoid + BCE)
    """

    def __init__(
        self,
        alpha: torch.Tensor = None,
        gamma: float = 2.0,
        reduction: str = "mean",
        multi_label: bool = True,
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.multi_label = multi_label
        
        if alpha is not None:
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C) — raw logits from classifier
            targets: (B, C) for multi-label or (B,) for single-label
            
        Returns:
            Focal loss scalar
        """
        if self.multi_label:
            return self._multi_label_focal(logits, targets)
        else:
            return self._single_label_focal(logits, targets)

    def _multi_label_focal(self, logits, targets):
        """Binary focal loss per class (for multi-label classification)."""
        probs = torch.sigmoid(logits)
        
        # Binary cross-entropy per element
        bce = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction="none")
        
        # Focal modulation
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        
        loss = focal_weight * bce
        
        if self.alpha is not None:
            loss = loss * self.alpha.unsqueeze(0)
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

    def _single_label_focal(self, logits, targets):
        """Multi-class focal loss (for single-label classification)."""
        probs = F.softmax(logits, dim=-1)
        
        # Gather probability of true class
        targets_one_hot = F.one_hot(targets, num_classes=logits.shape[-1]).float()
        p_t = (probs * targets_one_hot).sum(dim=-1)
        
        # Focal modulation
        focal_weight = (1 - p_t) ** self.gamma
        
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        loss = focal_weight * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            loss = alpha_t * loss
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class CombinedLoss(nn.Module):
    """
    Multi-task combined loss for the Omni-Modal Framework.
    
    ℒ_total = ℒ_cls + λ₁·ℒ_align + λ₂·ℒ_reg
    
    Args:
        num_classes: Number of diagnostic classes
        alpha: Per-class focal loss weights
        gamma: Focal loss focusing parameter
        lambda_align: Weight for contrastive alignment loss
        multi_label: Whether classification is multi-label
    """

    def __init__(
        self,
        num_classes: int = 14,
        alpha: torch.Tensor = None,
        gamma: float = 2.0,
        lambda_align: float = 0.1,
        multi_label: bool = True,
    ):
        super().__init__()
        self.lambda_align = lambda_align
        
        self.cls_loss = FocalLoss(
            alpha=alpha,
            gamma=gamma,
            multi_label=multi_label,
        )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        alignment_loss: torch.Tensor = None,
    ) -> dict:
        """
        Args:
            logits: (B, C) — classification logits
            targets: (B, C) or (B,) — ground truth labels
            alignment_loss: scalar — from VisualSemanticAlignmentModule
            
        Returns:
            Dictionary with:
                'total_loss': scalar — weighted sum of all losses
                'cls_loss': scalar — classification (focal) loss
                'align_loss': scalar — contrastive alignment loss
        """
        cls_loss = self.cls_loss(logits, targets)
        
        total_loss = cls_loss
        align_loss_val = torch.tensor(0.0, device=logits.device)
        
        if alignment_loss is not None:
            align_loss_val = alignment_loss
            total_loss = total_loss + self.lambda_align * align_loss_val
        
        return {
            "total_loss": total_loss,
            "cls_loss": cls_loss,
            "align_loss": align_loss_val,
        }


# --- Quick test ---
if __name__ == "__main__":
    B, C = 8, 14
    
    # Multi-label test
    logits = torch.randn(B, C)
    targets = torch.randint(0, 2, (B, C)).float()
    
    focal = FocalLoss(gamma=2.0, multi_label=True)
    loss = focal(logits, targets)
    print(f"Multi-label Focal Loss: {loss.item():.4f}")
    
    # Combined loss test
    combined = CombinedLoss(num_classes=C, lambda_align=0.1)
    alignment = torch.tensor(2.5)
    result = combined(logits, targets, alignment)
    print(f"Total Loss:     {result['total_loss'].item():.4f}")
    print(f"  Cls Loss:     {result['cls_loss'].item():.4f}")
    print(f"  Align Loss:   {result['align_loss'].item():.4f}")
    print(f"  λ·Align:      {0.1 * result['align_loss'].item():.4f}")
