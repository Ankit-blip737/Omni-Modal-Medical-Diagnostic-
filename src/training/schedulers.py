"""
Learning Rate Schedulers
========================
Custom LR scheduler implementations for the 3-phase training strategy.

Includes:
    - WarmupCosineScheduler: Linear warmup → cosine annealing
    - WarmupLinearScheduler: Linear warmup → linear decay
    - get_scheduler: Factory function
"""

import math
import torch
from torch.optim.lr_scheduler import LambdaLR
from typing import Optional


class WarmupCosineScheduler(LambdaLR):
    """
    Linear warmup followed by cosine annealing decay.

    LR schedule:
        epoch < warmup_epochs:  lr = warmup_start_lr + (base_lr - warmup_start_lr) * epoch / warmup_epochs
        epoch >= warmup_epochs: lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(π * progress))

    Args:
        optimizer: PyTorch optimizer
        warmup_epochs: Number of warmup epochs
        total_epochs: Total training epochs
        min_lr_ratio: Minimum LR as fraction of base LR (default: 0.01)
        warmup_start_lr_ratio: Starting LR ratio during warmup (default: 0.01)
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr_ratio: float = 0.01,
        warmup_start_lr_ratio: float = 0.01,
        last_epoch: int = -1,
    ):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr_ratio = min_lr_ratio
        self.warmup_start_lr_ratio = warmup_start_lr_ratio

        super().__init__(optimizer, self._lr_lambda, last_epoch=last_epoch)

    def _lr_lambda(self, epoch: int) -> float:
        """Compute LR multiplier for given epoch."""
        if epoch < self.warmup_epochs:
            # Linear warmup
            alpha = epoch / max(1, self.warmup_epochs)
            return self.warmup_start_lr_ratio + (1.0 - self.warmup_start_lr_ratio) * alpha
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            return self.min_lr_ratio + 0.5 * (1.0 - self.min_lr_ratio) * (1 + math.cos(math.pi * progress))


class WarmupLinearScheduler(LambdaLR):
    """
    Linear warmup followed by linear decay.

    Args:
        optimizer: PyTorch optimizer
        warmup_epochs: Number of warmup epochs
        total_epochs: Total training epochs
        min_lr_ratio: Minimum LR as fraction of base LR
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr_ratio: float = 0.0,
        last_epoch: int = -1,
    ):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr_ratio = min_lr_ratio

        super().__init__(optimizer, self._lr_lambda, last_epoch=last_epoch)

    def _lr_lambda(self, epoch: int) -> float:
        if epoch < self.warmup_epochs:
            return epoch / max(1, self.warmup_epochs)
        else:
            progress = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            return max(self.min_lr_ratio, 1.0 - progress * (1.0 - self.min_lr_ratio))


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = "cosine",
    warmup_epochs: int = 10,
    total_epochs: int = 100,
    min_lr_ratio: float = 0.01,
) -> LambdaLR:
    """
    Factory function to create a learning rate scheduler.

    Args:
        optimizer: PyTorch optimizer
        scheduler_type: 'cosine' or 'linear'
        warmup_epochs: Number of warmup epochs
        total_epochs: Total epochs
        min_lr_ratio: Minimum LR ratio

    Returns:
        LR scheduler instance
    """
    if scheduler_type == "cosine":
        return WarmupCosineScheduler(optimizer, warmup_epochs, total_epochs, min_lr_ratio)
    elif scheduler_type == "linear":
        return WarmupLinearScheduler(optimizer, warmup_epochs, total_epochs, min_lr_ratio)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}. Use 'cosine' or 'linear'.")


# --- Quick test ---
if __name__ == "__main__":
    model = torch.nn.Linear(10, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    scheduler = get_scheduler(optimizer, "cosine", warmup_epochs=5, total_epochs=50)

    print("Epoch | LR")
    print("-" * 20)
    for epoch in range(50):
        lr = optimizer.param_groups[0]["lr"]
        if epoch % 5 == 0:
            print(f"  {epoch:3d}  | {lr:.2e}")
        scheduler.step()
