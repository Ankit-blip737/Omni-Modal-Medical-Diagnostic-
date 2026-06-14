"""
Module 5: Diagnostic Classification Head
=========================================
MLP classifier that takes the fused multi-modal representation and
produces diagnostic predictions.

Supports:
    - Single-label classification (softmax)
    - Multi-label classification (sigmoid per class)
    - Hierarchical classification (disease type + severity)
"""

import torch
import torch.nn as nn


class DiagnosticHead(nn.Module):
    """
    MLP diagnostic classifier.

    Architecture: Dropout → Linear → GELU → Dropout → Linear → Output

    Args:
        input_dim: Dimension of fused representation (768)
        hidden_dim: Hidden layer dimension (512)
        num_classes: Number of diagnostic classes
        dropout: Dropout rate
        multi_label: If True, uses sigmoid (multi-label); else softmax (single-label)
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 512,
        num_classes: int = 14,
        dropout: float = 0.3,
        multi_label: bool = True,
    ):
        super().__init__()
        self.multi_label = multi_label

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout * 0.67),  # Slightly less dropout in 2nd layer
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, fused_features: torch.Tensor) -> dict:
        """
        Args:
            fused_features: (B, input_dim) from fusion engine

        Returns:
            Dictionary with:
                'logits': (B, num_classes) — raw logits
                'probabilities': (B, num_classes) — predicted probabilities
        """
        logits = self.classifier(fused_features)

        if self.multi_label:
            probs = torch.sigmoid(logits)
        else:
            probs = torch.softmax(logits, dim=-1)

        return {
            "logits": logits,
            "probabilities": probs,
        }


# --- Quick test ---
if __name__ == "__main__":
    head = DiagnosticHead(input_dim=768, num_classes=14, multi_label=True)
    dummy = torch.randn(4, 768)
    out = head(dummy)
    print(f"Logits: {out['logits'].shape}")          # (4, 14)
    print(f"Probs:  {out['probabilities'].shape}")    # (4, 14)
    print(f"Params: {sum(p.numel() for p in head.parameters()):,}")
