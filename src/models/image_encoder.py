"""
Module 1: Joint Image Encoder
==============================
Encodes multiple MRI modalities (T1, T2, FLAIR) through a shared ConvNeXt
backbone (early layers) and modality-specific branches (late layers), then
fuses them via cross-attention.

Architecture:
    Input (B, M, 1, 224, 224)  →  Shared ConvNeXt Stages 1-2
                                →  M parallel branches (Stages 3-4)
                                →  Cross-Attention Fusion
                                →  Output (B, N_patches, D)
"""

import torch
import torch.nn as nn
from torchvision.models import convnext_base, ConvNeXt_Base_Weights


class ModalityBranch(nn.Module):
    """Late-stage modality-specific ConvNeXt branch (Stages 3-4)."""

    def __init__(self, convnext_model: nn.Module):
        super().__init__()
        # Extract stages 3 and 4 from ConvNeXt
        # ConvNeXt features: Sequential of 4 stages (each = Sequential of blocks)
        self.stage3 = convnext_model.features[4:6]  # Downsampling + Stage 3 blocks
        self.stage4 = convnext_model.features[6:]    # Downsampling + Stage 4 blocks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Shared features (B, 192, 28, 28)
        Returns:
            Modality-specific features (B, 768, 7, 7)
        """
        x = self.stage3(x)
        x = self.stage4(x)
        return x


class ImageCrossAttentionFusion(nn.Module):
    """
    Fuses features from multiple imaging modalities using multi-head
    cross-attention. Each modality attends to all others.
    """

    def __init__(self, embed_dim: int = 768, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Self-attention across concatenated modality patches
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Feed-forward after attention
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

        # Learnable modality embeddings to distinguish T1/T2/FLAIR
        self.modality_embeddings = nn.ParameterDict()

    def register_modalities(self, num_modalities: int, num_patches: int):
        """Register learnable modality position embeddings."""
        for i in range(num_modalities):
            self.modality_embeddings[f"mod_{i}"] = nn.Parameter(
                torch.randn(1, num_patches, self.embed_dim) * 0.02
            )

    def forward(self, modality_features: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            modality_features: List of M tensors, each (B, N_patches, D)
        Returns:
            Fused features (B, N_patches, D)
        """
        # Add modality-specific embeddings
        for i, feat in enumerate(modality_features):
            key = f"mod_{i}"
            if key in self.modality_embeddings:
                modality_features[i] = feat + self.modality_embeddings[key]

        # Concatenate all modality patches: (B, M*N_patches, D)
        concat = torch.cat(modality_features, dim=1)

        # Self-attention across all modality patches
        attn_out, attn_weights = self.cross_attn(concat, concat, concat)
        concat = self.norm1(concat + attn_out)

        # FFN
        ffn_out = self.ffn(concat)
        concat = self.norm2(concat + ffn_out)

        # Average pool back to single set of patches: (B, N_patches, D)
        M = len(modality_features)
        B, MN, D = concat.shape
        N = MN // M
        concat = concat.view(B, M, N, D).mean(dim=1)

        return concat


class JointImageEncoder(nn.Module):
    """
    Full Joint Image Encoder: Shared ConvNeXt backbone → modality-specific
    branches → cross-attention fusion.

    Args:
        num_modalities: Number of imaging modalities (default: 3 for T1/T2/FLAIR)
        pretrained: Whether to use ImageNet pretrained weights
        freeze_shared: Whether to freeze the shared early layers
    """

    def __init__(
        self,
        num_modalities: int = 3,
        pretrained: bool = True,
        freeze_shared: bool = True,
    ):
        super().__init__()
        self.num_modalities = num_modalities

        # Load pretrained ConvNeXt-Base
        weights = ConvNeXt_Base_Weights.DEFAULT if pretrained else None
        base_model = convnext_base(weights=weights)

        # === Shared early layers (Stages 1-2) ===
        # ConvNeXt features[0] = stem (Conv2d 4x4 stride 4)
        # features[1] = Stage 1 blocks
        # features[2] = Downsampling layer
        # features[3] = Stage 2 blocks
        self.shared_stem_and_stages = base_model.features[:4]

        # Modify stem to accept 1-channel (grayscale) input instead of 3-channel RGB
        original_stem_conv = self.shared_stem_and_stages[0][0]
        self.shared_stem_and_stages[0][0] = nn.Conv2d(
            in_channels=1,
            out_channels=original_stem_conv.out_channels,
            kernel_size=original_stem_conv.kernel_size,
            stride=original_stem_conv.stride,
            padding=original_stem_conv.padding,
        )
        # Initialize by averaging the original RGB weights
        if pretrained:
            with torch.no_grad():
                self.shared_stem_and_stages[0][0].weight.copy_(
                    original_stem_conv.weight.mean(dim=1, keepdim=True)
                )

        if freeze_shared:
            for param in self.shared_stem_and_stages.parameters():
                param.requires_grad = False

        # === Modality-specific branches (Stages 3-4) ===
        self.branches = nn.ModuleList()
        for _ in range(num_modalities):
            branch_model = convnext_base(weights=weights)
            self.branches.append(ModalityBranch(branch_model))

        # === Cross-attention fusion ===
        self.fusion = ImageCrossAttentionFusion(embed_dim=768, num_heads=8)
        self.fusion.register_modalities(num_modalities, num_patches=49)  # 7x7=49

        # Output dimension
        self.output_dim = 768

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (B, M, 1, H, W) — M modalities, each grayscale
        Returns:
            fused_features: (B, 49, 768) — 49 spatial patches, 768-dim each
        """
        B, M, C, H, W = images.shape
        assert M == self.num_modalities, (
            f"Expected {self.num_modalities} modalities, got {M}"
        )

        # Pass each modality through shared early layers
        modality_features = []
        for i in range(M):
            x = images[:, i]  # (B, 1, H, W)
            shared_feat = self.shared_stem_and_stages(x)  # (B, 192, 28, 28)

            # Pass through modality-specific branch
            branch_feat = self.branches[i](shared_feat)  # (B, 768, 7, 7)

            # Flatten spatial dims to patches: (B, 49, 768)
            branch_feat = branch_feat.flatten(2).transpose(1, 2)
            modality_features.append(branch_feat)

        # Fuse via cross-attention
        fused = self.fusion(modality_features)  # (B, 49, 768)

        return fused


# --- Quick test ---
if __name__ == "__main__":
    model = JointImageEncoder(num_modalities=3, pretrained=False, freeze_shared=False)
    dummy = torch.randn(2, 3, 1, 224, 224)
    out = model(dummy)
    print(f"Input:  {dummy.shape}")
    print(f"Output: {out.shape}")  # Expected: (2, 49, 768)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
