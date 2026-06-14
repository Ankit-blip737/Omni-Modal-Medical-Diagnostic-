"""
Module 3: Visual-Semantic Alignment Module
==========================================
Maps visual features from the Joint Image Encoder and text embeddings from
BioBERT into a unified 256-dimensional metric space using projection heads
and contrastive alignment (symmetric InfoNCE / CLIP-style loss).

This is the CORE INNOVATION of the framework — it bridges the modality gap
between spatial visual features and sequential text embeddings.

Architecture:
    Visual: (B, 49, 768) → GAP → (B, 768) → MLP → L2Norm → v̂ ∈ ℝ²⁵⁶
    Text:   (B, 768)     →        (B, 768) → MLP → L2Norm → t̂ ∈ ℝ²⁵⁶
    Loss:   Symmetric InfoNCE(v̂, t̂, τ)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """
    2-layer MLP projection head that maps encoder outputs into the
    contrastive alignment space.

    Architecture: Linear → BatchNorm → GELU → Linear → L2Normalize

    Args:
        input_dim: Input feature dimension (768 for both ConvNeXt and BERT)
        hidden_dim: Hidden layer dimension (512)
        output_dim: Alignment space dimension (256)
        dropout: Dropout rate
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 512,
        output_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, input_dim)
        Returns:
            L2-normalized projection: (B, output_dim)
        """
        projected = self.projection(x)
        return F.normalize(projected, p=2, dim=-1)


class SymmetricInfoNCELoss(nn.Module):
    """
    Symmetric InfoNCE (CLIP-style) contrastive loss.

    Computes bidirectional cross-entropy between cosine similarity matrix
    and identity labels:
        ℒ = (ℒ_v→t + ℒ_t→v) / 2

    Args:
        init_temperature: Initial temperature τ (default: 0.07, following CLIP)
        learnable_temperature: Whether τ should be learned during training
        max_temperature: Upper bound on τ to prevent training instability
    """

    def __init__(
        self,
        init_temperature: float = 0.07,
        learnable_temperature: bool = True,
        max_temperature: float = 100.0,
    ):
        super().__init__()
        # Store log(1/τ) as learnable parameter (more numerically stable)
        self.log_temperature = nn.Parameter(
            torch.tensor(1.0 / init_temperature).log(),
            requires_grad=learnable_temperature,
        )
        self.max_temperature = max_temperature

    @property
    def temperature(self) -> torch.Tensor:
        """Current temperature value."""
        return torch.clamp(
            self.log_temperature.exp().reciprocal(),
            min=1e-4,
            max=self.max_temperature,
        )

    def forward(
        self,
        visual_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
    ) -> dict:
        """
        Args:
            visual_embeddings: L2-normalized visual projections (B, D)
            text_embeddings: L2-normalized text projections (B, D)

        Returns:
            Dictionary with:
                'loss': scalar contrastive loss
                'similarity_matrix': (B, B) cosine similarity matrix
                'temperature': current temperature value
                'accuracy_v2t': image-to-text retrieval accuracy (top-1)
                'accuracy_t2v': text-to-image retrieval accuracy (top-1)
        """
        B = visual_embeddings.shape[0]
        device = visual_embeddings.device

        # Compute cosine similarity matrix scaled by temperature
        # S_ij = (v_i · t_j) / τ
        logits = (visual_embeddings @ text_embeddings.T) / self.temperature

        # Labels: diagonal entries are positives (paired samples)
        labels = torch.arange(B, device=device)

        # Bidirectional cross-entropy
        loss_v2t = F.cross_entropy(logits, labels)       # image-to-text
        loss_t2v = F.cross_entropy(logits.T, labels)     # text-to-image
        loss = (loss_v2t + loss_t2v) / 2.0

        # Compute retrieval accuracy for monitoring
        with torch.no_grad():
            preds_v2t = logits.argmax(dim=1)
            preds_t2v = logits.argmax(dim=0)
            acc_v2t = (preds_v2t == labels).float().mean()
            acc_t2v = (preds_t2v == labels).float().mean()

        return {
            "loss": loss,
            "similarity_matrix": logits.detach(),
            "temperature": self.temperature.item(),
            "accuracy_v2t": acc_v2t.item(),
            "accuracy_t2v": acc_t2v.item(),
        }


class VisualSemanticAlignmentModule(nn.Module):
    """
    Full Visual-Semantic Alignment Module.

    Takes raw visual features (B, N_patches, D) and text [CLS] embeddings (B, D),
    projects them into a shared metric space, and computes the contrastive
    alignment loss.

    Args:
        visual_dim: Dimension of visual features (768)
        text_dim: Dimension of text [CLS] embedding (768)
        projection_dim: Shared alignment space dimension (256)
        hidden_dim: Hidden layer in projection heads (512)
        temperature: Initial contrastive temperature (0.07)
    """

    def __init__(
        self,
        visual_dim: int = 768,
        text_dim: int = 768,
        projection_dim: int = 256,
        hidden_dim: int = 512,
        temperature: float = 0.07,
    ):
        super().__init__()

        # Global average pooling for visual features
        self.visual_pool = nn.AdaptiveAvgPool1d(1)

        # Projection heads
        self.visual_projector = ProjectionHead(
            input_dim=visual_dim,
            hidden_dim=hidden_dim,
            output_dim=projection_dim,
        )
        self.text_projector = ProjectionHead(
            input_dim=text_dim,
            hidden_dim=hidden_dim,
            output_dim=projection_dim,
        )

        # Contrastive loss
        self.contrastive_loss = SymmetricInfoNCELoss(
            init_temperature=temperature,
            learnable_temperature=True,
        )

    def forward(
        self,
        visual_features: torch.Tensor,
        cls_embedding: torch.Tensor,
        compute_loss: bool = True,
    ) -> dict:
        """
        Args:
            visual_features: (B, N_patches, D) from Joint Image Encoder
            cls_embedding: (B, D) from Text Encoder [CLS] token
            compute_loss: Whether to compute contrastive loss

        Returns:
            Dictionary with:
                'visual_projection': (B, projection_dim) — aligned visual embedding
                'text_projection': (B, projection_dim) — aligned text embedding
                'alignment_loss': scalar (if compute_loss=True)
                'metrics': retrieval accuracy metrics (if compute_loss=True)
        """
        # Global average pool visual features: (B, N, D) → (B, D)
        visual_global = visual_features.mean(dim=1)  # (B, D)

        # Project into shared space
        v_proj = self.visual_projector(visual_global)   # (B, projection_dim)
        t_proj = self.text_projector(cls_embedding)      # (B, projection_dim)

        result = {
            "visual_projection": v_proj,
            "text_projection": t_proj,
        }

        if compute_loss:
            loss_output = self.contrastive_loss(v_proj, t_proj)
            result["alignment_loss"] = loss_output["loss"]
            result["metrics"] = {
                "temperature": loss_output["temperature"],
                "acc_v2t": loss_output["accuracy_v2t"],
                "acc_t2v": loss_output["accuracy_t2v"],
            }

        return result


# --- Quick test ---
if __name__ == "__main__":
    B, N, D = 8, 49, 768

    visual_feats = torch.randn(B, N, D)
    cls_emb = torch.randn(B, D)

    module = VisualSemanticAlignmentModule()
    output = module(visual_feats, cls_emb, compute_loss=True)

    print(f"Visual projection: {output['visual_projection'].shape}")  # (8, 256)
    print(f"Text projection:   {output['text_projection'].shape}")    # (8, 256)
    print(f"Alignment loss:    {output['alignment_loss'].item():.4f}")
    print(f"Temperature:       {output['metrics']['temperature']:.4f}")
    print(f"V→T accuracy:      {output['metrics']['acc_v2t']:.4f}")
    print(f"T→V accuracy:      {output['metrics']['acc_t2v']:.4f}")
    print(f"Params:            {sum(p.numel() for p in module.parameters()):,}")
