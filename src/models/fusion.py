"""
Module 4: Bidirectional Progressive Fusion Engine
=================================================
The core reciprocal fusion module that progressively integrates visual and
textual features through multi-stage cross-attention with gated aggregation.

Architecture:
    Stage 1 (Coarse): Bidirectional cross-attention for global alignment
    Stage 2 (Fine):   Bidirectional cross-attention for local grounding
    Stage 3 (Gate):   Dynamic modality weighting via sigmoid gating

Paper references:
    - BiPVL-Seg (2025): Bidirectional Progressive Vision-Language Fusion
    - DRIFA-Net (WACV 2025): Dual Robust Information Fusion Attention
    - M³Bind (2025): Multi-modal alignment via shared text space
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class CrossAttentionBlock(nn.Module):
    """
    Single cross-attention block with residual connections and FFN.

    Query modality attends to the Key/Value modality to extract
    relevant context.

    Args:
        embed_dim: Feature dimension
        num_heads: Number of attention heads
        dropout: Dropout rate
        use_ffn: Whether to include a feed-forward network after attention
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_ffn: bool = True,
    ):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_attn = nn.LayerNorm(embed_dim)

        self.use_ffn = use_ffn
        if use_ffn:
            self.ffn = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim * 4, embed_dim),
                nn.Dropout(dropout),
            )
            self.norm_ffn = nn.LayerNorm(embed_dim)

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        context_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: (B, N_q, D) — modality being updated
            context: (B, N_kv, D) — modality providing information
            context_key_padding_mask: (B, N_kv) — True for padded positions

        Returns:
            updated: (B, N_q, D) — updated query features
            attn_weights: (B, N_q, N_kv) — attention weight matrix
        """
        # Cross-attention with pre-norm residual
        attn_out, attn_weights = self.cross_attn(
            query=query,
            key=context,
            value=context,
            key_padding_mask=context_key_padding_mask,
        )
        updated = self.norm_attn(query + attn_out)

        # Feed-forward with pre-norm residual
        if self.use_ffn:
            ffn_out = self.ffn(updated)
            updated = self.norm_ffn(updated + ffn_out)

        return updated, attn_weights


class BidirectionalFusionStage(nn.Module):
    """
    A single bidirectional fusion stage: visual features attend to text
    AND text features attend to visual simultaneously.

    This implements the reciprocal information exchange described in
    BiPVL-Seg (2025).

    Args:
        embed_dim: Feature dimension (768)
        num_heads: Number of attention heads (8)
        dropout: Dropout rate
        use_ffn: Whether to include FFN after attention
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_ffn: bool = True,
    ):
        super().__init__()

        # Visual attends to Text (V→T: "which text tokens inform this image patch?")
        self.v2t_attn = CrossAttentionBlock(embed_dim, num_heads, dropout, use_ffn)

        # Text attends to Visual (T→V: "which image patches ground this text token?")
        self.t2v_attn = CrossAttentionBlock(embed_dim, num_heads, dropout, use_ffn)

    def forward(
        self,
        visual_features: torch.Tensor,
        text_features: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Args:
            visual_features: (B, N_patches, D)
            text_features: (B, L, D)
            text_mask: (B, L) — True for padded text positions

        Returns:
            visual_updated: (B, N_patches, D) — text-informed visual features
            text_updated: (B, L, D) — visually-grounded text features
            attn_maps: dict with 'v2t' and 't2v' attention weight matrices
        """
        # Bidirectional cross-attention
        visual_updated, v2t_attn = self.v2t_attn(
            query=visual_features,
            context=text_features,
            context_key_padding_mask=text_mask,
        )
        text_updated, t2v_attn = self.t2v_attn(
            query=text_features,
            context=visual_features,
            # No mask for visual features (all patches are valid)
        )

        attn_maps = {
            "v2t": v2t_attn,  # (B, N_patches, L) — which text informs each patch
            "t2v": t2v_attn,  # (B, L, N_patches) — which patches ground each token
        }

        return visual_updated, text_updated, attn_maps


class GatedModalityAggregation(nn.Module):
    """
    Dynamic gated aggregation that learns instance-specific weights for
    visual vs. textual contributions.

    For each input sample, computes sigmoid gates:
        g_v = σ(W_v · [v; t] + b_v)   — visual importance
        g_t = σ(W_t · [v; t] + b_t)   — text importance

    Then fuses: F = g_v ⊙ v + g_t ⊙ t

    This is critical for clinical robustness: when a radiology report is
    detailed but the image is noisy, the model learns to rely more on text
    (and vice versa).

    Args:
        embed_dim: Feature dimension (768)
        dropout: Dropout rate
    """

    def __init__(self, embed_dim: int = 768, dropout: float = 0.1):
        super().__init__()

        # Gate networks: take concatenated [visual; text] → per-dim gate values
        self.visual_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid(),
        )
        self.text_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid(),
        )

        # Attentive pooling to collapse sequence dim → single vector
        self.attn_pool_query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.attn_pool = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=8,
            batch_first=True,
        )
        self.pool_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        visual_features: torch.Tensor,
        text_features: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            visual_features: (B, N_patches, D)
            text_features: (B, L, D)
            text_mask: (B, L) — True for padded positions

        Returns:
            fused: (B, D) — single fused representation per sample
            gate_values: dict with 'g_v' and 'g_t' scalar gate means
        """
        B = visual_features.shape[0]

        # Pool both modalities to global vectors for gating
        v_global = visual_features.mean(dim=1)  # (B, D)

        # Mask-aware pooling for text
        if text_mask is not None:
            # Invert mask: True=valid → use as weight
            valid_mask = (~text_mask).float().unsqueeze(-1)  # (B, L, 1)
            t_global = (text_features * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1)
        else:
            t_global = text_features.mean(dim=1)  # (B, D)

        # Compute gates from concatenated global features
        combined = torch.cat([v_global, t_global], dim=-1)  # (B, 2D)
        g_v = self.visual_gate(combined)  # (B, D)
        g_t = self.text_gate(combined)    # (B, D)

        # Apply gates element-wise
        gated_visual = g_v * v_global   # (B, D)
        gated_text = g_t * t_global     # (B, D)
        fused_global = gated_visual + gated_text  # (B, D)

        # Also create a richer fused sequence for attentive pooling
        # Concatenate gated sequences and add the global fused vector
        gated_v_seq = g_v.unsqueeze(1) * visual_features  # (B, N, D)
        gated_t_seq = g_t.unsqueeze(1) * text_features    # (B, L, D)
        fused_seq = torch.cat([gated_v_seq, gated_t_seq], dim=1)  # (B, N+L, D)

        # Attentive pooling: learnable query attends to fused sequence
        query = self.attn_pool_query.expand(B, -1, -1)  # (B, 1, D)
        pooled, _ = self.attn_pool(query, fused_seq, fused_seq)
        pooled = self.pool_norm(pooled.squeeze(1))  # (B, D)

        # Combine global gated fusion with attentive pooling
        fused = fused_global + pooled  # (B, D)

        gate_values = {
            "g_v": g_v.mean().item(),  # Average visual gate activation
            "g_t": g_t.mean().item(),  # Average text gate activation
            "g_v_per_sample": g_v.mean(dim=-1),  # (B,) per-sample visual weight
            "g_t_per_sample": g_t.mean(dim=-1),  # (B,) per-sample text weight
        }

        return fused, gate_values


class BidirectionalProgressiveFusion(nn.Module):
    """
    Full Bidirectional Progressive Fusion Engine.

    Performs multi-stage reciprocal fusion of visual and textual features:
        Stage 1 (Coarse): Global cross-modal alignment
        Stage 2 (Fine):   Local feature grounding
        Stage 3 (Gate):   Dynamic modality aggregation

    This is the central fusion mechanism of the Omni-Modal Diagnostic
    Framework, combining insights from BiPVL-Seg, DRIFA-Net, and
    gated multimodal fusion literature.

    Args:
        visual_dim: Dimension of visual features (768)
        text_dim: Dimension of text token embeddings (768)
        fusion_dim: Internal fusion dimension (768)
        num_heads: Number of attention heads per stage
        num_fine_stages: Number of fine-grained fusion stages (default: 1)
        dropout: Dropout rate
        modality_dropout_p: Probability of dropping an entire modality during training
    """

    def __init__(
        self,
        visual_dim: int = 768,
        text_dim: int = 768,
        fusion_dim: int = 768,
        num_heads: int = 8,
        num_fine_stages: int = 1,
        dropout: float = 0.1,
        modality_dropout_p: float = 0.15,
    ):
        super().__init__()
        self.fusion_dim = fusion_dim
        self.modality_dropout_p = modality_dropout_p

        # Projection to common fusion dimension (if needed)
        self.visual_proj = (
            nn.Linear(visual_dim, fusion_dim) if visual_dim != fusion_dim
            else nn.Identity()
        )
        self.text_proj = (
            nn.Linear(text_dim, fusion_dim) if text_dim != fusion_dim
            else nn.Identity()
        )

        # Stage 1: Coarse global fusion (no FFN, just cross-attention)
        self.coarse_stage = BidirectionalFusionStage(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_ffn=False,  # Keep it lightweight for global alignment
        )

        # Stage 2: Fine local fusion (with FFN for richer representations)
        self.fine_stages = nn.ModuleList([
            BidirectionalFusionStage(
                embed_dim=fusion_dim,
                num_heads=num_heads,
                dropout=dropout,
                use_ffn=True,
            )
            for _ in range(num_fine_stages)
        ])

        # Stage 3: Gated modality aggregation
        self.gated_aggregation = GatedModalityAggregation(
            embed_dim=fusion_dim,
            dropout=dropout,
        )

    def _apply_modality_dropout(
        self,
        visual: torch.Tensor,
        text: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Randomly zero out an entire modality during training.
        This forces the model to be robust to missing modalities.

        Never drops BOTH modalities simultaneously.
        """
        if not self.training or self.modality_dropout_p <= 0:
            return visual, text

        # Decide which modality to drop (if any)
        rand = torch.rand(1).item()
        if rand < self.modality_dropout_p / 2:
            # Drop visual
            visual = torch.zeros_like(visual)
        elif rand < self.modality_dropout_p:
            # Drop text
            text = torch.zeros_like(text)

        return visual, text

    def forward(
        self,
        visual_features: torch.Tensor,
        text_features: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Args:
            visual_features: (B, N_patches, D_v) from Joint Image Encoder
            text_features: (B, L, D_t) from Text Encoder token embeddings
            text_mask: (B, L) — True for padded text positions

        Returns:
            Dictionary with:
                'fused_representation': (B, fusion_dim) — final fused vector
                'gate_values': dict with modality gate activations
                'attention_maps': dict with cross-attention weights from each stage
                'visual_refined': (B, N_patches, D) — text-refined visual features
                'text_refined': (B, L, D) — visually-grounded text features
        """
        # Project to common dimension
        V = self.visual_proj(visual_features)  # (B, N, D)
        T = self.text_proj(text_features)      # (B, L, D)

        # Apply modality dropout during training
        V, T = self._apply_modality_dropout(V, T)

        all_attn_maps = {}

        # --- Stage 1: Coarse Global Fusion ---
        V, T, attn_maps = self.coarse_stage(V, T, text_mask)
        all_attn_maps["coarse"] = attn_maps

        # --- Stage 2: Fine Local Fusion ---
        for i, fine_stage in enumerate(self.fine_stages):
            V, T, attn_maps = fine_stage(V, T, text_mask)
            all_attn_maps[f"fine_{i}"] = attn_maps

        # --- Stage 3: Gated Modality Aggregation ---
        fused, gate_values = self.gated_aggregation(V, T, text_mask)

        return {
            "fused_representation": fused,        # (B, fusion_dim)
            "gate_values": gate_values,
            "attention_maps": all_attn_maps,
            "visual_refined": V,                   # For visualization
            "text_refined": T,                     # For visualization
        }


# --- Quick test ---
if __name__ == "__main__":
    B, N_patches, L, D = 4, 49, 128, 768

    visual = torch.randn(B, N_patches, D)
    text = torch.randn(B, L, D)
    text_mask = torch.zeros(B, L, dtype=torch.bool)
    text_mask[:, 100:] = True  # Last 28 tokens are padding

    fusion = BidirectionalProgressiveFusion(
        visual_dim=D, text_dim=D, fusion_dim=D,
        num_heads=8, num_fine_stages=1,
        modality_dropout_p=0.15,
    )
    fusion.train()

    output = fusion(visual, text, text_mask)

    print(f"Fused representation: {output['fused_representation'].shape}")  # (4, 768)
    print(f"Visual refined:       {output['visual_refined'].shape}")        # (4, 49, 768)
    print(f"Text refined:         {output['text_refined'].shape}")          # (4, 128, 768)
    print(f"Gate values:          g_v={output['gate_values']['g_v']:.3f}, "
          f"g_t={output['gate_values']['g_t']:.3f}")
    print(f"Attention map stages: {list(output['attention_maps'].keys())}")
    print(f"Params:               {sum(p.numel() for p in fusion.parameters()):,}")
