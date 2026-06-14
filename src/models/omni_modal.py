"""
Omni-Modal Diagnostic Framework — Full Model
=============================================
Ties all 5 modules together into a single end-to-end framework:

    Module 1: JointImageEncoder     — Multi-modal MRI encoding
    Module 2: ClinicalTextEncoder   — BioBERT text encoding
    Module 3: VisualSemanticAlignment — Contrastive metric space alignment
    Module 4: BidirectionalProgressiveFusion — Reciprocal cross-attention fusion
    Module 5: DiagnosticHead        — Classification output

Usage:
    model = OmniModalFramework(num_classes=14)
    output = model(images, input_ids, attention_mask)
"""

import torch
import torch.nn as nn
from typing import Optional

from .image_encoder import JointImageEncoder
from .text_encoder import ClinicalTextEncoder
from .alignment import VisualSemanticAlignmentModule
from .fusion import BidirectionalProgressiveFusion
from .classifier import DiagnosticHead


class OmniModalFramework(nn.Module):
    """
    Complete Omni-Modal Diagnostic Framework.

    Fuses multi-modal medical images (MRI T1/T2/FLAIR or single-modality)
    with clinical text (radiology reports / EHR) for diagnostic classification.

    Args:
        num_modalities: Number of imaging modalities (1 for X-ray, 3 for MRI)
        num_classes: Number of diagnostic output classes
        text_model: Text encoder model name ('pubmedbert', 'biobert', 'clinicalbert')
        freeze_image_backbone: Whether to freeze ConvNeXt shared layers
        freeze_text_layers: Number of BioBERT layers to freeze
        lora_rank: LoRA rank for text encoder fine-tuning
        fusion_dim: Internal dimension for fusion module
        projection_dim: Dimension of contrastive alignment space
        num_fusion_stages: Number of fine fusion stages
        modality_dropout_p: Probability of modality dropout during training
        multi_label: Whether classification is multi-label
        pretrained: Whether to use pretrained weights
    """

    def __init__(
        self,
        num_modalities: int = 3,
        num_classes: int = 14,
        text_model: str = "pubmedbert",
        freeze_image_backbone: bool = True,
        freeze_text_layers: int = 8,
        lora_rank: int = 8,
        fusion_dim: int = 768,
        projection_dim: int = 256,
        num_fusion_stages: int = 1,
        modality_dropout_p: float = 0.15,
        multi_label: bool = True,
        pretrained: bool = True,
    ):
        super().__init__()
        self.num_modalities = num_modalities
        self.num_classes = num_classes

        # ─── Module 1: Joint Image Encoder ───
        self.image_encoder = JointImageEncoder(
            num_modalities=num_modalities,
            pretrained=pretrained,
            freeze_shared=freeze_image_backbone,
        )

        # ─── Module 2: Clinical Text Encoder ───
        self.text_encoder = ClinicalTextEncoder(
            model_name=text_model,
            max_length=512,
            freeze_layers=freeze_text_layers,
            lora_rank=lora_rank,
        )

        # ─── Module 3: Visual-Semantic Alignment ───
        self.alignment = VisualSemanticAlignmentModule(
            visual_dim=self.image_encoder.output_dim,
            text_dim=self.text_encoder.output_dim,
            projection_dim=projection_dim,
        )

        # ─── Module 4: Bidirectional Progressive Fusion ───
        self.fusion = BidirectionalProgressiveFusion(
            visual_dim=self.image_encoder.output_dim,
            text_dim=self.text_encoder.output_dim,
            fusion_dim=fusion_dim,
            num_heads=8,
            num_fine_stages=num_fusion_stages,
            modality_dropout_p=modality_dropout_p,
        )

        # ─── Module 5: Diagnostic Classification Head ───
        self.classifier = DiagnosticHead(
            input_dim=fusion_dim,
            hidden_dim=fusion_dim // 2,  # 384 when fusion_dim=768
            num_classes=num_classes,
            multi_label=multi_label,
        )

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        compute_alignment_loss: bool = True,
    ) -> dict:
        """
        Full forward pass through all 5 modules.

        Args:
            images: (B, M, 1, H, W) — M modalities, grayscale
            input_ids: (B, L) — tokenized clinical text
            attention_mask: (B, L) — text attention mask
            token_type_ids: (B, L) — optional BERT token types
            compute_alignment_loss: Whether to compute contrastive alignment loss

        Returns:
            Dictionary with:
                'logits': (B, num_classes) — raw classification logits
                'probabilities': (B, num_classes) — classification probabilities
                'alignment_loss': scalar — contrastive alignment loss
                'alignment_metrics': dict — retrieval accuracy metrics
                'gate_values': dict — modality gate activations
                'attention_maps': dict — cross-attention weights per stage
                'visual_features': (B, N, D) — encoded visual features
                'text_cls': (B, D) — text [CLS] embedding
        """
        # ─── Module 1: Encode Images ───
        visual_features = self.image_encoder(images)  # (B, 49, 768)

        # ─── Module 2: Encode Text ───
        text_output = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        cls_embedding = text_output["cls_embedding"]       # (B, 768)
        token_embeddings = text_output["token_embeddings"]  # (B, L, 768)
        text_pad_mask = ~attention_mask.bool()              # True = padded

        # ─── Module 3: Visual-Semantic Alignment ───
        alignment_output = self.alignment(
            visual_features=visual_features,
            cls_embedding=cls_embedding,
            compute_loss=compute_alignment_loss,
        )

        # ─── Module 4: Bidirectional Progressive Fusion ───
        fusion_output = self.fusion(
            visual_features=visual_features,
            text_features=token_embeddings,
            text_mask=text_pad_mask,
        )

        # ─── Module 5: Diagnostic Classification ───
        fused_representation = fusion_output["fused_representation"]  # (B, D)
        cls_output = self.classifier(fused_representation)

        # ─── Aggregate all outputs ───
        result = {
            # Classification
            "logits": cls_output["logits"],
            "probabilities": cls_output["probabilities"],
            # Alignment
            "alignment_loss": alignment_output.get("alignment_loss", None),
            "alignment_metrics": alignment_output.get("metrics", {}),
            # Fusion
            "fused_representation": fused_representation,
            "gate_values": fusion_output["gate_values"],
            "attention_maps": fusion_output["attention_maps"],
            # Raw features (for visualization / downstream use)
            "visual_features": visual_features,
            "text_cls": cls_embedding,
            "visual_refined": fusion_output["visual_refined"],
            "text_refined": fusion_output["text_refined"],
        }

        return result

    def get_parameter_groups(self, lr_backbone: float = 1e-6, lr_fusion: float = 5e-5):
        """
        Get parameter groups with different learning rates for
        the 3-phase training strategy.

        Returns list of dicts compatible with torch.optim.
        """
        # Frozen backbone parameters (very low LR if unfrozen)
        backbone_params = []
        backbone_params.extend(self.image_encoder.shared_stem_and_stages.parameters())
        backbone_params.extend(self.text_encoder.bert.embeddings.parameters())

        # LoRA / branch parameters (medium LR)
        branch_params = []
        branch_params.extend(self.image_encoder.branches.parameters())
        branch_params.extend(self.image_encoder.fusion.parameters())
        for name, param in self.text_encoder.named_parameters():
            if "lora" in name.lower():
                branch_params.append(param)

        # Fusion + classifier parameters (high LR)
        fusion_params = []
        fusion_params.extend(self.alignment.parameters())
        fusion_params.extend(self.fusion.parameters())
        fusion_params.extend(self.classifier.parameters())

        return [
            {"params": backbone_params, "lr": lr_backbone, "name": "backbone"},
            {"params": branch_params, "lr": lr_fusion * 0.5, "name": "branches"},
            {"params": fusion_params, "lr": lr_fusion, "name": "fusion"},
        ]

    def count_parameters(self) -> dict:
        """Count total and trainable parameters per module."""
        modules = {
            "image_encoder": self.image_encoder,
            "text_encoder": self.text_encoder,
            "alignment": self.alignment,
            "fusion": self.fusion,
            "classifier": self.classifier,
        }

        stats = {}
        total_params = 0
        total_trainable = 0

        for name, module in modules.items():
            total = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            stats[name] = {"total": total, "trainable": trainable}
            total_params += total
            total_trainable += trainable

        stats["TOTAL"] = {"total": total_params, "trainable": total_trainable}
        return stats


# --- Quick test ---
if __name__ == "__main__":
    print("=" * 60)
    print("Omni-Modal Diagnostic Framework — Architecture Test")
    print("=" * 60)

    # Create model (without pre-trained weights for quick testing)
    model = OmniModalFramework(
        num_modalities=3,
        num_classes=14,
        pretrained=False,
    )

    # Dummy inputs
    B = 2
    images = torch.randn(B, 3, 1, 224, 224)
    input_ids = torch.randint(0, 1000, (B, 128))
    attention_mask = torch.ones(B, 128, dtype=torch.long)
    attention_mask[:, 100:] = 0  # Last 28 tokens are padding

    # Forward pass
    model.train()
    output = model(images, input_ids, attention_mask)

    # Print results
    print(f"\n{'─' * 40}")
    print(f"Logits:             {output['logits'].shape}")
    print(f"Probabilities:      {output['probabilities'].shape}")
    print(f"Alignment loss:     {output['alignment_loss']:.4f}")
    print(f"Gate values:        g_v={output['gate_values']['g_v']:.3f}, "
          f"g_t={output['gate_values']['g_t']:.3f}")
    print(f"Attention stages:   {list(output['attention_maps'].keys())}")

    # Parameter count
    print(f"\n{'─' * 40}")
    print("Parameter Count:")
    stats = model.count_parameters()
    for module_name, counts in stats.items():
        trainable_pct = counts['trainable'] / max(counts['total'], 1) * 100
        print(f"  {module_name:20s}: {counts['total']:>12,} total, "
              f"{counts['trainable']:>12,} trainable ({trainable_pct:.1f}%)")
