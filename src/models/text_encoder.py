"""
Module 2: Clinical Text Encoder
================================
Encodes clinical text (radiology reports, EHR notes) using a pre-trained
biomedical language model (PubMedBERT or BioBERT).

Produces two types of embeddings:
    1. Global [CLS] embedding — sentence-level semantics
    2. Token-level embeddings — for local cross-attention with image patches

Supports LoRA fine-tuning for parameter-efficient adaptation.

Architecture:
    Input text → Tokenizer → BioBERT (frozen layers 1-8, LoRA layers 9-12)
                            → [CLS] global embedding (B, 768)
                            → Token embeddings (B, L, 768)
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class LoRALinear(nn.Module):
    """
    Low-Rank Adaptation (LoRA) wrapper for a linear layer.
    Adds a low-rank trainable bypass: output = W_frozen(x) + B @ A(x)

    Args:
        original_linear: The frozen linear layer to adapt
        rank: Rank of the low-rank decomposition (default: 8)
        alpha: Scaling factor (default: 16)
    """

    def __init__(self, original_linear: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.original = original_linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = original_linear.in_features
        out_features = original_linear.out_features

        # Freeze the original weights
        for param in self.original.parameters():
            param.requires_grad = False

        # Low-rank trainable matrices
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)

        # Initialize A with Kaiming, B with zeros (so LoRA starts as identity)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original frozen output + scaled low-rank adaptation
        return self.original(x) + self.lora_B(self.lora_A(x)) * self.scaling


class ClinicalTextEncoder(nn.Module):
    """
    Clinical Text Encoder using PubMedBERT/BioBERT.

    Args:
        model_name: HuggingFace model name
            - "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext" (recommended)
            - "dmis-lab/biobert-base-cased-v1.1"
            - "emilyalsentzer/Bio_ClinicalBERT"
        max_length: Maximum token sequence length
        freeze_layers: Number of transformer layers to freeze from the bottom
        lora_rank: LoRA rank for fine-tuning upper layers (0 = no LoRA)
        lora_alpha: LoRA scaling factor
    """

    SUPPORTED_MODELS = {
        "pubmedbert": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        "biobert": "dmis-lab/biobert-base-cased-v1.1",
        "clinicalbert": "emilyalsentzer/Bio_ClinicalBERT",
    }

    def __init__(
        self,
        model_name: str = "pubmedbert",
        max_length: int = 512,
        freeze_layers: int = 8,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
    ):
        super().__init__()
        self.max_length = max_length

        # Resolve shorthand names
        full_model_name = self.SUPPORTED_MODELS.get(model_name, model_name)

        # Load pre-trained model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(full_model_name)
        self.bert = AutoModel.from_pretrained(full_model_name)

        # Freeze embedding layer
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False

        # Freeze early transformer layers
        for i, layer in enumerate(self.bert.encoder.layer):
            if i < freeze_layers:
                for param in layer.parameters():
                    param.requires_grad = False
            elif lora_rank > 0:
                # Apply LoRA to the query and value projections in attention
                layer.attention.self.query = LoRALinear(
                    layer.attention.self.query, rank=lora_rank, alpha=lora_alpha
                )
                layer.attention.self.value = LoRALinear(
                    layer.attention.self.value, rank=lora_rank, alpha=lora_alpha
                )

        # Output dimension matches BERT hidden size
        self.output_dim = self.bert.config.hidden_size  # 768

    def tokenize(self, texts: list[str], device: torch.device = None) -> dict:
        """
        Tokenize a batch of clinical texts.

        Args:
            texts: List of clinical text strings
            device: Target device for tokenized tensors

        Returns:
            Dictionary with 'input_ids', 'attention_mask', 'token_type_ids'
        """
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        if device is not None:
            encoded = {k: v.to(device) for k, v in encoded.items()}
        return encoded

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor = None,
    ) -> dict:
        """
        Args:
            input_ids: (B, L) — tokenized text
            attention_mask: (B, L) — attention mask
            token_type_ids: (B, L) — optional token type IDs

        Returns:
            Dictionary with:
                'cls_embedding': (B, 768) — global sentence embedding
                'token_embeddings': (B, L, 768) — per-token embeddings
                'attention_mask': (B, L) — passed through for fusion masking
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # [CLS] token is the first token
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # (B, 768)
        token_embeddings = outputs.last_hidden_state         # (B, L, 768)

        return {
            "cls_embedding": cls_embedding,
            "token_embeddings": token_embeddings,
            "attention_mask": attention_mask,
        }

    def get_trainable_params(self) -> int:
        """Count trainable parameters (useful for logging)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_params(self) -> int:
        """Count total parameters."""
        return sum(p.numel() for p in self.parameters())


# --- Quick test ---
if __name__ == "__main__":
    # Test with a dummy model name (will fail without internet/model download)
    print("ClinicalTextEncoder module loaded successfully.")
    print(f"Supported models: {list(ClinicalTextEncoder.SUPPORTED_MODELS.keys())}")

    # Architecture summary
    print("\nArchitecture:")
    print("  Input:  raw clinical text → tokenizer → (B, L) token IDs")
    print("  Output: cls_embedding (B, 768) + token_embeddings (B, L, 768)")
    print("  LoRA:   applied to Q,V projections of layers 9-12")
