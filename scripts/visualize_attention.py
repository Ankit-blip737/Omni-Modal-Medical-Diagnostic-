#!/usr/bin/env python3
"""
Attention Visualization Script
===============================
Generate cross-attention heatmaps and gate value visualizations.

Usage:
    python scripts/visualize_attention.py --checkpoint checkpoints/best_phase2.pth \
        --image-path data/sample.jpg --text "Right pleural effusion" --output-dir viz/
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.omni_modal import OmniModalFramework
from src.data.transforms import get_val_transforms
from src.evaluation.visualization import visualize_cross_attention, visualize_gate_values


def main():
    parser = argparse.ArgumentParser(description="Visualize Attention Maps")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Config YAML")
    parser.add_argument("--image-path", type=str, required=True, help="Input image path")
    parser.add_argument("--text", type=str, default="", help="Clinical text input")
    parser.add_argument("--output-dir", type=str, default="./visualizations")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    model_config = config.get("model", {})

    # Build and load model
    print("Loading model...")
    model = OmniModalFramework(
        num_modalities=model_config.get("num_modalities", 1),
        num_classes=model_config.get("num_classes", 14),
        pretrained=False,
    )
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Load and preprocess image
    print(f"Loading image: {args.image_path}")
    raw_image = Image.open(args.image_path).convert("L")
    raw_image_np = np.array(raw_image)

    transform = get_val_transforms(224)
    image_tensor = transform(raw_image).unsqueeze(0).unsqueeze(0)  # (1, 1, 1, H, W)
    image_tensor = image_tensor.to(device)

    # Tokenize text
    text = args.text or "No clinical information provided."
    print(f"Text: {text}")

    tokenized = model.text_encoder.tokenize([text], device=torch.device(device))

    # Forward pass
    with torch.no_grad():
        output = model(
            images=image_tensor,
            input_ids=tokenized["input_ids"],
            attention_mask=tokenized["attention_mask"],
            compute_alignment_loss=False,
        )

    # Extract attention maps
    attn_maps = output["attention_maps"]
    gate_values = output["gate_values"]

    print(f"\nGate values: g_v={gate_values['g_v']:.4f}, g_t={gate_values['g_t']:.4f}")
    print(f"Prediction probs: {output['probabilities'].cpu().numpy().round(3)}")

    # Visualize V→T attention (which text tokens each image patch attends to)
    for stage_name, maps in attn_maps.items():
        v2t_attn = maps["v2t"].cpu().numpy()[0]  # (N_patches, L)
        # Average across text tokens to get per-patch importance
        patch_importance = v2t_attn.mean(axis=-1)  # (N_patches,)

        path = os.path.join(args.output_dir, f"attention_{stage_name}_v2t.png")
        visualize_cross_attention(
            raw_image_np, patch_importance,
            title=f"V→T Attention ({stage_name})",
            output_path=path,
        )
        print(f"Saved: {path}")

    # Visualize gate values
    gate_path = os.path.join(args.output_dir, "gate_values.png")
    visualize_gate_values(
        np.array([gate_values["g_v"]]),
        np.array([gate_values["g_t"]]),
        sample_ids=["Input Sample"],
        output_path=gate_path,
    )
    print(f"Saved: {gate_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()
