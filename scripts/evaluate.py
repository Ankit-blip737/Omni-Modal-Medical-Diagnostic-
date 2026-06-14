#!/usr/bin/env python3
"""
Evaluation Script
=================
Evaluate a trained Omni-Modal Diagnostic Framework checkpoint.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/best_phase2.pth --config configs/default.yaml
    python scripts/evaluate.py --checkpoint checkpoints/best_phase2.pth --split test --output-dir results/
"""

import os
import sys
import json
import argparse
import yaml
import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.omni_modal import OmniModalFramework
from src.evaluation.metrics import compute_multilabel_metrics, format_metrics_table, MetricTracker
from src.evaluation.visualization import plot_roc_curves


def main():
    parser = argparse.ArgumentParser(description="Evaluate Omni-Modal Diagnostic Framework")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Config YAML")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--output-dir", type=str, default="./results", help="Output directory")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # Load config
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model_config = config.get("model", {})
    data_config = config.get("data", {})

    if args.batch_size:
        data_config["batch_size"] = args.batch_size

    # Build model
    print("Loading model...")
    model = OmniModalFramework(
        num_modalities=model_config.get("num_modalities", 3),
        num_classes=model_config.get("num_classes", 14),
        text_model=model_config.get("text_model", "pubmedbert"),
        pretrained=False,
    )

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"Loaded checkpoint: epoch {checkpoint.get('epoch', '?')}")

    # Create dataloader
    dataset_type = data_config.get("dataset", "mimic_cxr")
    if dataset_type == "mimic_cxr":
        from src.data.mimic_cxr import get_mimic_dataloaders, CHEXPERT_LABELS
        loaders = get_mimic_dataloaders(data_config)
        class_names = CHEXPERT_LABELS
    else:
        from src.data.brats import get_brats_dataloaders, BRATS_CLASSES
        loaders = get_brats_dataloaders(data_config)
        class_names = BRATS_CLASSES

    loader = loaders.get(args.split, loaders.get("val"))

    # Run evaluation
    print(f"Evaluating on {args.split} set...")
    tracker = MetricTracker(class_names=class_names)

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            images = batch["images"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]

            output = model(images, input_ids, attention_mask, compute_alignment_loss=False)
            probs = output["probabilities"].cpu().numpy()
            tracker.update(labels.numpy(), probs)

    # Compute metrics
    results = tracker.compute()

    # Print results
    print(format_metrics_table(results))

    # Save results
    results_path = os.path.join(args.output_dir, f"eval_{args.split}_results.json")

    # Convert numpy types for JSON serialization
    def make_serializable(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        return obj

    with open(results_path, "w") as f:
        json.dump(make_serializable(results), f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Plot ROC curves
    all_true = np.concatenate([t for t in tracker._all_true], axis=0)
    all_prob = np.concatenate([p for p in tracker._all_prob], axis=0)
    roc_path = os.path.join(args.output_dir, f"roc_curves_{args.split}.png")
    plot_roc_curves(all_true, all_prob, class_names, output_path=roc_path)
    print(f"ROC curves saved to {roc_path}")


if __name__ == "__main__":
    main()
