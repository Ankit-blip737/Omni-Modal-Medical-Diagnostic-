#!/usr/bin/env python3
"""
Training Entry Point
====================
Main script to train the Omni-Modal Diagnostic Framework.

Usage:
    # Run all 3 phases:
    python scripts/train.py --config configs/default.yaml

    # Run specific phase:
    python scripts/train.py --config configs/default.yaml --phase 2

    # Resume from checkpoint:
    python scripts/train.py --config configs/default.yaml --resume checkpoints/best_phase1.pth --phase 2

    # Smoke test (1 batch):
    python scripts/train.py --config configs/default.yaml --smoke-test
"""

import os
import sys
import argparse
import yaml
import torch
import random
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.omni_modal import OmniModalFramework
from src.training.trainer import OmniModalTrainer
from src.utils.logging import setup_logger, log_model_summary


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def create_dataloaders(config: dict) -> tuple:
    """Create train and validation DataLoaders based on config."""
    data_config = config.get("data", {})
    dataset_type = data_config.get("dataset", "mimic_cxr")

    if dataset_type == "mimic_cxr":
        from src.data.mimic_cxr import get_mimic_dataloaders
        loaders = get_mimic_dataloaders(data_config)
        return loaders.get("train"), loaders.get("val")
    elif dataset_type == "brats":
        from src.data.brats import get_brats_dataloaders
        loaders = get_brats_dataloaders(data_config)
        return loaders.get("train"), loaders.get("val")
    else:
        raise ValueError(f"Unknown dataset: {dataset_type}")


def main():
    parser = argparse.ArgumentParser(description="Train Omni-Modal Diagnostic Framework")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML")
    parser.add_argument("--phase", type=int, default=0, choices=[0, 1, 2, 3],
                        help="Training phase (0=all, 1/2/3=specific phase)")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--smoke-test", action="store_true", help="Run 1 batch for validation")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    training_config = config.get("training", {})
    model_config = config.get("model", {})

    # Set seed
    set_seed(training_config.get("seed", 42))

    # Device
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Logger
    logger = setup_logger("train", config.get("logging", {}).get("log_dir", "./logs"))

    # Create model
    logger.info("Building model...")
    model = OmniModalFramework(
        num_modalities=model_config.get("num_modalities", 3),
        num_classes=model_config.get("num_classes", 14),
        text_model=model_config.get("text_model", "pubmedbert"),
        freeze_image_backbone=model_config.get("freeze_image_backbone", True),
        freeze_text_layers=model_config.get("freeze_text_layers", 8),
        lora_rank=model_config.get("lora_rank", 8),
        fusion_dim=model_config.get("fusion_dim", 768),
        projection_dim=model_config.get("projection_dim", 256),
        num_fusion_stages=model_config.get("num_fusion_stages", 1),
        modality_dropout_p=model_config.get("modality_dropout_p", 0.15),
        multi_label=model_config.get("multi_label", True),
        pretrained=not args.smoke_test,  # Skip download for smoke test
    )
    log_model_summary(model, logger)

    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(config)

    # Create trainer
    trainer = OmniModalTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config,
        device=device,
        log_dir=config.get("logging", {}).get("log_dir", "./logs"),
        checkpoint_dir=config.get("logging", {}).get("checkpoint_dir", "./checkpoints"),
    )

    # Resume from checkpoint
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Smoke test
    if args.smoke_test:
        logger.info("Running smoke test (1 batch)...")
        # Override to minimal settings for CPU testing
        config["data"]["batch_size"] = 2
        config["data"]["num_workers"] = 0
        config["data"]["pin_memory"] = False
        train_loader, val_loader = create_dataloaders(config)
        trainer.train_loader = train_loader
        trainer.val_loader = val_loader
        trainer.config["phase1"] = {"epochs": 1, "lr": 1e-4, "warmup_epochs": 0}
        trainer.train_phase1(epochs=1)
        logger.info("Smoke test passed!")
        return

    # Run training
    if args.phase == 0:
        logger.info("Starting all training phases...")
        trainer.train_all_phases()
    elif args.phase == 1:
        trainer.train_phase1()
    elif args.phase == 2:
        trainer.train_phase2()
    elif args.phase == 3:
        trainer.train_phase3()

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
