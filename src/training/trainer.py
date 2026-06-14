"""
Omni-Modal Trainer
==================
Main training loop implementing the 3-phase curriculum training strategy:

    Phase 1: Contrastive pre-training (alignment only)
    Phase 2: Fusion training (fusion + classifier + LoRA)
    Phase 3: End-to-end fine-tuning (all parameters, low LR)

Features:
    - Mixed precision training (AMP)
    - Gradient clipping
    - Checkpoint saving/loading
    - TensorBoard logging
    - tqdm progress bars
"""

import os
import time
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional

from src.models.omni_modal import OmniModalFramework
from src.training.losses import CombinedLoss
from src.training.schedulers import get_scheduler


class OmniModalTrainer:
    """
    Trainer for the Omni-Modal Diagnostic Framework.

    Implements 3-phase curriculum training with mixed precision,
    gradient clipping, and comprehensive logging.

    Args:
        model: OmniModalFramework instance
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        config: Training configuration dictionary
        device: Target device ('cuda' or 'cpu')
        log_dir: Directory for TensorBoard logs
        checkpoint_dir: Directory for model checkpoints
    """

    def __init__(
        self,
        model: OmniModalFramework,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        device: str = "cuda",
        log_dir: str = "./logs",
        checkpoint_dir: str = "./checkpoints",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir

        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Loss function
        self.criterion = CombinedLoss(
            num_classes=model.num_classes,
            gamma=2.0,
            lambda_align=config.get("lambda_align", 0.1),
            multi_label=True,
        )

        # Mixed precision
        self.use_amp = config.get("mixed_precision", True) and device == "cuda"
        self.scaler = GradScaler(enabled=self.use_amp)

        # Gradient clipping
        self.gradient_clip_norm = config.get("gradient_clip_norm", 1.0)

        # TensorBoard writer (lazy init)
        self._writer = None

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_metric = 0.0

    @property
    def writer(self):
        """Lazy-init TensorBoard writer."""
        if self._writer is None:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._writer = SummaryWriter(log_dir=self.log_dir)
            except ImportError:
                self._writer = None
        return self._writer

    def _configure_optimizer_phase1(self) -> torch.optim.Optimizer:
        """Phase 1: Only train alignment module projection heads."""
        params = list(self.model.alignment.parameters())
        return torch.optim.AdamW(
            params,
            lr=self.config.get("phase1", {}).get("lr", 1e-4),
            weight_decay=self.config.get("phase1", {}).get("weight_decay", 1e-4),
        )

    def _configure_optimizer_phase2(self) -> torch.optim.Optimizer:
        """Phase 2: Train fusion + classifier + LoRA adapters."""
        param_groups = self.model.get_parameter_groups(
            lr_backbone=1e-6,  # Almost frozen
            lr_fusion=self.config.get("phase2", {}).get("lr", 5e-5),
        )
        return torch.optim.AdamW(
            param_groups,
            weight_decay=self.config.get("phase2", {}).get("weight_decay", 1e-4),
        )

    def _configure_optimizer_phase3(self) -> torch.optim.Optimizer:
        """Phase 3: Fine-tune everything with very low LR."""
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.get("phase3", {}).get("lr", 1e-6),
            weight_decay=self.config.get("phase3", {}).get("weight_decay", 1e-5),
        )

    def train_epoch(
        self,
        optimizer: torch.optim.Optimizer,
        phase: int,
        epoch: int,
    ) -> dict:
        """
        Train for one epoch.

        Args:
            optimizer: Configured optimizer for current phase
            phase: Training phase (1, 2, or 3)
            epoch: Current epoch number

        Returns:
            Dictionary of average training metrics
        """
        self.model.train()
        total_loss = 0.0
        total_cls_loss = 0.0
        total_align_loss = 0.0
        num_batches = 0

        pbar = tqdm(
            self.train_loader,
            desc=f"Phase {phase} | Epoch {epoch}",
            leave=False,
        )

        for batch in pbar:
            # Move batch to device
            images = batch["images"].to(self.device)
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            optimizer.zero_grad()

            # Forward pass with mixed precision
            with autocast(enabled=self.use_amp):
                output = self.model(
                    images=images,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    compute_alignment_loss=True,
                )

                # Compute combined loss
                loss_output = self.criterion(
                    logits=output["logits"],
                    targets=labels,
                    alignment_loss=output.get("alignment_loss"),
                )

                if phase == 1:
                    # Phase 1: only alignment loss
                    loss = output["alignment_loss"]
                else:
                    loss = loss_output["total_loss"]

            # Backward pass
            self.scaler.scale(loss).backward()

            # Gradient clipping
            if self.gradient_clip_norm > 0:
                self.scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip_norm,
                )

            self.scaler.step(optimizer)
            self.scaler.update()

            # Track metrics
            total_loss += loss.item()
            total_cls_loss += loss_output["cls_loss"].item()
            total_align_loss += loss_output["align_loss"].item()
            num_batches += 1
            self.global_step += 1

            # Update progress bar
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "cls": f"{loss_output['cls_loss'].item():.4f}",
                "align": f"{loss_output['align_loss'].item():.4f}",
            })

            # Log to TensorBoard
            if self.writer and self.global_step % self.config.get("log_every_n_steps", 50) == 0:
                self.writer.add_scalar("train/loss", loss.item(), self.global_step)
                self.writer.add_scalar("train/cls_loss", loss_output["cls_loss"].item(), self.global_step)
                self.writer.add_scalar("train/align_loss", loss_output["align_loss"].item(), self.global_step)

                # Log gate values
                if "gate_values" in output:
                    self.writer.add_scalar("train/gate_visual", output["gate_values"]["g_v"], self.global_step)
                    self.writer.add_scalar("train/gate_text", output["gate_values"]["g_t"], self.global_step)

        metrics = {
            "loss": total_loss / max(num_batches, 1),
            "cls_loss": total_cls_loss / max(num_batches, 1),
            "align_loss": total_align_loss / max(num_batches, 1),
        }
        return metrics

    @torch.no_grad()
    def validate(self) -> dict:
        """
        Run validation and compute metrics.

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        all_probs = []
        all_labels = []
        num_batches = 0

        pbar = tqdm(self.val_loader, desc="Validation", leave=False)

        for batch in pbar:
            images = batch["images"].to(self.device)
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            with autocast(enabled=self.use_amp):
                output = self.model(
                    images=images,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    compute_alignment_loss=True,
                )
                loss_output = self.criterion(
                    logits=output["logits"],
                    targets=labels,
                    alignment_loss=output.get("alignment_loss"),
                )

            total_loss += loss_output["total_loss"].item()
            all_probs.append(output["probabilities"].cpu())
            all_labels.append(labels.cpu())
            num_batches += 1

        # Compute AUC and other metrics
        all_probs = torch.cat(all_probs, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()

        metrics = {"val_loss": total_loss / max(num_batches, 1)}

        try:
            from sklearn.metrics import roc_auc_score, f1_score
            # Per-class AUC (only for classes that appear)
            valid_classes = all_labels.sum(axis=0) > 0
            if valid_classes.any():
                auc = roc_auc_score(
                    all_labels[:, valid_classes],
                    all_probs[:, valid_classes],
                    average="macro",
                )
                metrics["val_auc"] = auc

            # F1 score
            preds = (all_probs > 0.5).astype(int)
            f1 = f1_score(all_labels, preds, average="macro", zero_division=0)
            metrics["val_f1"] = f1
        except Exception:
            pass

        # Log to TensorBoard
        if self.writer:
            for key, val in metrics.items():
                self.writer.add_scalar(f"val/{key}", val, self.current_epoch)

        return metrics

    def _run_phase(self, phase: int, epochs: int):
        """Run a complete training phase."""
        print(f"\n{'='*60}")
        print(f"  PHASE {phase} — {'Contrastive Pre-training' if phase == 1 else 'Fusion Training' if phase == 2 else 'End-to-End Fine-tuning'}")
        print(f"  Epochs: {epochs}")
        print(f"{'='*60}\n")

        # Configure optimizer
        if phase == 1:
            optimizer = self._configure_optimizer_phase1()
        elif phase == 2:
            optimizer = self._configure_optimizer_phase2()
        else:
            optimizer = self._configure_optimizer_phase3()

        # Configure scheduler
        phase_config = self.config.get(f"phase{phase}", {})
        scheduler = get_scheduler(
            optimizer,
            scheduler_type="cosine",
            warmup_epochs=phase_config.get("warmup_epochs", 5),
            total_epochs=epochs,
        )

        for epoch in range(epochs):
            self.current_epoch += 1

            # Train
            train_metrics = self.train_epoch(optimizer, phase, self.current_epoch)
            scheduler.step()

            # Validate
            val_metrics = self.validate()

            # Print epoch summary
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"  Epoch {self.current_epoch:3d} | "
                f"LR: {lr:.2e} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Val Loss: {val_metrics['val_loss']:.4f} | "
                f"Val AUC: {val_metrics.get('val_auc', 0):.4f} | "
                f"Val F1: {val_metrics.get('val_f1', 0):.4f}"
            )

            # Save best checkpoint
            current_metric = val_metrics.get("val_auc", -val_metrics["val_loss"])
            if current_metric > self.best_val_metric:
                self.best_val_metric = current_metric
                self.save_checkpoint(
                    os.path.join(self.checkpoint_dir, f"best_phase{phase}.pth"),
                    phase=phase,
                    epoch=self.current_epoch,
                    metrics=val_metrics,
                )
                print(f"  ✓ New best model saved (metric: {current_metric:.4f})")

        # Save final phase checkpoint
        self.save_checkpoint(
            os.path.join(self.checkpoint_dir, f"final_phase{phase}.pth"),
            phase=phase,
            epoch=self.current_epoch,
            metrics=val_metrics,
        )

    def train_phase1(self, epochs: Optional[int] = None):
        """Phase 1: Contrastive pre-training."""
        epochs = epochs or self.config.get("phase1", {}).get("epochs", 50)
        self._run_phase(1, epochs)

    def train_phase2(self, epochs: Optional[int] = None):
        """Phase 2: Fusion training."""
        epochs = epochs or self.config.get("phase2", {}).get("epochs", 100)
        self._run_phase(2, epochs)

    def train_phase3(self, epochs: Optional[int] = None):
        """Phase 3: End-to-end fine-tuning."""
        epochs = epochs or self.config.get("phase3", {}).get("epochs", 20)
        self._run_phase(3, epochs)

    def train_all_phases(self):
        """Run all 3 training phases sequentially."""
        self.train_phase1()
        self.train_phase2()
        self.train_phase3()

    def save_checkpoint(self, path: str, **extra_state):
        """Save model checkpoint with training state."""
        state = {
            "model_state_dict": self.model.state_dict(),
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "best_val_metric": self.best_val_metric,
            "config": self.config,
            **extra_state,
        }
        torch.save(state, path)

    def load_checkpoint(self, path: str):
        """Load model checkpoint and restore training state."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.current_epoch = checkpoint.get("epoch", 0)
        self.global_step = checkpoint.get("global_step", 0)
        self.best_val_metric = checkpoint.get("best_val_metric", 0.0)
        print(f"Loaded checkpoint from {path} (epoch {self.current_epoch})")


# --- Quick test ---
if __name__ == "__main__":
    print("OmniModalTrainer loaded successfully.")
    print("Usage:")
    print("  trainer = OmniModalTrainer(model, train_loader, val_loader, config)")
    print("  trainer.train_all_phases()")
