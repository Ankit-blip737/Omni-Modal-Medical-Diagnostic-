"""
Logging Utilities
=================
Setup helpers for Python logging, TensorBoard, and model summary printing.
"""

import os
import sys
import logging
from typing import Optional


def setup_logger(
    name: str = "omni_modal",
    log_dir: str = "./logs",
    level: int = logging.INFO,
    log_to_file: bool = True,
) -> logging.Logger:
    """
    Configure a Python logger with console and optional file output.

    Args:
        name: Logger name
        log_dir: Directory for log files
        level: Logging level
        log_to_file: Whether to also log to a file

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s — %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_to_file:
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(log_dir, f"{name}.log"))
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class TensorBoardLogger:
    """
    Wrapper around TensorBoard SummaryWriter with convenience methods.

    Args:
        log_dir: Directory for TensorBoard event files
    """

    def __init__(self, log_dir: str = "./logs/tensorboard"):
        os.makedirs(log_dir, exist_ok=True)
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=log_dir)
            self.enabled = True
        except ImportError:
            print("Warning: TensorBoard not installed. Logging disabled.")
            self.writer = None
            self.enabled = False

    def log_scalar(self, tag: str, value: float, step: int):
        """Log a single scalar value."""
        if self.enabled:
            self.writer.add_scalar(tag, value, step)

    def log_metrics(self, metrics: dict, step: int, prefix: str = ""):
        """
        Log a dictionary of metrics.

        Args:
            metrics: Dictionary of metric_name → value
            step: Global step or epoch
            prefix: Prefix for metric names (e.g., 'train/', 'val/')
        """
        if not self.enabled:
            return
        for key, val in metrics.items():
            if isinstance(val, (int, float)):
                tag = f"{prefix}{key}" if prefix else key
                self.writer.add_scalar(tag, val, step)

    def log_image(self, tag: str, image, step: int):
        """Log an image tensor."""
        if self.enabled:
            self.writer.add_image(tag, image, step)

    def close(self):
        """Flush and close the writer."""
        if self.enabled and self.writer:
            self.writer.flush()
            self.writer.close()


def log_model_summary(model, logger: Optional[logging.Logger] = None):
    """
    Print a summary of model parameters per module.

    Args:
        model: PyTorch model
        logger: Optional logger (prints to stdout if None)
    """
    output = lambda msg: logger.info(msg) if logger else print(msg)

    output("\n" + "=" * 60)
    output("  MODEL PARAMETER SUMMARY")
    output("=" * 60)

    total_params = 0
    total_trainable = 0

    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        total_params += params
        total_trainable += trainable

        pct = trainable / max(params, 1) * 100
        output(f"  {name:25s} | {params:>12,} total | {trainable:>12,} trainable ({pct:.1f}%)")

    output("-" * 60)
    train_pct = total_trainable / max(total_params, 1) * 100
    output(f"  {'TOTAL':25s} | {total_params:>12,} total | {total_trainable:>12,} trainable ({train_pct:.1f}%)")
    output("=" * 60 + "\n")


# --- Quick test ---
if __name__ == "__main__":
    logger = setup_logger("test_logger", log_to_file=False)
    logger.info("Logger initialized successfully.")

    tb = TensorBoardLogger("./logs/test_tb")
    tb.log_scalar("test/metric", 0.95, 0)
    tb.close()
    print("TensorBoard logger test passed.")
