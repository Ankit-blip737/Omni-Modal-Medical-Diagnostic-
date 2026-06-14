"""
Visualization Utilities
========================
Tools for visualizing attention maps, gate values, training curves,
and ROC curves for the Omni-Modal Diagnostic Framework.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/script use
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import Optional


def visualize_cross_attention(
    image: np.ndarray,
    attention_weights: np.ndarray,
    title: str = "Cross-Attention Heatmap",
    output_path: Optional[str] = None,
    figsize: tuple = (12, 5),
    cmap: str = "jet",
    alpha: float = 0.5,
) -> plt.Figure:
    """
    Overlay cross-attention heatmap on a medical image.

    Args:
        image: (H, W) grayscale image array
        attention_weights: (N_patches,) or (H_p, W_p) attention weights
        title: Plot title
        output_path: Path to save figure (if None, returns figure)
        figsize: Figure size
        cmap: Colormap for attention overlay
        alpha: Opacity of attention overlay

    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Original image
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Attention heatmap
    if attention_weights.ndim == 1:
        side = int(np.sqrt(attention_weights.shape[0]))
        attn_map = attention_weights.reshape(side, side)
    else:
        attn_map = attention_weights

    # Resize attention map to match image
    from PIL import Image as PILImage
    attn_resized = np.array(
        PILImage.fromarray(
            ((attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8) * 255).astype(np.uint8)
        ).resize((image.shape[1], image.shape[0]), PILImage.BILINEAR)
    ).astype(np.float32) / 255.0

    axes[1].imshow(attn_resized, cmap=cmap)
    axes[1].set_title("Attention Map")
    axes[1].axis("off")

    # Overlay
    axes[2].imshow(image, cmap="gray")
    axes[2].imshow(attn_resized, cmap=cmap, alpha=alpha)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig


def visualize_gate_values(
    gate_v: np.ndarray,
    gate_t: np.ndarray,
    sample_ids: Optional[list[str]] = None,
    output_path: Optional[str] = None,
    figsize: tuple = (10, 5),
) -> plt.Figure:
    """
    Bar chart comparing visual vs text modality gate activations.

    Args:
        gate_v: (N,) visual gate values per sample
        gate_t: (N,) text gate values per sample
        sample_ids: Labels for each sample
        output_path: Path to save figure
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    N = len(gate_v)
    if sample_ids is None:
        sample_ids = [f"Sample {i}" for i in range(N)]

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(N)
    width = 0.35

    bars_v = ax.bar(x - width / 2, gate_v, width, label="Visual Gate (g_v)", color="#4A90D9", alpha=0.85)
    bars_t = ax.bar(x + width / 2, gate_t, width, label="Text Gate (g_t)", color="#E74C3C", alpha=0.85)

    ax.set_xlabel("Samples")
    ax.set_ylabel("Gate Activation")
    ax.set_title("Dynamic Modality Weighting — Gate Values", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(sample_ids, rotation=45, ha="right", fontsize=8)
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_training_curves(
    train_losses: list[float],
    val_losses: list[float],
    val_aucs: Optional[list[float]] = None,
    output_path: Optional[str] = None,
    figsize: tuple = (12, 5),
) -> plt.Figure:
    """
    Plot training and validation loss/metric curves.

    Args:
        train_losses: Per-epoch training losses
        val_losses: Per-epoch validation losses
        val_aucs: Per-epoch validation AUC scores
        output_path: Path to save figure
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    n_plots = 2 if val_aucs else 1
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    if n_plots == 1:
        axes = [axes]

    epochs = range(1, len(train_losses) + 1)

    # Loss curves
    axes[0].plot(epochs, train_losses, "b-", label="Train Loss", linewidth=2)
    axes[0].plot(epochs, val_losses, "r-", label="Val Loss", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss", fontweight="bold")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # AUC curve
    if val_aucs:
        axes[1].plot(epochs[:len(val_aucs)], val_aucs, "g-", label="Val AUC", linewidth=2)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("AUC-ROC")
        axes[1].set_title("Validation AUC-ROC", fontweight="bold")
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        axes[1].set_ylim(0, 1)

    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str],
    output_path: Optional[str] = None,
    figsize: tuple = (10, 8),
) -> plt.Figure:
    """
    Plot multi-class ROC curves with per-class AUC.

    Args:
        y_true: (N, C) ground truth labels
        y_prob: (N, C) predicted probabilities
        class_names: List of class names
        output_path: Path to save figure
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    from sklearn.metrics import roc_curve, auc

    fig, ax = plt.subplots(figsize=figsize)

    colors = cm.tab20(np.linspace(0, 1, len(class_names)))

    for i, (name, color) in enumerate(zip(class_names, colors)):
        if len(np.unique(y_true[:, i])) < 2:
            continue

        fpr, tpr, _ = roc_curve(y_true[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)

        ax.plot(fpr, tpr, color=color, linewidth=2, label=f"{name} (AUC={roc_auc:.3f})")

    # Diagonal
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random")

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — Multi-Label Classification", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig


# --- Quick test ---
if __name__ == "__main__":
    # Test attention visualization
    dummy_image = np.random.randint(0, 256, (224, 224), dtype=np.uint8)
    dummy_attn = np.random.rand(49)  # 7x7 patches
    visualize_cross_attention(dummy_image, dummy_attn, output_path="./test_attn.png")
    print("Attention visualization saved to ./test_attn.png")

    # Test gate values
    gate_v = np.random.rand(8) * 0.3 + 0.4
    gate_t = np.random.rand(8) * 0.3 + 0.3
    visualize_gate_values(gate_v, gate_t, output_path="./test_gates.png")
    print("Gate visualization saved to ./test_gates.png")

    # Test training curves
    train_l = [2.0 - 0.03 * i + np.random.randn() * 0.05 for i in range(50)]
    val_l = [2.1 - 0.025 * i + np.random.randn() * 0.08 for i in range(50)]
    val_a = [0.5 + 0.008 * i + np.random.randn() * 0.02 for i in range(50)]
    plot_training_curves(train_l, val_l, val_a, output_path="./test_curves.png")
    print("Training curves saved to ./test_curves.png")
