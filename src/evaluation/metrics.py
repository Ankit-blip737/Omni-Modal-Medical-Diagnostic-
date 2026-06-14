"""
Evaluation Metrics
==================
Comprehensive medical imaging classification metrics including:
    - Per-class AUC-ROC, F1, precision, recall, specificity
    - Macro/micro averaging
    - MetricTracker for batch accumulation
"""

import numpy as np
from typing import Optional
from collections import defaultdict


def compute_multilabel_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: Optional[list[str]] = None,
    threshold: float = 0.5,
) -> dict:
    """
    Compute comprehensive multi-label classification metrics.

    Args:
        y_true: (N, C) ground truth binary labels
        y_pred: (N, C) predicted binary labels
        y_prob: (N, C) predicted probabilities
        class_names: List of class names for per-class reporting
        threshold: Classification threshold for probabilities

    Returns:
        Dictionary with per-class and aggregate metrics
    """
    from sklearn.metrics import (
        roc_auc_score,
        f1_score,
        precision_score,
        recall_score,
        average_precision_score,
    )

    N, C = y_true.shape
    if class_names is None:
        class_names = [f"class_{i}" for i in range(C)]

    if y_pred is None:
        y_pred = (y_prob >= threshold).astype(int)

    metrics = {"per_class": {}, "aggregate": {}}

    # Per-class metrics
    for i, name in enumerate(class_names):
        cls_metrics = {}
        true_i = y_true[:, i]
        pred_i = y_pred[:, i]
        prob_i = y_prob[:, i]

        # Only compute AUC if both classes present
        if len(np.unique(true_i)) > 1:
            cls_metrics["auc_roc"] = float(roc_auc_score(true_i, prob_i))
            cls_metrics["avg_precision"] = float(average_precision_score(true_i, prob_i))
        else:
            cls_metrics["auc_roc"] = float("nan")
            cls_metrics["avg_precision"] = float("nan")

        # Confusion matrix elements
        tp = ((pred_i == 1) & (true_i == 1)).sum()
        tn = ((pred_i == 0) & (true_i == 0)).sum()
        fp = ((pred_i == 1) & (true_i == 0)).sum()
        fn = ((pred_i == 0) & (true_i == 1)).sum()

        cls_metrics["precision"] = float(tp / max(tp + fp, 1))
        cls_metrics["recall"] = float(tp / max(tp + fn, 1))  # Sensitivity
        cls_metrics["specificity"] = float(tn / max(tn + fp, 1))
        cls_metrics["f1"] = float(2 * tp / max(2 * tp + fp + fn, 1))
        cls_metrics["support"] = int(true_i.sum())

        metrics["per_class"][name] = cls_metrics

    # Aggregate metrics
    valid_mask = y_true.sum(axis=0) > 0
    if valid_mask.any():
        metrics["aggregate"]["macro_auc"] = float(
            roc_auc_score(y_true[:, valid_mask], y_prob[:, valid_mask], average="macro")
        )
        try:
            metrics["aggregate"]["micro_auc"] = float(
                roc_auc_score(y_true[:, valid_mask], y_prob[:, valid_mask], average="micro")
            )
        except Exception:
            metrics["aggregate"]["micro_auc"] = float("nan")

    metrics["aggregate"]["macro_f1"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    metrics["aggregate"]["micro_f1"] = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
    metrics["aggregate"]["macro_precision"] = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    metrics["aggregate"]["macro_recall"] = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

    return metrics


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[list[str]] = None,
) -> dict:
    """
    Compute per-class confusion matrix elements.

    Args:
        y_true: (N, C) ground truth
        y_pred: (N, C) predictions
        class_names: Class names

    Returns:
        Dictionary with TP, TN, FP, FN per class
    """
    C = y_true.shape[1]
    if class_names is None:
        class_names = [f"class_{i}" for i in range(C)]

    result = {}
    for i, name in enumerate(class_names):
        true_i = y_true[:, i]
        pred_i = y_pred[:, i]
        result[name] = {
            "TP": int(((pred_i == 1) & (true_i == 1)).sum()),
            "TN": int(((pred_i == 0) & (true_i == 0)).sum()),
            "FP": int(((pred_i == 1) & (true_i == 0)).sum()),
            "FN": int(((pred_i == 0) & (true_i == 1)).sum()),
        }
    return result


class MetricTracker:
    """
    Accumulates predictions over batches and computes final metrics.

    Usage:
        tracker = MetricTracker(class_names)
        for batch in loader:
            tracker.update(labels, probabilities)
        results = tracker.compute()

    Args:
        class_names: List of class names
        threshold: Classification threshold
    """

    def __init__(
        self,
        class_names: Optional[list[str]] = None,
        threshold: float = 0.5,
    ):
        self.class_names = class_names
        self.threshold = threshold
        self.reset()

    def reset(self):
        """Clear all accumulated predictions."""
        self._all_true = []
        self._all_prob = []
        self._losses = defaultdict(list)

    def update(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        losses: Optional[dict] = None,
    ):
        """
        Add a batch of predictions.

        Args:
            y_true: (B, C) ground truth
            y_prob: (B, C) predicted probabilities
            losses: Optional dict of loss values for this batch
        """
        self._all_true.append(y_true)
        self._all_prob.append(y_prob)

        if losses:
            for key, val in losses.items():
                self._losses[key].append(val)

    def compute(self) -> dict:
        """
        Compute all metrics from accumulated predictions.

        Returns:
            Dictionary with all metrics + average losses
        """
        if not self._all_true:
            return {}

        y_true = np.concatenate(self._all_true, axis=0)
        y_prob = np.concatenate(self._all_prob, axis=0)
        y_pred = (y_prob >= self.threshold).astype(int)

        metrics = compute_multilabel_metrics(
            y_true, y_pred, y_prob,
            class_names=self.class_names,
            threshold=self.threshold,
        )

        # Add average losses
        metrics["avg_losses"] = {
            key: float(np.mean(vals)) for key, vals in self._losses.items()
        }

        return metrics

    @property
    def num_samples(self) -> int:
        """Total number of accumulated samples."""
        return sum(t.shape[0] for t in self._all_true)


def format_metrics_table(metrics: dict) -> str:
    """Format metrics dictionary as a readable table string."""
    lines = []
    lines.append(f"\n{'Class':30s} | {'AUC':>7s} | {'F1':>7s} | {'Prec':>7s} | {'Rec':>7s} | {'Spec':>7s} | {'N':>5s}")
    lines.append("-" * 85)

    for name, m in metrics.get("per_class", {}).items():
        lines.append(
            f"{name:30s} | {m['auc_roc']:7.4f} | {m['f1']:7.4f} | "
            f"{m['precision']:7.4f} | {m['recall']:7.4f} | "
            f"{m['specificity']:7.4f} | {m['support']:5d}"
        )

    lines.append("-" * 85)
    agg = metrics.get("aggregate", {})
    lines.append(f"{'MACRO AVERAGE':30s} | {agg.get('macro_auc', 0):7.4f} | {agg.get('macro_f1', 0):7.4f} |")
    lines.append(f"{'MICRO AVERAGE':30s} | {agg.get('micro_auc', 0):7.4f} | {agg.get('micro_f1', 0):7.4f} |")

    return "\n".join(lines)


# --- Quick test ---
if __name__ == "__main__":
    np.random.seed(42)
    N, C = 100, 5
    y_true = np.random.randint(0, 2, (N, C))
    y_prob = np.random.rand(N, C)
    y_pred = (y_prob > 0.5).astype(int)

    class_names = ["Disease A", "Disease B", "Disease C", "Disease D", "Disease E"]

    metrics = compute_multilabel_metrics(y_true, y_pred, y_prob, class_names)
    print(format_metrics_table(metrics))

    # Test MetricTracker
    tracker = MetricTracker(class_names)
    tracker.update(y_true[:50], y_prob[:50])
    tracker.update(y_true[50:], y_prob[50:])
    result = tracker.compute()
    print(f"\nTracker samples: {tracker.num_samples}")
    print(f"Macro AUC: {result['aggregate']['macro_auc']:.4f}")
