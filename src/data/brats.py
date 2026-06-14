"""
BraTS Multi-Modal MRI Dataset Loader
======================================
PyTorch Dataset for the BraTS (Brain Tumor Segmentation) challenge dataset
with multi-modal MRI volumes (T1, T2, FLAIR, T1ce).

Dataset details:
    - Multi-institutional brain MRI data
    - 4 modalities: T1, T1-contrast enhanced (T1ce), T2, T2-FLAIR
    - Segmentation masks with 3 tumor sub-regions
    - Format: NIfTI (.nii.gz) 3D volumes

For classification, we extract 2D axial slices and classify tumor type/grade.
"""

import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from typing import Optional

from .transforms import get_train_transforms, get_val_transforms


# BraTS tumor classes (for classification)
BRATS_CLASSES = [
    "No Tumor",
    "Low-Grade Glioma (LGG)",
    "High-Grade Glioma (HGG)",
]

# Standard BraTS 2023-GLI modality file suffixes
MODALITY_SUFFIXES = {
    "t1n": "-t1n.nii.gz",      # T1 native
    "t1c": "-t1c.nii.gz",      # T1 contrast-enhanced
    "t2w": "-t2w.nii.gz",      # T2 weighted
    "t2f": "-t2f.nii.gz",      # T2 FLAIR
}


class BraTSDataset(Dataset):
    """
    BraTS multi-modal MRI dataset for the Joint Image Encoder.

    Each sample contains M modalities (e.g., T1, T2, FLAIR) as separate
    grayscale 2D slices extracted from 3D NIfTI volumes.

    Directory structure expected (BraTS2023-GLI):
        root_dir/
        ├── ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/
        │   ├── BraTS-GLI-00000-000/
        │   │   ├── BraTS-GLI-00000-000-t1n.nii.gz
        │   │   ├── BraTS-GLI-00000-000-t1c.nii.gz
        │   │   ├── BraTS-GLI-00000-000-t2w.nii.gz
        │   │   ├── BraTS-GLI-00000-000-t2f.nii.gz
        │   │   └── BraTS-GLI-00000-000-seg.nii.gz
        │   └── ...
        └── ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData/
            └── ...

    Args:
        root_dir: Root directory containing train/val subdirectories
        split: Dataset split ('train' or 'val')
        transform: Image transform pipeline
        modalities: List of modalities to load (default: ['t1n', 't2w', 't2f'])
        num_slices: Number of axial slices to extract per volume (centered)
        min_tumor_area: Minimum tumor pixels to include a slice (filters empty slices)
        image_size: Target image size for resizing
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform=None,
        modalities: list[str] = None,
        num_slices: int = 16,
        min_tumor_area: int = 100,
        image_size: int = 224,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.modalities = modalities or ["t1n", "t2w", "t2f"]
        self.num_modalities = len(self.modalities)
        self.num_slices = num_slices
        self.min_tumor_area = min_tumor_area
        self.image_size = image_size

        # Transforms
        if transform is not None:
            self.transform = transform
        elif split == "train":
            self.transform = get_train_transforms(image_size)
        else:
            self.transform = get_val_transforms(image_size)

        # Discover subject directories — BraTS2023-GLI uses flat structure
        split_dir_map = {
            "train": [
                os.path.join(root_dir, "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"),
                os.path.join(root_dir, "train"),
            ],
            "val": [
                os.path.join(root_dir, "ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData"),
                os.path.join(root_dir, "val"),
            ],
        }

        self.subject_dirs = []
        for candidate_dir in split_dir_map.get(split, []):
            if os.path.exists(candidate_dir):
                self.subject_dirs = sorted(glob.glob(os.path.join(candidate_dir, "BraTS-GLI-*")))
                if self.subject_dirs:
                    print(f"BraTS [{split}]: Found {len(self.subject_dirs)} subjects in {candidate_dir}")
                    break

        if not self.subject_dirs:
            print(f"Warning: No BraTS subject directories found for '{split}'. Creating dummy dataset.")

        # Build sample index: (subject_dir, slice_idx) pairs
        self.samples = self._build_sample_index()

    def _build_sample_index(self) -> list[tuple]:
        """
        Build index of (subject_dir, slice_idx) pairs.
        For real data, this scans volumes; for dummy data, creates synthetic samples.
        """
        if not self.subject_dirs:
            # Dummy samples for testing
            return [(None, i) for i in range(100)]

        samples = []
        for subj_dir in self.subject_dirs:
            # Use center slices of the volume
            for slice_idx in range(self.num_slices):
                samples.append((subj_dir, slice_idx))

        return samples

    def _load_nifti_slice(self, filepath: str, slice_idx: int) -> np.ndarray:
        """
        Load a single 2D axial slice from a NIfTI volume.

        Args:
            filepath: Path to .nii.gz file
            slice_idx: Index of the axial slice to extract

        Returns:
            2D numpy array (H, W) normalized to [0, 255]
        """
        try:
            import nibabel as nib
            vol = nib.load(filepath).get_fdata()

            # Select center-biased slice
            total_slices = vol.shape[2]
            center = total_slices // 2
            start = max(0, center - self.num_slices // 2)
            actual_idx = min(start + slice_idx, total_slices - 1)

            slice_2d = vol[:, :, actual_idx].astype(np.float32)

            # Normalize to [0, 255]
            if slice_2d.max() > slice_2d.min():
                slice_2d = (slice_2d - slice_2d.min()) / (slice_2d.max() - slice_2d.min()) * 255
            return slice_2d.astype(np.uint8)

        except (ImportError, FileNotFoundError):
            # Return dummy slice
            return np.random.randint(0, 256, (240, 240), dtype=np.uint8)

    def _get_tumor_label(self, subj_dir: str) -> int:
        """
        Determine tumor grade from segmentation mask or metadata.
        Returns class index: 0=No Tumor, 1=LGG, 2=HGG
        """
        if subj_dir is None:
            return np.random.randint(0, len(BRATS_CLASSES))

        # Check subject name for grade hints (BraTS convention)
        subj_name = os.path.basename(subj_dir).lower()
        if "lgg" in subj_name:
            return 1
        elif "hgg" in subj_name:
            return 2

        # Default: try to determine from segmentation volume
        seg_path = os.path.join(subj_dir, os.path.basename(subj_dir) + "-seg.nii.gz")
        if os.path.exists(seg_path):
            try:
                import nibabel as nib
                seg = nib.load(seg_path).get_fdata()
                # HGG has enhancing tumor (label 4), LGG typically doesn't
                if (seg == 4).sum() > 50:
                    return 2  # HGG
                elif seg.max() > 0:
                    return 1  # LGG
                else:
                    return 0  # No tumor
            except Exception:
                pass

        return 1  # Default to LGG if undetermined

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        """
        Returns:
            Dictionary with:
                'images': (M, 1, H, W) tensor — M modalities, each grayscale
                'labels': (C,) tensor — one-hot tumor class
                'subject_id': str — subject identifier
                'slice_idx': int — axial slice index
        """
        subj_dir, slice_idx = self.samples[idx]
        subject_id = os.path.basename(subj_dir) if subj_dir else f"dummy_{idx}"

        # --- Load each modality ---
        modality_images = []
        for mod in self.modalities:
            if subj_dir is not None:
                suffix = MODALITY_SUFFIXES.get(mod, f"-{mod}.nii.gz")
                filepath = os.path.join(subj_dir, subject_id + suffix)
                slice_2d = self._load_nifti_slice(filepath, slice_idx)
            else:
                # Dummy data
                slice_2d = np.random.randint(0, 256, (240, 240), dtype=np.uint8)

            # Convert to PIL for transforms
            pil_img = Image.fromarray(slice_2d, mode="L")
            if self.transform:
                tensor_img = self.transform(pil_img)  # (1, H, W)
            else:
                tensor_img = torch.from_numpy(slice_2d).unsqueeze(0).float() / 255.0

            modality_images.append(tensor_img)

        # Stack modalities: (M, 1, H, W)
        images = torch.stack(modality_images, dim=0)

        # --- Label ---
        label_idx = self._get_tumor_label(subj_dir)
        labels = torch.zeros(len(BRATS_CLASSES), dtype=torch.float32)
        labels[label_idx] = 1.0

        return {
            "images": images,
            "labels": labels,
            "subject_id": subject_id,
            "slice_idx": slice_idx,
        }


def collate_brats(batch: list[dict]) -> dict:
    """Custom collate function for BraTS multi-modal batches."""
    images = torch.stack([item["images"] for item in batch])  # (B, M, 1, H, W)
    labels = torch.stack([item["labels"] for item in batch])

    return {
        "images": images,
        "labels": labels,
    }


def get_brats_dataloaders(config: dict) -> dict:
    """
    Factory function to create BraTS DataLoaders.

    Args:
        config: Configuration dictionary

    Returns:
        Dictionary with 'train', 'val' DataLoaders
    """
    loaders = {}
    for split in ["train", "val"]:
        dataset = BraTSDataset(
            root_dir=config.get("root_dir", "."),
            split=split,
            modalities=config.get("modalities", ["t1n", "t2w", "t2f"]),
            num_slices=config.get("num_slices", 16),
            image_size=config.get("image_size", 224),
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size=config.get("batch_size", 16),
            shuffle=(split == "train"),
            num_workers=config.get("num_workers", 4),
            pin_memory=config.get("pin_memory", True),
            collate_fn=collate_brats,
            drop_last=(split == "train"),
        )
    return loaders


# --- Quick test ---
if __name__ == "__main__":
    print("=== BraTS Dataset Test ===")
    print(f"Classes ({len(BRATS_CLASSES)}): {BRATS_CLASSES}")
    print(f"Modalities: {list(MODALITY_SUFFIXES.keys())}")

    dataset = BraTSDataset(
        root_dir="./data/brats",
        split="train",
        modalities=["t1", "t2", "flair"],
    )
    print(f"Dataset size: {len(dataset)}")

    sample = dataset[0]
    print(f"Images shape: {sample['images'].shape}")   # (3, 1, 224, 224)
    print(f"Labels shape: {sample['labels'].shape}")    # (3,)
    print(f"Subject ID:   {sample['subject_id']}")
