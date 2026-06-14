"""
CheXmultimodal Dataset Loader
===============================
PyTorch Dataset for the CheXmultimodal dataset from Stanford University.

Dataset details:
    - 324 patient studies from Stanford University Hospital
    - Each study contains: chest X-ray (JPG) + clinical indication text
    - Indication format: "68 years of age, Female, chest pain"
    - Size: ~489 MB
    - License: Stanford University Dataset Research Use Agreement

Download:
    https://stanfordaimi.azurewebsites.net/datasets/3a7548a4-8f65-4ab7-85fa-3d68c9efc1bd
    (or via Redivis: https://doi.org/10.57761/j8hp-mh04)

Expected directory structure after download:
    chexmultimodal/
    ├── new_indications_324.csv     # Columns: DESCRIPTION, indication, comparison, patient_id
    └── new_patient_images_324/
        ├── 0.jpg
        ├── 1.jpg
        └── ...
"""

import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from typing import Optional
from sklearn.model_selection import train_test_split

from .transforms import get_train_transforms, get_val_transforms
from .text_utils import ClinicalTextPreprocessor


# Common findings to detect from indications (binary classification targets)
# Since CheXmultimodal doesn't have explicit pathology labels, we derive
# them from the indication text or use the model for feature learning.
CHEX_MULTIMODAL_LABELS = [
    "Chest Pain",
    "Shortness of Breath",
    "Cough",
    "Fever",
    "Post-surgical",
    "Trauma",
    "Follow-up",
    "Pneumonia",
    "Effusion",
    "Other",
]


def _extract_labels_from_indication(indication: str) -> np.ndarray:
    """
    Extract pseudo-labels from clinical indication text.
    Maps keywords to binary labels for training.

    Args:
        indication: Clinical indication string

    Returns:
        Binary label array of shape (num_labels,)
    """
    labels = np.zeros(len(CHEX_MULTIMODAL_LABELS), dtype=np.float32)
    text = indication.lower() if indication else ""

    keyword_map = {
        0: ["chest pain", "pain", "angina", "discomfort", "cp ", " cp"],
        1: ["shortness of breath", "dyspnea", "sob", "respiratory distress", "breathing"],
        2: ["cough", "coughing"],
        3: ["fever", "febrile", "temperature"],
        4: ["post-surgical", "post-op", "surgery", "operative", "postoperative", "post op", "cabg", "stent"],
        5: ["trauma", "injury", "fall", "accident", "fracture"],
        6: ["follow-up", "follow up", "followup", "routine", "interval", "recheck"],
        7: ["pneumonia", "pna", "infection", "infiltrate", "consolidation"],
        8: ["effusion", "pleural", "fluid", "ptx", "pneumothorax"],
        9: [],  # "Other" — set if no other label matched
    }

    any_matched = False
    for label_idx, keywords in keyword_map.items():
        for kw in keywords:
            if kw in text:
                labels[label_idx] = 1.0
                any_matched = True
                break

    # If no specific label matched, mark as "Other"
    if not any_matched:
        labels[-1] = 1.0

    return labels


class CheXmultimodalDataset(Dataset):
    """
    CheXmultimodal dataset: 324 chest X-rays + clinical indications.

    This dataset is ideal for:
        - Pre-training visual-semantic alignment (contrastive learning)
        - Testing the multi-modal fusion pipeline
        - Proof-of-concept demonstrations

    Args:
        root_dir: Path to the CheXmultimodal dataset directory
        split: 'train', 'val', or 'test' (auto-split from single CSV)
        transform: Image transform pipeline
        max_text_length: Maximum tokenized text length
        tokenizer_name: HuggingFace tokenizer model name
        train_ratio: Fraction for training split
        val_ratio: Fraction for validation split
        seed: Random seed for reproducible splits
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform=None,
        max_text_length: int = 128,
        tokenizer_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        seed: int = 42,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.max_text_length = max_text_length
        self.num_classes = len(CHEX_MULTIMODAL_LABELS)

        # Transforms
        if transform is not None:
            self.transform = transform
        elif split == "train":
            self.transform = get_train_transforms(224)
        else:
            self.transform = get_val_transforms(224)

        # Text preprocessor (lightweight for short indications)
        self.text_preprocessor = ClinicalTextPreprocessor(
            max_length=max_text_length,
            sections_to_use="all",  # Indications are short, use full text
        )

        # Lazy tokenizer
        self._tokenizer = None
        self._tokenizer_name = tokenizer_name

        # Load and split metadata
        self.metadata = self._load_and_split(train_ratio, val_ratio, seed)
        print(f"CheXmultimodal [{split}]: {len(self.metadata)} samples")

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self._tokenizer_name)
        return self._tokenizer

    def _load_and_split(
        self,
        train_ratio: float,
        val_ratio: float,
        seed: int,
    ) -> pd.DataFrame:
        """
        Load dataset CSV and split into train/val/test.
        Searches for common CSV filenames in the root directory.
        """
        # Try common CSV names (CheXmultimodal uses new_indications_324.csv)
        csv_candidates = [
            "new_indications_324.csv",
            "dataset.csv", "metadata.csv", "data.csv",
            "chexmultimodal.csv", "labels.csv", "index.csv",
        ]

        df = None
        for csv_name in csv_candidates:
            csv_path = os.path.join(self.root_dir, csv_name)
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                print(f"  Loaded: {csv_path} ({len(df)} rows)")
                break

        if df is None:
            # Also check for any CSV in the root
            csv_files = [f for f in os.listdir(self.root_dir) if f.endswith('.csv')]
            if csv_files:
                csv_path = os.path.join(self.root_dir, csv_files[0])
                df = pd.read_csv(csv_path)
                print(f"  Loaded: {csv_path} ({len(df)} rows)")
            else:
                print(f"  Warning: No CSV found in {self.root_dir}. Using dummy data.")
                return self._create_dummy_metadata()

        # Normalize column names (lowercase, strip whitespace)
        df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]

        # Auto-detect image path and text columns
        self._image_col = self._find_column(df, ["patient_id", "image_path", "filename", "file", "image", "path", "dicom_id"])
        self._text_col = self._find_column(df, ["indication", "clinical_history", "text", "report", "history", "findings"])

        if self._image_col is None:
            print(f"  Warning: Could not find image column. Available: {list(df.columns)}")
            self._image_col = df.columns[0]

        if self._text_col is None:
            print(f"  Warning: Could not find text column. Available: {list(df.columns)}")
            self._text_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]

        # Drop rows with missing indication text
        df = df.dropna(subset=[self._text_col]).reset_index(drop=True)

        print(f"  Using image column: '{self._image_col}', text column: '{self._text_col}'")
        print(f"  Valid samples: {len(df)}")

        # Split into train/val/test
        test_ratio = 1.0 - train_ratio - val_ratio
        train_df, temp_df = train_test_split(df, test_size=(val_ratio + test_ratio), random_state=seed)
        val_df, test_df = train_test_split(temp_df, test_size=test_ratio / (val_ratio + test_ratio), random_state=seed)

        splits = {"train": train_df, "val": val_df, "test": test_df}
        return splits[self.split].reset_index(drop=True)

    @staticmethod
    def _find_column(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
        """Find the first matching column name from candidates."""
        for col in candidates:
            if col in df.columns:
                return col
            # Partial match
            for df_col in df.columns:
                if col in df_col:
                    return df_col
        return None

    def _create_dummy_metadata(self, n_samples: int = 50) -> pd.DataFrame:
        """Create dummy metadata for testing without real data."""
        self._image_col = "image_path"
        self._text_col = "indication"
        data = {
            "image_path": [f"dummy_{i}.jpg" for i in range(n_samples)],
            "indication": [
                "68 years of age, Female, chest pain",
                "45 years of age, Male, shortness of breath",
                "72 years of age, Female, cough and fever",
                "55 years of age, Male, post-surgical follow-up",
                "30 years of age, Male, trauma after fall",
            ] * (n_samples // 5),
        }
        return pd.DataFrame(data)

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> dict:
        """
        Returns:
            Dictionary with:
                'image': (1, H, W) tensor — chest X-ray
                'input_ids': (L,) tensor — tokenized indication text
                'attention_mask': (L,) tensor — attention mask
                'labels': (C,) tensor — pseudo-labels from indication
                'text_raw': str — raw indication text
        """
        row = self.metadata.iloc[idx]

        # --- Load Image ---
        img_id = str(row[self._image_col]).strip()
        # Build filename: patient_id → {patient_id}.jpg
        img_filename = f"{img_id}.jpg" if not img_id.lower().endswith(('.jpg', '.jpeg', '.png')) else img_id
        # Try multiple possible image locations
        img_paths = [
            os.path.join(self.root_dir, "new_patient_images_324", img_filename),
            os.path.join(self.root_dir, "images", img_filename),
            os.path.join(self.root_dir, img_filename),
        ]

        image = None
        for path in img_paths:
            if os.path.exists(path):
                image = Image.open(path).convert("L")  # Grayscale
                break

        if image is None:
            # Dummy image for testing
            image = Image.fromarray(
                np.random.randint(0, 256, (256, 256), dtype=np.uint8), mode="L"
            )

        if self.transform:
            image = self.transform(image)

        # --- Process Text ---
        raw_text = str(row.get(self._text_col, "No indication provided"))
        processed_text = self.text_preprocessor(raw_text)

        # Tokenize
        encoded = self.tokenizer(
            processed_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt",
        )

        # --- Labels (derived from indication text) ---
        labels = _extract_labels_from_indication(raw_text)
        labels = torch.tensor(labels, dtype=torch.float32)

        return {
            "image": image,
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": labels,
            "text_raw": raw_text,
        }


def collate_chex_multimodal(batch: list[dict]) -> dict:
    """
    Collate function for CheXmultimodal.
    Wraps single images into multi-modality format for OmniModalFramework.
    """
    images = torch.stack([item["image"] for item in batch])       # (B, 1, H, W)
    images = images.unsqueeze(1)                                   # (B, 1, 1, H, W) — 1 modality
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])

    return {
        "images": images,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def get_chex_multimodal_dataloaders(config: dict) -> dict:
    """
    Factory function to create CheXmultimodal DataLoaders.

    Args:
        config: Configuration dict with keys:
            - root_dir: Path to CheXmultimodal data
            - batch_size: Batch size (default: 16)
            - num_workers: DataLoader workers (default: 2)
            - max_text_length: Max token length (default: 128)
            - pin_memory: Pin GPU memory (default: True)

    Returns:
        Dictionary with 'train', 'val', 'test' DataLoaders
    """
    loaders = {}
    for split in ["train", "val", "test"]:
        dataset = CheXmultimodalDataset(
            root_dir=config.get("root_dir", "./data/chexmultimodal"),
            split=split,
            max_text_length=config.get("max_text_length", 128),
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size=config.get("batch_size", 16),
            shuffle=(split == "train"),
            num_workers=config.get("num_workers", 2),
            pin_memory=config.get("pin_memory", True),
            collate_fn=collate_chex_multimodal,
            drop_last=(split == "train"),
        )
    return loaders


# --- Quick test ---
if __name__ == "__main__":
    print("=== CheXmultimodal Dataset Test ===")
    print(f"Labels ({len(CHEX_MULTIMODAL_LABELS)}): {CHEX_MULTIMODAL_LABELS}")

    # Test label extraction
    test_indications = [
        "68 years of age, Female, chest pain",
        "45 years of age, Male, shortness of breath and cough",
        "72 years of age, Female, follow-up after pneumonia",
        "55 years of age, Male, post-surgical evaluation",
    ]
    for ind in test_indications:
        labels = _extract_labels_from_indication(ind)
        active = [CHEX_MULTIMODAL_LABELS[i] for i, v in enumerate(labels) if v > 0]
        print(f"  '{ind[:50]}...' → {active}")

    # Create dummy dataset
    dataset = CheXmultimodalDataset(
        root_dir="./data/chexmultimodal",
        split="train",
    )
    print(f"\nDataset size: {len(dataset)}")
    print("Dataset initialized successfully.")
