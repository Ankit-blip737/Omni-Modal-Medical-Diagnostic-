# 🧠 Omni-Modal Medical Diagnostic Framework

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.1+](https://img.shields.io/badge/pytorch-2.1%2B-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A unified deep learning framework that fuses **multi-modal medical images** (MRI T1/T2/FLAIR, Chest X-rays) with **clinical text** (radiology reports, EHR notes) for diagnostic classification using **bidirectional progressive fusion** and **visual-semantic alignment**.

## ✨ Key Features

| Module | Description |
|--------|-------------|
| 🖼️ **Joint Image Encoder** | ConvNeXt backbone with shared early layers and modality-specific branches for multi-modal MRI |
| 📝 **Clinical Text Encoder** | PubMedBERT/BioBERT with LoRA fine-tuning for parameter-efficient adaptation |
| 🔗 **Visual-Semantic Alignment** | CLIP-style contrastive learning to map images and text into a shared 256-dim metric space |
| 🔄 **Bidirectional Progressive Fusion** | Multi-stage reciprocal cross-attention with dynamic gated modality aggregation |
| 🩺 **Diagnostic Head** | Multi-label classifier with Focal Loss for handling class imbalance |

## 🏗️ Architecture

```
Multi-Modal Images ──→ Joint Image Encoder ──→ Visual Features (B,49,768)
                                                      │
                                                      ├──→ Visual-Semantic Alignment ──→ Contrastive Loss
                                                      │              ↑
Clinical Text ────→ Text Encoder ──→ [CLS] + Tokens   │    Text Projection
                                          │            │
                                          └────────────┤
                                                       ↓
                                              Bidirectional Progressive
                                              Fusion Engine (2 stages)
                                                       │
                                                       ↓
                                              Gated Aggregation
                                              (g_v ⊙ V + g_t ⊙ T)
                                                       │
                                                       ↓
                                              Diagnostic Classification
                                                       │
                                                       ↓
                                                  🩺 Diagnosis
```

## 📁 Project Structure

```
├── configs/
│   ├── default.yaml          # Training hyperparameters
│   ├── model.yaml            # Architecture variants (base/small/single)
│   └── data.yaml             # Dataset configurations
├── src/
│   ├── models/
│   │   ├── image_encoder.py  # Module 1: Joint Image Encoder (ConvNeXt)
│   │   ├── text_encoder.py   # Module 2: Clinical Text Encoder (BioBERT/PubMedBERT)
│   │   ├── alignment.py      # Module 3: Visual-Semantic Alignment (InfoNCE)
│   │   ├── fusion.py         # Module 4: Bidirectional Progressive Fusion
│   │   ├── classifier.py     # Module 5: Diagnostic Classification Head
│   │   └── omni_modal.py     # Full Framework (ties all modules)
│   ├── data/
│   │   ├── mimic_cxr.py      # MIMIC-CXR dataset loader
│   │   ├── brats.py          # BraTS multi-modal MRI loader
│   │   ├── transforms.py     # Medical image augmentations (CLAHE, etc.)
│   │   └── text_utils.py     # Clinical text preprocessing
│   ├── training/
│   │   ├── trainer.py        # 3-phase curriculum trainer
│   │   ├── losses.py         # Focal Loss + Combined multi-task loss
│   │   └── schedulers.py     # Warmup cosine/linear LR schedulers
│   ├── evaluation/
│   │   ├── metrics.py        # AUC, F1, precision, recall, specificity
│   │   └── visualization.py  # Attention heatmaps, ROC curves, gate plots
│   └── utils/
│       └── logging.py        # Logger + TensorBoard utilities
├── scripts/
│   ├── train.py              # Training entry point
│   ├── evaluate.py           # Evaluation script
│   └── visualize_attention.py # Attention visualization
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/Ankit-blip737/Omni-Modal-Medical-Diagnostic-.git
cd Omni-Modal-Medical-Diagnostic-
pip install -r requirements.txt
```

### 2. Training

```bash
# Run all 3 training phases
python scripts/train.py --config configs/default.yaml

# Run specific phase
python scripts/train.py --config configs/default.yaml --phase 1  # Contrastive pre-training
python scripts/train.py --config configs/default.yaml --phase 2  # Fusion training
python scripts/train.py --config configs/default.yaml --phase 3  # End-to-end fine-tuning

# Resume from checkpoint
python scripts/train.py --config configs/default.yaml --resume checkpoints/best_phase1.pth --phase 2

# Smoke test (validate setup without data)
python scripts/train.py --config configs/default.yaml --smoke-test
```

### 3. Evaluation

```bash
python scripts/evaluate.py --checkpoint checkpoints/best_phase2.pth --split test --output-dir results/
```

### 4. Visualization

```bash
python scripts/visualize_attention.py \
    --checkpoint checkpoints/best_phase2.pth \
    --image-path data/sample_xray.jpg \
    --text "Large right-sided pleural effusion with associated atelectasis" \
    --output-dir visualizations/
```

## 📊 Training Strategy (3-Phase Curriculum)

| Phase | Objective | Trainable Components | Epochs | LR |
|-------|-----------|---------------------|--------|-----|
| **Phase 1** | Contrastive Alignment | Projection heads only | 50 | 1e-4 |
| **Phase 2** | Fusion + Classification | LoRA + Branches + Fusion + Classifier | 100 | 5e-5 |
| **Phase 3** | End-to-End Fine-tuning | All parameters | 20 | 1e-6 |

## 📦 Datasets

| Dataset | Type | Size | Access |
|---------|------|------|--------|
| **MIMIC-CXR v2** | Chest X-ray + Reports | 377K images | [PhysioNet](https://physionet.org/content/mimic-cxr/2.0.0/) (credentialed) |
| **BraTS 2024** | Multi-modal Brain MRI | ~2K subjects | [Synapse](https://www.synapse.org/#!Synapse:syn51156910) |
| **CheXpert** | Chest X-ray + Labels | 224K images | [Stanford](https://stanfordmlgroup.github.io/competitions/chexpert/) |

## 🔬 Technical Details

- **Vision Backbone**: ConvNeXt-Base (ImageNet pretrained)
- **Language Model**: PubMedBERT / BioBERT v1.1 with LoRA (rank=8)
- **Alignment**: 256-dim shared metric space with symmetric InfoNCE loss
- **Fusion**: 2-stage bidirectional cross-attention (coarse → fine) with gated aggregation
- **Robustness**: Modality dropout (p=0.15) during training for missing-modality resilience
- **Class Imbalance**: Focal Loss (γ=2) with per-class alpha weighting

## 📄 Citation

```bibtex
@misc{omni-modal-diagnostic-2026,
  title={Omni-Modal Medical Diagnostic Framework: Bidirectional Progressive Fusion
         of Multi-Modal Medical Images and Clinical Text},
  author={Ankit},
  year={2026},
  url={https://github.com/Ankit-blip737/Omni-Modal-Medical-Diagnostic-}
}
```

## 📜 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
