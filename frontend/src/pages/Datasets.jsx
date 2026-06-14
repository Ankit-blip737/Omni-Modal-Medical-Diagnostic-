import React from 'react';

const datasets = [
  { name: 'MIMIC-CXR v2', modalities: 'Chest X-ray + Radiology Reports', size: '377K images, 228K reports', access: 'PhysioNet (credentialed)', recommended: true, note: 'Gold standard for image-text pairs. Requires CITI training + DUA. Plan 2-4 weeks for access.' },
  { name: 'BraTS 2024', modalities: 'MRI T1, T1ce, T2, FLAIR', size: '~2,000 subjects', access: 'Synapse / Kaggle', recommended: true, note: 'Best for multi-modal MRI brain tumor segmentation and diagnosis.' },
  { name: 'TCGA (GDC)', modalities: 'Histopathology + Genomics + Clinical', size: '~11,000 cases', access: 'NIH GDC Portal', recommended: false, note: 'Massive multi-modal clinical dataset from The Cancer Genome Atlas.' },
  { name: 'CheXpert', modalities: 'Chest X-ray + Labels', size: '224K images', access: 'Stanford ML Group', recommended: false, note: 'Good starting point — freely available. Can be used with synthetic text labels for prototyping.' },
  { name: 'PadChest', modalities: 'Chest X-ray + Reports (Spanish)', size: '160K images', access: 'BIMCV', recommended: false, note: 'Large dataset with Spanish reports — useful for multilingual experiments.' },
  { name: 'RadNLI', modalities: 'NLI pairs from radiology', size: '1K pairs', access: 'Public', recommended: false, note: 'Small but useful for text encoder evaluation and NLI benchmarking.' },
];

const hardware = [
  { component: 'GPU', minimum: '1× RTX 3090 (24GB)', recommended: '2× A100 (80GB)' },
  { component: 'RAM', minimum: '32 GB', recommended: '64 GB' },
  { component: 'Storage', minimum: '100 GB (MIMIC-CXR)', recommended: '500 GB (all datasets)' },
  { component: 'Training Time', minimum: '~48h (Phase 1+2)', recommended: '~24h with 2× A100' },
];

const Datasets = () => (
  <div style={{ paddingTop: '5rem' }}>
    <section className="section" style={{ textAlign: 'center', paddingBottom: '2rem' }}>
      <p className="section-label">Data & Infrastructure</p>
      <h1 className="section-title" style={{ fontSize: '3.5rem' }}>Datasets & Hardware</h1>
      <p className="section-subtitle">Recommended datasets for training and the hardware requirements for each training phase.</p>
    </section>

    {/* Datasets */}
    <section style={{ maxWidth: 1200, margin: '0 auto', padding: '0 4rem 4rem' }}>
      <h2 style={{ fontFamily: 'var(--font-serif)', fontSize: '2rem', marginBottom: '1.5rem' }}>Recommended Datasets</h2>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '1.5rem' }}>
        {datasets.map(d => (
          <div key={d.name} style={{ background: '#fff', border: d.recommended ? '2px solid #111' : '1px solid #e8e8e4', borderRadius: 16, padding: '2rem', position: 'relative', transition: 'all 0.3s' }}>
            {d.recommended && <span style={{ position: 'absolute', top: -10, right: 20, background: '#111', color: '#fff', padding: '2px 12px', borderRadius: 50, fontSize: '0.7rem', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em' }}>Recommended</span>}
            <h3 style={{ fontFamily: 'var(--font-serif)', fontSize: '1.3rem', marginBottom: '0.75rem' }}>{d.name}</h3>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '1rem' }}>
              <div><div style={{ fontSize: '0.65rem', textTransform: 'uppercase', letterSpacing: '0.05em', color: '#999', marginBottom: 4, fontWeight: 600 }}>Modalities</div><div style={{ fontSize: '0.9rem' }}>{d.modalities}</div></div>
              <div><div style={{ fontSize: '0.65rem', textTransform: 'uppercase', letterSpacing: '0.05em', color: '#999', marginBottom: 4, fontWeight: 600 }}>Size</div><div style={{ fontSize: '0.9rem' }}>{d.size}</div></div>
            </div>
            <div style={{ fontSize: '0.65rem', textTransform: 'uppercase', letterSpacing: '0.05em', color: '#999', marginBottom: 4, fontWeight: 600 }}>Access</div>
            <div style={{ fontSize: '0.9rem', marginBottom: '0.75rem' }}>{d.access}</div>
            <p style={{ fontSize: '0.8rem', color: '#888', lineHeight: 1.5, fontStyle: 'italic', borderTop: '1px solid #e8e8e4', paddingTop: '0.75rem' }}>{d.note}</p>
          </div>
        ))}
      </div>
    </section>

    {/* Hardware */}
    <section style={{ maxWidth: 1200, margin: '0 auto', padding: '0 4rem 4rem' }}>
      <h2 style={{ fontFamily: 'var(--font-serif)', fontSize: '2rem', marginBottom: '1.5rem' }}>Hardware Requirements</h2>
      <div style={{ background: '#fff', border: '1px solid #e8e8e4', borderRadius: 16, overflow: 'hidden' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.95rem' }}>
          <thead>
            <tr style={{ background: '#111', color: '#fff' }}>
              <th style={{ textAlign: 'left', padding: '1rem 1.5rem', fontWeight: 600 }}>Component</th>
              <th style={{ textAlign: 'left', padding: '1rem 1.5rem', fontWeight: 600 }}>Minimum</th>
              <th style={{ textAlign: 'left', padding: '1rem 1.5rem', fontWeight: 600 }}>Recommended</th>
            </tr>
          </thead>
          <tbody>
            {hardware.map((h, i) => (
              <tr key={h.component} style={{ borderBottom: i < hardware.length - 1 ? '1px solid #e8e8e4' : 'none' }}>
                <td style={{ padding: '1rem 1.5rem', fontWeight: 600 }}>{h.component}</td>
                <td style={{ padding: '1rem 1.5rem', color: '#555' }}>{h.minimum}</td>
                <td style={{ padding: '1rem 1.5rem', color: '#555' }}>{h.recommended}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>

    {/* Project Structure */}
    <section style={{ maxWidth: 1200, margin: '0 auto', padding: '0 4rem 4rem' }}>
      <h2 style={{ fontFamily: 'var(--font-serif)', fontSize: '2rem', marginBottom: '1.5rem' }}>Project Structure</h2>
      <div style={{ background: '#111', borderRadius: 16, padding: '2.5rem', color: '#e8e8e4', overflow: 'auto' }}>
        <pre style={{ fontFamily: 'monospace', fontSize: '0.85rem', lineHeight: 1.8, whiteSpace: 'pre' }}>
{`btp/
├── configs/
│   ├── default.yaml          # Training hyperparameters
│   ├── model.yaml            # Architecture configuration
│   └── data.yaml             # Dataset paths
├── src/
│   ├── models/
│   │   ├── image_encoder.py  # Module 1: Joint Image Encoder
│   │   ├── text_encoder.py   # Module 2: BioBERT Text Encoder
│   │   ├── alignment.py      # Module 3: Visual-Semantic Alignment
│   │   ├── fusion.py         # Module 4: Bidirectional Progressive Fusion
│   │   ├── classifier.py     # Module 5: Diagnostic Head
│   │   └── omni_modal.py     # Full Framework
│   ├── data/
│   │   ├── mimic_cxr.py      # MIMIC-CXR dataloader
│   │   ├── brats.py          # BraTS MRI dataloader
│   │   ├── transforms.py     # Medical image augmentations
│   │   └── text_utils.py     # Tokenization & preprocessing
│   ├── training/
│   │   ├── trainer.py        # 3-phase curriculum training loop
│   │   ├── losses.py         # InfoNCE, Focal Loss, combined
│   │   └── schedulers.py     # LR schedulers with warmup
│   ├── evaluation/
│   │   ├── metrics.py        # AUC, F1, sensitivity, specificity
│   │   └── visualization.py  # Attention map visualization
│   └── utils/
│       └── logging.py        # W&B / TensorBoard integration
├── scripts/
│   ├── train.py              # Main training entry point
│   ├── evaluate.py           # Evaluation script
│   └── visualize_attention.py
├── requirements.txt
└── README.md`}
        </pre>
      </div>
    </section>
  </div>
);

export default Datasets;
