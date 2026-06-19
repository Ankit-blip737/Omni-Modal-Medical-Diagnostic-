import React, { useState } from 'react';

const modules = [
  {
    id: 1, title: 'Joint Image Encoder', subtitle: 'Multi-Modal MRI',
    purpose: 'Encode multiple imaging modalities (T1, T2, FLAIR) into a unified visual representation while preserving modality-specific features.',
    arch: 'Shared-then-Split ConvNeXt',
    details: [
      { label: 'Shared Layers', value: 'ConvNeXt Stages 1-2 (frozen) — Low-level features (edges, textures) shared across modalities, reducing parameters by ~40%' },
      { label: 'Split Branches', value: 'Stages 3-4 (trainable) — T1 captures anatomy, T2 captures pathology, FLAIR captures edema' },
      { label: 'Fusion', value: 'Multi-Head Cross-Attention (8 heads, 768-dim) dynamically learns which modality patches are most relevant' },
    ],
    dims: [
      { stage: 'Input (per modality)', shape: '(B, 1, 224, 224)', note: 'Grayscale MRI slices' },
      { stage: 'After Stage 2 (shared)', shape: '(B, 192, 28, 28)', note: 'Shared low-level' },
      { stage: 'After Stage 4 (branch)', shape: '(B, 768, 7, 7)', note: 'Modality-specific' },
      { stage: 'After Cross-Attention', shape: '(B, 49, 768)', note: 'Fused visual features' },
    ],
    image: '/card-imaging.png',
    rationale: 'ConvNeXt outperforms ResNet50 by ~3% on medical imaging benchmarks while maintaining CNN inductive biases.',
  },
  {
    id: 2, title: 'Clinical Text Encoder', subtitle: 'NLP Pipeline',
    purpose: 'Encode clinical text (radiology reports, EHR notes) into both global (sentence-level) and local (token-level) semantic embeddings.',
    arch: 'PubMedBERT + LoRA',
    details: [
      { label: 'Base Model', value: 'PubMedBERT — trained from scratch on PubMed, no domain mismatch. Primary choice over BioBERT, ClinicalBERT, BioGPT.' },
      { label: 'Adaptation', value: 'LoRA (rank=8) on layers 9-12 — adds only ~0.3% trainable parameters while preserving pre-trained biomedical knowledge' },
      { label: 'Output', value: 'Dual: [CLS] token (d=768) for global semantic + All tokens (B, L, 768) for local semantic' },
    ],
    dims: [
      { stage: 'Input', shape: 'max_len=512', note: 'Tokenized clinical text' },
      { stage: 'Layers 1-8', shape: '(B, L, 768)', note: 'Frozen' },
      { stage: 'Layers 9-12', shape: '(B, L, 768)', note: 'LoRA rank=8' },
      { stage: '[CLS] Output', shape: '(B, 768)', note: 'Global embedding' },
    ],
    image: '/card-text.png',
    rationale: 'LoRA prevents catastrophic forgetting with limited medical data — full fine-tuning risks losing pre-trained biomedical knowledge.',
  },
  {
    id: 3, title: 'Visual-Semantic Alignment', subtitle: 'Core Innovation ⭐',
    purpose: 'Maps visual features and text embeddings into a shared 256-dim metric space using CLIP-style symmetric InfoNCE contrastive loss.',
    arch: 'Projection Heads + Contrastive Loss',
    details: [
      { label: 'Visual Projection', value: 'GAP → Linear(768,512) → BatchNorm + GELU → Linear(512,256) → L2 Normalize' },
      { label: 'Text Projection', value: '[CLS] → Linear(768,512) → BatchNorm + GELU → Linear(512,256) → L2 Normalize' },
      { label: 'Loss', value: 'Symmetric InfoNCE: ℒ_align = (ℒ_v→t + ℒ_t→v) / 2 with learnable temperature τ (init 0.07)' },
    ],
    dims: [
      { stage: 'Visual input', shape: '(B, 49, 768)', note: 'From Module 1' },
      { stage: 'After GAP', shape: '(B, 768)', note: 'Pooled visual' },
      { stage: 'Projected', shape: '(B, 256)', note: 'L2 normalized' },
      { stage: 'Similarity Matrix', shape: '(B, B)', note: 'Cosine sim / τ' },
    ],
    image: '/card-fusion.png',
    rationale: '256-dim acts as an information bottleneck, forcing the model to retain only diagnostically relevant cross-modal features.',
  },
  {
    id: 4, title: 'Progressive Fusion Engine', subtitle: 'Bidirectional ⭐',
    purpose: 'Performs reciprocal, multi-stage fusion of visual and textual features, allowing each modality to progressively refine the other.',
    arch: '3-Stage Bidirectional Cross-Attention',
    details: [
      { label: 'Stage 1: Coarse', value: 'V→T and T→V global cross-attention — visual patches "ask" text which regions matter' },
      { label: 'Stage 2: Fine', value: 'Repeated cross-attention on refined features + FFN — captures local alignments like "3mm nodule" ↔ specific patch' },
      { label: 'Stage 3: Gated', value: 'Sigmoid gates g_v, g_t dynamically weight each modality: F = g_v ⊙ V\' + g_t ⊙ T\'' },
    ],
    dims: [
      { stage: 'Visual in', shape: '(B, 49, 768)', note: 'Image features' },
      { stage: 'Text in', shape: '(B, L, 768)', note: 'Token features' },
      { stage: 'Gates', shape: 'g_v, g_t ∈ [0,1]', note: 'Instance-specific' },
      { stage: 'F_fused', shape: '(B, 768)', note: 'Attentive pooled' },
    ],
    image: '/card-fusion.png',
    rationale: 'Cross-attention weights from Stage 2 are directly interpretable — solving the black-box explainability problem.',
  },
  {
    id: 5, title: 'Diagnostic Head', subtitle: 'Classification',
    purpose: 'MLP classifier with dropout regularization supporting multi-label classification via Focal Loss.',
    arch: 'MLP with Focal Loss',
    details: [
      { label: 'Layer 1', value: 'Dropout(0.3) → Linear(768, 512) → GELU' },
      { label: 'Layer 2', value: 'Dropout(0.2) → Linear(512, num_classes)' },
      { label: 'Output', value: 'Softmax (single-label) or Sigmoid (multi-label) — Focal Loss (γ=2) handles class imbalance' },
    ],
    dims: [
      { stage: 'Input', shape: '(B, 768)', note: 'F_fused from Module 4' },
      { stage: 'Hidden', shape: '(B, 512)', note: 'After GELU' },
      { stage: 'Output', shape: '(B, C)', note: 'Per-class logits' },
    ],
    image: null,
    rationale: 'Focal Loss with per-class α weighting is essential for medical datasets with extreme class imbalance.',
  },
];

const Architecture = () => {
  const [active, setActive] = useState(0);
  const m = modules[active];

  return (
    <div style={{ paddingTop: '5rem' }}>
      {/* Page Hero */}
      <section className="section" style={{ textAlign: 'center', paddingBottom: '2rem' }}>
        <p className="section-label">System Architecture</p>
        <h1 className="section-title" style={{ fontSize: '3.5rem' }}>Five modules. One pipeline.</h1>
        <p className="section-subtitle">Click any module below to explore its architecture, dimensions, and design rationale in detail.</p>
      </section>

      {/* Pipeline Visual */}
      <div style={{ maxWidth: 1200, margin: '0 auto', padding: '0 4rem 3rem' }}>
        <div style={{ display: 'flex', gap: '0.75rem', justifyContent: 'center', flexWrap: 'wrap' }}>
          {modules.map((mod, i) => (
            <button key={mod.id} onClick={() => setActive(i)} style={{
              flex: '1 1 0', minWidth: 160, padding: '1.25rem 1rem', border: active === i ? '2px solid #111' : '1px solid #e8e8e4',
              borderRadius: 14, background: active === i ? '#111' : '#fff', color: active === i ? '#fff' : '#111',
              cursor: 'pointer', transition: 'all 0.3s', textAlign: 'center', fontFamily: 'var(--font-sans)',
            }}>
              <div style={{ fontFamily: 'var(--font-serif)', fontSize: '1.8rem', fontWeight: 300, opacity: 0.4, marginBottom: 4 }}>0{mod.id}</div>
              <div style={{ fontWeight: 600, fontSize: '0.85rem' }}>{mod.title}</div>
              <div style={{ fontSize: '0.7rem', opacity: 0.6, marginTop: 2 }}>{mod.subtitle}</div>
            </button>
          ))}
        </div>
      </div>

      {/* Module Detail */}
      <section style={{ maxWidth: 1200, margin: '0 auto', padding: '0 4rem 4rem' }}>
        <div style={{ display: 'grid', gridTemplateColumns: m.image ? '1fr 1fr' : '1fr', gap: '3rem', alignItems: 'start' }}>
          <div>
            <p style={{ fontSize: '0.7rem', textTransform: 'uppercase', letterSpacing: '0.15em', color: '#999', marginBottom: 8 }}>Module {m.id}</p>
            <h2 style={{ fontFamily: 'var(--font-serif)', fontSize: '2.5rem', marginBottom: '0.5rem' }}>{m.title}</h2>
            <p style={{ color: '#555', lineHeight: 1.7, marginBottom: '2rem', fontSize: '1.05rem' }}>{m.purpose}</p>

            <h4 style={{ fontFamily: 'var(--font-serif)', fontSize: '1.1rem', marginBottom: '1rem' }}>Architecture: {m.arch}</h4>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem', marginBottom: '2rem' }}>
              {m.details.map((d, i) => (
                <div key={i} style={{ background: '#fff', border: '1px solid #e8e8e4', borderRadius: 12, padding: '1.25rem' }}>
                  <div style={{ fontWeight: 600, fontSize: '0.8rem', textTransform: 'uppercase', letterSpacing: '0.05em', color: '#999', marginBottom: 6 }}>{d.label}</div>
                  <div style={{ fontSize: '0.95rem', lineHeight: 1.6 }}>{d.value}</div>
                </div>
              ))}
            </div>

            <div style={{ background: '#f4f4f0', border: '1px solid #e8e8e4', borderRadius: 12, padding: '1.25rem', marginBottom: '1.5rem' }}>
              <div style={{ fontWeight: 600, fontSize: '0.75rem', textTransform: 'uppercase', letterSpacing: '0.05em', color: '#999', marginBottom: 8 }}>💡 Design Rationale</div>
              <p style={{ fontSize: '0.9rem', lineHeight: 1.6, color: '#444' }}>{m.rationale}</p>
            </div>
          </div>

          <div>
            {m.image && <img src={m.image} alt={m.title} style={{ width: '100%', borderRadius: 16, marginBottom: '2rem', border: '1px solid #e8e8e4' }} />}
            <h4 style={{ fontFamily: 'var(--font-serif)', fontSize: '1.1rem', marginBottom: '1rem' }}>Key Dimensions</h4>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem' }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #111' }}>
                  <th style={{ textAlign: 'left', padding: '0.75rem 0', fontWeight: 600 }}>Stage</th>
                  <th style={{ textAlign: 'left', padding: '0.75rem 0', fontWeight: 600 }}>Shape</th>
                  <th style={{ textAlign: 'left', padding: '0.75rem 0', fontWeight: 600 }}>Notes</th>
                </tr>
              </thead>
              <tbody>
                {m.dims.map((d, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #e8e8e4' }}>
                    <td style={{ padding: '0.75rem 0' }}>{d.stage}</td>
                    <td style={{ padding: '0.75rem 0', fontFamily: 'monospace', fontSize: '0.8rem' }}>{d.shape}</td>
                    <td style={{ padding: '0.75rem 0', color: '#888' }}>{d.note}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </section>
    </div>
  );
};

export default Architecture;
