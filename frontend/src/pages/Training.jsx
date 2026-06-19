import React from 'react';

const Training = () => (
  <div style={{ paddingTop: '5rem' }}>
    <section className="section" style={{ textAlign: 'center', paddingBottom: '2rem' }}>
      <p className="section-label">Training Strategy</p>
      <h1 className="section-title" style={{ fontSize: '3.5rem' }}>Three-phase curriculum learning</h1>
      <p className="section-subtitle">A carefully designed training curriculum that progressively unfreezes model components for stable convergence.</p>
    </section>

    {/* Timeline */}
    <section style={{ maxWidth: 900, margin: '0 auto', padding: '0 4rem 4rem' }}>
      {[
        {
          phase: 1, title: 'Contrastive Pre-training', subtitle: 'Alignment',
          objective: 'Train only Module 3 (Visual-Semantic Alignment) + projection heads',
          loss: 'ℒ_align (Symmetric InfoNCE)',
          freeze: 'Image encoder (ConvNeXt) + Text encoder (BioBERT) fully frozen',
          epochs: 50, lr: '1e-4', batch: 256,
          purpose: 'Establish the shared metric space before fusion',
          color: '#e8f4f8',
        },
        {
          phase: 2, title: 'Fusion Training', subtitle: 'Core Training',
          objective: 'Train Module 4 (Fusion) + Module 5 (Classifier)',
          loss: 'ℒ_cls (Focal Loss) + 0.1 × ℒ_align',
          freeze: 'Unfreeze: LoRA on BioBERT 9-12, ConvNeXt Stages 3-4',
          epochs: 100, lr: '5e-5', batch: null,
          purpose: 'Cosine Annealing with Warmup (10 epochs)',
          color: '#f0f0e8',
        },
        {
          phase: 3, title: 'End-to-End Fine-tuning', subtitle: 'Refinement',
          objective: 'Fine-tune entire pipeline end-to-end',
          loss: 'ℒ_total = ℒ_cls + λ₁ℒ_align + λ₂ℒ_reg',
          freeze: 'Everything unfrozen',
          epochs: 20, lr: '1e-6', batch: null,
          purpose: 'Very low LR to preserve learned representations',
          color: '#f4eef8',
        },
      ].map((p, i) => (
        <div key={p.phase} style={{ display: 'flex', gap: '2rem', marginBottom: i < 2 ? '0' : '0', position: 'relative' }}>
          {/* Timeline line */}
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', width: 60, flexShrink: 0 }}>
            <div style={{
              width: 48, height: 48, borderRadius: '50%', background: '#111', color: '#fff',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              fontFamily: 'var(--font-serif)', fontSize: '1.2rem', zIndex: 1
            }}>{p.phase}</div>
            {i < 2 && <div style={{ width: 2, flex: 1, background: '#e8e8e4' }}></div>}
          </div>

          {/* Card */}
          <div style={{
            flex: 1, background: p.color, border: '1px solid #e8e8e4', borderRadius: 16, padding: '2rem',
            marginBottom: '2rem', transition: 'all 0.3s'
          }}>
            <div style={{ fontSize: '0.7rem', textTransform: 'uppercase', letterSpacing: '0.1em', color: '#999', marginBottom: 4 }}>{p.subtitle}</div>
            <h3 style={{ fontFamily: 'var(--font-serif)', fontSize: '1.5rem', marginBottom: '1rem' }}>{p.title}</h3>
            
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '1rem' }}>
              <div style={{ background: 'rgba(255,255,255,0.7)', borderRadius: 10, padding: '1rem' }}>
                <div style={{ fontSize: '0.65rem', textTransform: 'uppercase', letterSpacing: '0.05em', color: '#999', marginBottom: 4, fontWeight: 600 }}>Objective</div>
                <div style={{ fontSize: '0.9rem' }}>{p.objective}</div>
              </div>
              <div style={{ background: 'rgba(255,255,255,0.7)', borderRadius: 10, padding: '1rem' }}>
                <div style={{ fontSize: '0.65rem', textTransform: 'uppercase', letterSpacing: '0.05em', color: '#999', marginBottom: 4, fontWeight: 600 }}>Loss Function</div>
                <div style={{ fontSize: '0.9rem', fontFamily: 'monospace' }}>{p.loss}</div>
              </div>
              <div style={{ background: 'rgba(255,255,255,0.7)', borderRadius: 10, padding: '1rem' }}>
                <div style={{ fontSize: '0.65rem', textTransform: 'uppercase', letterSpacing: '0.05em', color: '#999', marginBottom: 4, fontWeight: 600 }}>Frozen / Unfrozen</div>
                <div style={{ fontSize: '0.9rem' }}>{p.freeze}</div>
              </div>
              <div style={{ background: 'rgba(255,255,255,0.7)', borderRadius: 10, padding: '1rem', display: 'flex', gap: '1.5rem' }}>
                <div><div style={{ fontSize: '0.65rem', textTransform: 'uppercase', letterSpacing: '0.05em', color: '#999', marginBottom: 4, fontWeight: 600 }}>Epochs</div><div style={{ fontSize: '1.5rem', fontFamily: 'var(--font-serif)' }}>{p.epochs}</div></div>
                <div><div style={{ fontSize: '0.65rem', textTransform: 'uppercase', letterSpacing: '0.05em', color: '#999', marginBottom: 4, fontWeight: 600 }}>LR</div><div style={{ fontSize: '1rem', fontFamily: 'monospace' }}>{p.lr}</div></div>
                {p.batch && <div><div style={{ fontSize: '0.65rem', textTransform: 'uppercase', letterSpacing: '0.05em', color: '#999', marginBottom: 4, fontWeight: 600 }}>Batch</div><div style={{ fontSize: '1.5rem', fontFamily: 'var(--font-serif)' }}>{p.batch}</div></div>}
              </div>
            </div>
            <p style={{ fontSize: '0.85rem', color: '#666', fontStyle: 'italic' }}>💡 {p.purpose}</p>
          </div>
        </div>
      ))}
    </section>

    {/* Loss Function Summary */}
    <section style={{ maxWidth: 900, margin: '0 auto', padding: '0 4rem 4rem' }}>
      <h2 style={{ fontFamily: 'var(--font-serif)', fontSize: '2rem', marginBottom: '1.5rem', textAlign: 'center' }}>Loss Function Summary</h2>
      <div style={{ background: '#111', borderRadius: 16, padding: '2.5rem', color: '#e8e8e4' }}>
        <pre style={{ fontFamily: 'monospace', fontSize: '0.95rem', lineHeight: 1.8, overflowX: 'auto', whiteSpace: 'pre-wrap' }}>
{`ℒ_total = ℒ_cls + λ₁ · ℒ_align + λ₂ · ℒ_modality_dropout

where:
  ℒ_cls     = Focal Loss (γ=2, α per class)    — handles class imbalance
  ℒ_align   = Symmetric InfoNCE               — visual-semantic alignment
  ℒ_reg     = L2 weight decay (1e-4)           — regularization
  λ₁ = 0.1, λ₂ = 0.01`}
        </pre>
      </div>
      <div style={{ background: '#f4f4f0', border: '1px solid #e8e8e4', borderRadius: 12, padding: '1.25rem', marginTop: '1.5rem' }}>
        <p style={{ fontSize: '0.9rem', lineHeight: 1.6, color: '#444' }}>
          <strong>Modality Dropout:</strong> During training, randomly drop entire modalities (set to zero) with p=0.15. This forces the model to be robust when a modality is missing at inference time — critical for real clinical deployment where a patient may not have all imaging types.
        </p>
      </div>
    </section>
  </div>
);

export default Training;
