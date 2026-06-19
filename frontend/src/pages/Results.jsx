import React, { useState } from 'react';

const cases = [
  {
    id: 'PT-84729', name: 'John Doe', age: 58, sex: 'Male', scan: 'Oct 24, 2026', referral: 'Dr. A. Smith (Neurology)',
    modalities: ['MRI T1', 'MRI T2', 'FLAIR'],
    report: 'Multiple hyperintense white matter lesions seen on T2 and FLAIR sequences, primarily distributed in the periventricular and juxtacortical regions. The largest lesion measures approx 8mm in the left centrum semiovale. No abnormal contrast enhancement noted. Findings are highly suggestive of demyelinating disease.',
    highlights: ['hyperintense white matter lesions', 'periventricular', 'juxtacortical', '8mm', 'demyelinating disease'],
    diagnosis: 'Demyelinating Disease',
    confidence: 94.2,
    severity: 'Moderate',
    predictions: [
      { label: 'Multiple Sclerosis (MS)', score: 88.5 },
      { label: 'Ischemic Stroke', score: 12.1 },
      { label: 'Brain Tumor (Glioma)', score: 0.4 },
      { label: 'Normal', score: 0.2 },
    ],
    gates: { visual: 0.65, text: 0.82 },
    explainability: {
      visual: 'Periventricular Hyperintensities — FLAIR modality showed strongest activation in deep white matter regions adjacent to lateral ventricles.',
      semantic: '"white matter lesions", "demyelinating", "periventricular" — these tokens had highest attention weights grounding to spatial patches.',
    },
    attention: [
      { token: 'hyperintense', regions: 'Deep white matter bilateral', weight: 0.94 },
      { token: 'periventricular', regions: 'Lateral ventricle margins', weight: 0.91 },
      { token: 'lesions', regions: 'Multiple foci in WM', weight: 0.87 },
      { token: 'demyelinating', regions: 'Centrum semiovale', weight: 0.83 },
      { token: '8mm', regions: 'Left centrum semiovale', weight: 0.79 },
    ],
  },
  {
    id: 'PT-31056', name: 'Maria Lopez', age: 45, sex: 'Female', scan: 'Nov 12, 2026', referral: 'Dr. K. Patel (Radiology)',
    modalities: ['MRI T1', 'MRI T2'],
    report: 'A well-defined ring-enhancing lesion in the right frontal lobe measuring 2.3cm with surrounding vasogenic edema. Mass effect noted with mild midline shift of 3mm. Findings concerning for high-grade glioma vs. metastatic disease.',
    highlights: ['ring-enhancing lesion', 'right frontal lobe', '2.3cm', 'vasogenic edema', 'midline shift'],
    diagnosis: 'Brain Tumor (Glioma)',
    confidence: 91.7,
    severity: 'High',
    predictions: [
      { label: 'Brain Tumor (Glioma)', score: 91.7 },
      { label: 'Metastatic Disease', score: 34.2 },
      { label: 'Brain Abscess', score: 8.3 },
      { label: 'Multiple Sclerosis', score: 1.1 },
    ],
    gates: { visual: 0.88, text: 0.71 },
    explainability: {
      visual: 'Ring-enhancing pattern in right frontal lobe — T1 post-contrast showed strongest activation at lesion periphery.',
      semantic: '"ring-enhancing", "mass effect", "midline shift" — high attention to spatial abnormalities.',
    },
    attention: [
      { token: 'ring-enhancing', regions: 'Right frontal lobe periphery', weight: 0.96 },
      { token: 'mass effect', regions: 'Right hemisphere compression', weight: 0.89 },
      { token: '2.3cm', regions: 'Lesion core measurement', weight: 0.85 },
      { token: 'edema', regions: 'Peritumoral white matter', weight: 0.82 },
    ],
  },
];

const Results = () => {
  const [activeCase, setActiveCase] = useState(0);
  const c = cases[activeCase];
  const severityColor = c.severity === 'High' ? '#dc2626' : c.severity === 'Moderate' ? '#f59e0b' : '#22c55e';

  return (
    <div style={{ paddingTop: '5rem' }}>
      <section className="section" style={{ textAlign: 'center', paddingBottom: '1rem' }}>
        <p className="section-label">Diagnostic Output</p>
        <h1 className="section-title" style={{ fontSize: '3.5rem' }}>Diagnostic Results</h1>
        <p className="section-subtitle">Interactive demo showing how OmniDiag processes multi-modal inputs and produces explainable diagnostic output.</p>
      </section>

      {/* Case Selector */}
      <div style={{ maxWidth: 1200, margin: '0 auto', padding: '0 4rem', display: 'flex', gap: '1rem', marginBottom: '2rem' }}>
        {cases.map((cs, i) => (
          <button key={cs.id} onClick={() => setActiveCase(i)} style={{
            flex: 1, padding: '1.25rem', border: activeCase === i ? '2px solid #111' : '1px solid #e8e8e4',
            borderRadius: 14, background: activeCase === i ? '#111' : '#fff', color: activeCase === i ? '#fff' : '#111',
            cursor: 'pointer', textAlign: 'left', fontFamily: 'var(--font-sans)', transition: 'all 0.3s',
          }}>
            <div style={{ fontWeight: 600, fontSize: '0.95rem' }}>{cs.name}</div>
            <div style={{ fontSize: '0.8rem', opacity: 0.6, marginTop: 2 }}>{cs.id} · {cs.sex} · {cs.age}y · {cs.diagnosis}</div>
          </button>
        ))}
      </div>

      {/* Result Dashboard */}
      <section style={{ maxWidth: 1200, margin: '0 auto', padding: '0 4rem 4rem' }}>
        {/* Patient Header */}
        <div style={{ background: '#fff', border: '1px solid #e8e8e4', borderRadius: 16, padding: '2rem', marginBottom: '2rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <h2 style={{ fontFamily: 'var(--font-serif)', fontSize: '2rem', marginBottom: '0.25rem' }}>{c.name}</h2>
            <p style={{ color: '#888', fontSize: '0.9rem' }}>{c.id} · {c.sex} · {c.age} years · Scanned {c.scan}</p>
            <p style={{ color: '#888', fontSize: '0.85rem', marginTop: 4 }}>Referral: {c.referral}</p>
          </div>
          <div style={{ display: 'flex', gap: '0.75rem' }}>
            {c.modalities.map(m => (
              <span key={m} style={{ background: '#f4f4f0', border: '1px solid #e8e8e4', padding: '0.4rem 1rem', borderRadius: 50, fontSize: '0.8rem', fontWeight: 500 }}>{m}</span>
            ))}
          </div>
        </div>

        {/* Primary Diagnosis Banner */}
        <div style={{ background: '#111', borderRadius: 16, padding: '2.5rem', color: '#fff', marginBottom: '2rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <div style={{ fontSize: '0.7rem', textTransform: 'uppercase', letterSpacing: '0.15em', color: '#888', marginBottom: 8 }}>Primary Diagnosis</div>
            <h2 style={{ fontFamily: 'var(--font-serif)', fontSize: '2.5rem', color: '#fff', marginBottom: 4 }}>{c.diagnosis}</h2>
            <div style={{ display: 'flex', gap: '1.5rem', marginTop: '0.75rem' }}>
              <span style={{ fontSize: '0.85rem', color: '#aaa' }}>Severity: <span style={{ color: severityColor, fontWeight: 600 }}>{c.severity}</span></span>
            </div>
          </div>
          <div style={{ textAlign: 'center' }}>
            <div style={{ fontSize: '4rem', fontFamily: 'var(--font-serif)', fontWeight: 500 }}>{c.confidence}%</div>
            <div style={{ fontSize: '0.75rem', color: '#888', textTransform: 'uppercase', letterSpacing: '0.1em' }}>Confidence</div>
          </div>
        </div>

        {/* 3-column grid */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '2rem', marginBottom: '2rem' }}>
          {/* Predictions */}
          <div style={{ background: '#fff', border: '1px solid #e8e8e4', borderRadius: 16, padding: '2rem' }}>
            <h4 style={{ fontFamily: 'var(--font-serif)', fontSize: '1.1rem', marginBottom: '1.5rem' }}>Multi-Label Predictions</h4>
            {c.predictions.map((p, i) => (
              <div key={p.label} style={{ marginBottom: '1.25rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: 6, color: i === 0 ? '#111' : '#888', fontWeight: i === 0 ? 600 : 400 }}>
                  <span>{p.label}</span>
                  <span>{p.score}%</span>
                </div>
                <div style={{ width: '100%', height: 8, background: '#f4f4f0', borderRadius: 4, overflow: 'hidden' }}>
                  <div style={{ width: `${p.score}%`, height: '100%', background: i === 0 ? '#111' : '#ccc', borderRadius: 4, transition: 'width 1s ease' }}></div>
                </div>
              </div>
            ))}
          </div>

          {/* Gate Values */}
          <div style={{ background: '#fff', border: '1px solid #e8e8e4', borderRadius: 16, padding: '2rem' }}>
            <h4 style={{ fontFamily: 'var(--font-serif)', fontSize: '1.1rem', marginBottom: '1.5rem' }}>Modality Gating (Module 4)</h4>
            <p style={{ fontSize: '0.85rem', color: '#888', lineHeight: 1.6, marginBottom: '1.5rem' }}>Instance-specific dynamic weighting shows which modality the model relied on more for this diagnosis.</p>
            
            <div style={{ textAlign: 'center', marginBottom: '1.5rem' }}>
              <div style={{ display: 'flex', justifyContent: 'center', gap: '2rem', marginBottom: '1rem' }}>
                <div>
                  <div style={{ width: 80, height: 80, borderRadius: '50%', border: '4px solid #111', display: 'flex', alignItems: 'center', justifyContent: 'center', fontFamily: 'var(--font-serif)', fontSize: '1.5rem', margin: '0 auto 0.5rem' }}>{c.gates.visual}</div>
                  <div style={{ fontSize: '0.75rem', color: '#888', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Visual g_v</div>
                </div>
                <div>
                  <div style={{ width: 80, height: 80, borderRadius: '50%', border: '4px solid #111', display: 'flex', alignItems: 'center', justifyContent: 'center', fontFamily: 'var(--font-serif)', fontSize: '1.5rem', margin: '0 auto 0.5rem' }}>{c.gates.text}</div>
                  <div style={{ fontSize: '0.75rem', color: '#888', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Text g_t</div>
                </div>
              </div>
              <p style={{ fontSize: '0.8rem', color: '#666', fontStyle: 'italic' }}>
                {c.gates.text > c.gates.visual
                  ? '📝 Text modality dominated — clinical report was highly descriptive.'
                  : '🧠 Visual modality dominated — imaging features were clearer.'}
              </p>
            </div>
          </div>

          {/* Explainability */}
          <div style={{ background: '#fff', border: '1px solid #e8e8e4', borderRadius: 16, padding: '2rem' }}>
            <h4 style={{ fontFamily: 'var(--font-serif)', fontSize: '1.1rem', marginBottom: '1.5rem' }}>Explainability (Module 3)</h4>
            <div style={{ borderLeft: '3px solid #111', paddingLeft: '1rem', marginBottom: '1.5rem' }}>
              <div style={{ fontSize: '0.65rem', textTransform: 'uppercase', letterSpacing: '0.05em', color: '#999', marginBottom: 4, fontWeight: 600 }}>🧠 Visual Trigger</div>
              <p style={{ fontSize: '0.85rem', lineHeight: 1.6, color: '#444' }}>{c.explainability.visual}</p>
            </div>
            <div style={{ borderLeft: '3px solid #888', paddingLeft: '1rem' }}>
              <div style={{ fontSize: '0.65rem', textTransform: 'uppercase', letterSpacing: '0.05em', color: '#999', marginBottom: 4, fontWeight: 600 }}>📝 Semantic Trigger</div>
              <p style={{ fontSize: '0.85rem', lineHeight: 1.6, color: '#444' }}>{c.explainability.semantic}</p>
            </div>
          </div>
        </div>

        {/* Attention Map Table */}
        <div style={{ background: '#fff', border: '1px solid #e8e8e4', borderRadius: 16, padding: '2rem', marginBottom: '2rem' }}>
          <h4 style={{ fontFamily: 'var(--font-serif)', fontSize: '1.25rem', marginBottom: '1.5rem' }}>Cross-Attention Map (T→V) — Stage 2 Fine Fusion</h4>
          <p style={{ fontSize: '0.85rem', color: '#888', lineHeight: 1.6, marginBottom: '1.5rem' }}>Shows which image regions each clinical text token is grounding to, with attention weight intensity.</p>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.9rem' }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #111' }}>
                <th style={{ textAlign: 'left', padding: '0.75rem 0', fontWeight: 600 }}>Token</th>
                <th style={{ textAlign: 'left', padding: '0.75rem 0', fontWeight: 600 }}>Grounded Region</th>
                <th style={{ textAlign: 'left', padding: '0.75rem 0', fontWeight: 600, width: 200 }}>Attention Weight</th>
              </tr>
            </thead>
            <tbody>
              {c.attention.map((a, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #e8e8e4' }}>
                  <td style={{ padding: '0.75rem 0', fontFamily: 'monospace', fontWeight: 600 }}>"{a.token}"</td>
                  <td style={{ padding: '0.75rem 0', color: '#555' }}>{a.regions}</td>
                  <td style={{ padding: '0.75rem 0' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                      <div style={{ flex: 1, height: 10, background: '#f4f4f0', borderRadius: 5, overflow: 'hidden' }}>
                        <div style={{ width: `${a.weight * 100}%`, height: '100%', background: '#111', borderRadius: 5, transition: 'width 1s ease' }}></div>
                      </div>
                      <span style={{ fontFamily: 'monospace', fontSize: '0.8rem', minWidth: 35 }}>{a.weight}</span>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Clinical Report */}
        <div style={{ background: '#fff', border: '1px solid #e8e8e4', borderRadius: 16, padding: '2rem' }}>
          <h4 style={{ fontFamily: 'var(--font-serif)', fontSize: '1.25rem', marginBottom: '1rem' }}>Clinical Report (Module 2 Input)</h4>
          <p style={{ fontSize: '1rem', lineHeight: 2, color: '#444' }}>
            {c.report.split(' ').map((word, i) => {
              const clean = word.replace(/[.,]/g, '');
              const isHL = c.highlights.some(h => h.toLowerCase().includes(clean.toLowerCase()) && clean.length > 3);
              return <span key={i}>{isHL ? <mark style={{ background: 'rgba(0,0,0,0.08)', padding: '2px 4px', borderRadius: 3, borderBottom: '2px solid #111' }}>{word}</mark> : word} </span>;
            })}
          </p>
        </div>
      </section>
    </div>
  );
};

export default Results;
