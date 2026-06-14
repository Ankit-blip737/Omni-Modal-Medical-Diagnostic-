import React from 'react';
import { Link } from 'react-router-dom';

const Home = () => (
  <>
    {/* Hero */}
    <section className="hero">
      <div className="hero-bg"><img src="/hero.png" alt="" /></div>
      <div className="hero-content">
        <div className="hero-badge"><span className="hero-badge-dot"></span>Omni-Modal Framework</div>
        <h1>Turn medical scans into <em>precise diagnostics.</em></h1>
        <p className="hero-sub">A unified framework fusing multi-modal MRI images and clinical text through bidirectional progressive fusion with dynamic attention-based modality weighting.</p>
        <div className="hero-buttons">
          <Link to="/architecture" className="btn-dark">Explore Architecture</Link>
          <Link to="/results" className="btn-outline">View Results</Link>
        </div>
      </div>
    </section>

    {/* Showcase */}
    <section className="showcase">
      <div className="showcase-grid">
        <Link to="/architecture" className="showcase-card">
          <img src="/card-imaging.png" alt="Multi-Modal MRI" />
          <div className="showcase-card-overlay"><h3>Multi-Modal MRI</h3><p>T1 · T2 · FLAIR</p></div>
        </Link>
        <Link to="/architecture" className="showcase-card tall">
          <img src="/card-fusion.png" alt="Bidirectional Fusion" />
          <div className="showcase-card-overlay"><h3>Step into the fusion of vision &amp; language.</h3><p>Bidirectional Progressive Cross-Attention</p></div>
        </Link>
        <Link to="/architecture" className="showcase-card">
          <img src="/card-text.png" alt="Clinical NLP" />
          <div className="showcase-card-overlay"><h3>Clinical NLP</h3><p>BioBERT · LoRA</p></div>
        </Link>
      </div>
    </section>

    {/* Quick overview */}
    <section className="section">
      <p className="section-label">The Pipeline</p>
      <h2 className="section-title">Five modules. One unified pipeline.</h2>
      <p className="section-subtitle">From raw MRI scans and clinical reports to explainable diagnostic output.</p>
      <div className="modules-grid">
        {[
          { n:'01', t:'Joint Image Encoder', d:'Shared-then-Split ConvNeXt encodes T1, T2, FLAIR with cross-attention fusion.', tags:['ConvNeXt','Cross-Attention','768-dim'] },
          { n:'02', t:'Clinical Text Encoder', d:'PubMedBERT + LoRA extracts global [CLS] and token-level features from reports.', tags:['PubMedBERT','LoRA','Dual Output'] },
          { n:'03', t:'Visual-Semantic Alignment', d:'CLIP-style contrastive learning maps images & text into a shared 256-dim metric space.', tags:['InfoNCE','256-dim','Zero-Shot'] },
          { n:'04', t:'Progressive Fusion', d:'3-stage bidirectional cross-attention with gated modality aggregation.', tags:['3-Stage','Gated','Explainable'] },
          { n:'05', t:'Diagnostic Head', d:'MLP classifier with Focal Loss for multi-label classification.', tags:['Focal Loss','Multi-Label','Dropout'] },
        ].map(m => (
          <div className="module-card" key={m.n}>
            <div className="module-number">{m.n}</div>
            <h3>{m.t}</h3><p>{m.d}</p>
            <div className="module-tags">{m.tags.map(t=><span className="module-tag" key={t}>{t}</span>)}</div>
          </div>
        ))}
      </div>
    </section>

    {/* Stats */}
    <div className="stats-band">
      <div className="stats-inner">
        <div className="stat-item"><h2>94.2%</h2><p>Diagnostic Confidence</p></div>
        <div className="stat-item"><h2>768</h2><p>Feature Dimensions</p></div>
        <div className="stat-item"><h2>377K</h2><p>Training Images</p></div>
        <div className="stat-item"><h2>5</h2><p>Core Modules</p></div>
      </div>
    </div>

    {/* CTA */}
    <section className="cta-section">
      <h2>Ready to explore?</h2>
      <p>Dive into the full architecture, training strategy, or see diagnostic results.</p>
      <div className="hero-buttons">
        <Link to="/architecture" className="btn-dark">Full Architecture</Link>
        <Link to="/results" className="btn-outline">Diagnostic Demo</Link>
      </div>
    </section>
  </>
);

export default Home;
