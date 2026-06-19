import React from 'react';
import { GitMerge, Network } from 'lucide-react';
import './FusionVisualizer.css';

const FusionVisualizer = () => {
  return (
    <div className="fusion-panel minimal-panel">
      <div className="panel-header">
        <div className="flex-center" style={{ gap: '12px' }}>
          <GitMerge size={18} strokeWidth={1.5} color="var(--accent-primary)" />
          <h3>Module 4: Progressive Fusion</h3>
        </div>
      </div>

      <div className="fusion-architecture">
        
        <div className="fusion-stage">
          <div className="stage-title">Stage 1: Coarse Global</div>
          <div className="cross-attention-box">
            <div className="stream visual-stream">V (Image)</div>
            <div className="attention-arrows">
              <span className="arrow-right">V queries T &rarr;</span>
              <span className="arrow-left">&larr; T queries V</span>
            </div>
            <div className="stream text-stream">T (Text)</div>
          </div>
        </div>

        <div className="fusion-connector"></div>

        <div className="fusion-stage">
          <div className="stage-title">Stage 2: Fine Local</div>
          <div className="cross-attention-box">
            <div className="stream visual-stream">V' (Refined)</div>
            <div className="attention-arrows">
              <span className="arrow-right">V' queries T' &rarr;</span>
              <span className="arrow-left">&larr; T' queries V'</span>
            </div>
            <div className="stream text-stream">T' (Refined)</div>
          </div>
        </div>

        <div className="fusion-connector"></div>

        <div className="fusion-stage final-stage">
          <div className="stage-title">Stage 3: Gated Aggregation</div>
          <div className="gated-aggregation">
            <div className="gate-box">
              <small>Visual Gate (g_v)</small>
              <div className="progress-bar"><div className="fill" style={{width: '65%', background: 'var(--accent-primary)'}}></div></div>
              <span>0.65</span>
            </div>
            <div className="math-symbol">+</div>
            <div className="gate-box">
              <small>Text Gate (g_t)</small>
              <div className="progress-bar"><div className="fill" style={{width: '82%', background: 'var(--accent-primary)'}}></div></div>
              <span>0.82</span>
            </div>
          </div>
          <div className="fused-output">
            <Network size={16} strokeWidth={1.5} />
            <span>F_fused (768-dim)</span>
          </div>
        </div>
        
      </div>
    </div>
  );
};

export default FusionVisualizer;
