import React, { useState } from 'react';
import { Maximize2, Layers } from 'lucide-react';
import './ImagingPanel.css';

const ImagingPanel = () => {
  const [activeModality, setActiveModality] = useState('T1');

  const modalities = [
    { id: 'T1', name: 'MRI T1', desc: 'Anatomy Focus' },
    { id: 'T2', name: 'MRI T2', desc: 'Pathology Focus' },
    { id: 'FLAIR', name: 'FLAIR', desc: 'Edema Focus' },
  ];

  return (
    <div className="imaging-panel minimal-panel">
      <div className="panel-header">
        <div className="flex-center" style={{ gap: '12px' }}>
          <Layers size={18} strokeWidth={1.5} color="var(--accent-primary)" />
          <h3>Module 1: Joint Image Encoder</h3>
        </div>
        <button className="button-secondary" style={{ padding: '0.4rem 0.8rem', fontSize: '0.75rem', border: 'none' }}>
          <Maximize2 size={14} style={{ marginRight: '6px' }} /> Expand
        </button>
      </div>

      <div className="modality-tabs">
        {modalities.map(mod => (
          <button
            key={mod.id}
            className={`modality-tab ${activeModality === mod.id ? 'active' : ''}`}
            onClick={() => setActiveModality(mod.id)}
          >
            {mod.name}
          </button>
        ))}
      </div>

      <div className="scan-viewer">
        <div className="scan-image-container">
          <div className="scan-placeholder" data-modality={activeModality}>
            <div className="scan-overlay"></div>
            <div className="scan-crosshair horizontal"></div>
            <div className="scan-crosshair vertical"></div>
            <div className="scan-data-top-left">Axial • Slice 42/128</div>
            <div className="scan-data-bottom-right">W: 1200 L: 400</div>
          </div>
        </div>
        
        <div className="encoder-visualization">
          <h4>Shared-then-Split ConvNeXt</h4>
          <div className="convnext-flow">
            <div className="flow-step shared">
              <span>Shared Stages 1-2</span>
              <small>Low-level features</small>
            </div>
            <div className="flow-arrow">→</div>
            <div className="flow-step split">
              <span>{activeModality} Branch (Stages 3-4)</span>
              <small>High-level specifics</small>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ImagingPanel;
