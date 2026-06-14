import React from 'react';
import { FileText, AlignLeft } from 'lucide-react';
import './TextReportPanel.css';

const TextReportPanel = () => {
  return (
    <div className="text-panel minimal-panel">
      <div className="panel-header">
        <div className="flex-center" style={{ gap: '12px' }}>
          <FileText size={18} strokeWidth={1.5} color="var(--accent-primary)" />
          <h3>Module 2: Clinical Text</h3>
        </div>
      </div>

      <div className="report-content">
        <h4 className="report-title">Radiology Report</h4>
        <p className="clinical-text">
          <span className="highlight-text">
            Multiple hyperintense white matter lesions
          </span> seen on T2 and FLAIR sequences, primarily distributed in the 
          periventricular and juxtacortical regions. 
          The largest lesion measures <span className="highlight-text secondary">approx 8mm</span> in the left centrum semiovale. 
          No abnormal contrast enhancement noted. 
          Findings are highly suggestive of demyelinating disease.
        </p>
      </div>

      <div className="encoder-visualization">
        <h4>BioBERT Encoding</h4>
        <div className="bert-flow">
          <div className="flow-step text-stream">
            <AlignLeft size={16} strokeWidth={1.5} />
            <span>Tokens</span>
          </div>
          <div className="flow-arrow">→</div>
          <div className="flow-step bert-core">
            <span>BioBERT Layers</span>
            <small>LoRA Rank=8</small>
          </div>
          <div className="flow-arrow">→</div>
          <div className="flow-outputs">
            <div className="flow-output-badge">[CLS] Global</div>
            <div className="flow-output-badge">Token Local</div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TextReportPanel;
