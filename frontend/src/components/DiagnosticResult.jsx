import React from 'react';
import { Activity, AlertTriangle, CheckCircle2 } from 'lucide-react';
import './DiagnosticResult.css';

const DiagnosticResult = () => {
  return (
    <div className="diagnostic-panel minimal-panel">
      <div className="panel-header">
        <div className="flex-center" style={{ gap: '12px' }}>
          <Activity size={18} strokeWidth={1.5} color="var(--accent-primary)" />
          <h3>Module 5: Diagnostic Output</h3>
        </div>
      </div>

      <div className="results-container">
        
        <div className="primary-diagnosis">
          <div className="diagnosis-status">
            <AlertTriangle size={20} strokeWidth={1.5} color="var(--accent-primary)" />
          </div>
          <div className="diagnosis-info">
            <h2>Demyelinating Disease</h2>
            <p>Confidence: <span className="confidence-high">94.2%</span></p>
          </div>
        </div>

        <div className="predictions-list">
          <h4>Multi-Label Classification</h4>
          
          <div className="prediction-item">
            <div className="pred-info">
              <span>Multiple Sclerosis (MS)</span>
              <span>88.5%</span>
            </div>
            <div className="pred-bar"><div className="fill" style={{width: '88.5%', background: 'var(--accent-primary)'}}></div></div>
          </div>
          
          <div className="prediction-item">
            <div className="pred-info">
              <span className="muted-text">Ischemic Stroke</span>
              <span className="muted-text">12.1%</span>
            </div>
            <div className="pred-bar"><div className="fill" style={{width: '12.1%', background: 'var(--border-highlight)'}}></div></div>
          </div>
          
          <div className="prediction-item">
            <div className="pred-info">
              <span className="muted-text">Brain Tumor (Glioma)</span>
              <span className="muted-text">0.4%</span>
            </div>
            <div className="pred-bar"><div className="fill" style={{width: '0.4%', background: 'var(--border-highlight)'}}></div></div>
          </div>
        </div>

        <div className="explainability-section">
          <h4>Explainability (Alignment)</h4>
          <div className="explanation-cards">
            <div className="exp-card">
              <span className="exp-label">Visual Trigger</span>
              <p>Periventricular Hyperintensities (FLAIR)</p>
            </div>
            <div className="exp-card">
              <span className="exp-label">Semantic Trigger</span>
              <p>"white matter lesions", "demyelinating"</p>
            </div>
          </div>
        </div>

        <button className="button-primary finalize-btn">
          Confirm & Finalize Report
        </button>
        
      </div>
    </div>
  );
};

export default DiagnosticResult;
