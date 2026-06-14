import React from 'react';
import { Calendar, User, FileText } from 'lucide-react';
import './PatientHeader.css';

const PatientHeader = () => {
  return (
    <div className="patient-header minimal-panel">
      <div className="patient-main-info">
        <div className="patient-avatar">
          <User size={32} strokeWidth={1} color="var(--accent-primary)" />
        </div>
        <div className="patient-details">
          <h1>John Doe</h1>
          <p className="patient-meta">ID: PT-84729 &nbsp;&bull;&nbsp; Male &nbsp;&bull;&nbsp; 58 years</p>
        </div>
      </div>
      
      <div className="patient-stats">
        <div className="stat-card">
          <Calendar size={16} strokeWidth={1.5} className="stat-icon" />
          <div className="stat-info">
            <span className="stat-label">Scan Date</span>
            <span className="stat-value">Oct 24, 2026</span>
          </div>
        </div>
        <div className="stat-card">
          <FileText size={16} strokeWidth={1.5} className="stat-icon" />
          <div className="stat-info">
            <span className="stat-label">Referral</span>
            <span className="stat-value">Dr. A. Smith (Neurology)</span>
          </div>
        </div>
        <div className="stat-action">
          <button className="button-primary">Generate Report</button>
        </div>
      </div>
    </div>
  );
};

export default PatientHeader;
