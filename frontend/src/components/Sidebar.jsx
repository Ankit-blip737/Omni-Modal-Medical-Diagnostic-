import React from 'react';
import { LayoutDashboard, Users, Brain, Activity, Settings, LogOut } from 'lucide-react';
import './Sidebar.css';

const Sidebar = () => {
  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <div className="logo-icon flex-center">
          <Brain size={24} color="var(--accent-primary)" />
        </div>
        <h2 className="logo-text">OmniDiag</h2>
      </div>

      <nav className="sidebar-nav">
        <a href="#" className="nav-item active">
          <LayoutDashboard size={18} strokeWidth={1.5} />
          <span>Dashboard</span>
        </a>
        <a href="#" className="nav-item">
          <Users size={18} strokeWidth={1.5} />
          <span>Patients</span>
        </a>
        <a href="#" className="nav-item">
          <Activity size={18} strokeWidth={1.5} />
          <span>Diagnostics</span>
        </a>
        <a href="#" className="nav-item">
          <Brain size={18} strokeWidth={1.5} />
          <span>Model Analysis</span>
        </a>
        <a href="#" className="nav-item">
          <Settings size={18} strokeWidth={1.5} />
          <span>Settings</span>
        </a>
      </nav>

      <div className="sidebar-footer">
        <div className="user-profile">
          <div className="avatar">Dr</div>
          <div className="user-info">
            <p className="user-name">Dr. Sarah Chen</p>
            <p className="user-role">Lead Radiologist</p>
          </div>
        </div>
        <button className="logout-btn">
          <LogOut size={16} strokeWidth={1.5} />
        </button>
      </div>
    </aside>
  );
};

export default Sidebar;
