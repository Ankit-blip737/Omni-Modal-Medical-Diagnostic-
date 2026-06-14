import React from 'react';
import { Outlet, NavLink, useLocation } from 'react-router-dom';
import './Layout.css';

const Layout = () => {
  const { pathname } = useLocation();
  const isHome = pathname === '/';

  return (
    <>
      <nav className={`navbar ${isHome ? 'navbar-transparent' : ''}`}>
        <NavLink to="/" className="navbar-brand">
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M12 2a8 8 0 0 0-8 8c0 3.4 2.1 6.3 5 7.5V20h6v-2.5c2.9-1.2 5-4.1 5-7.5a8 8 0 0 0-8-8z"/><path d="M10 20v2h4v-2"/><circle cx="12" cy="10" r="2"/></svg>
          OmniDiag
        </NavLink>
        <ul className="navbar-links">
          <li><NavLink to="/">Home</NavLink></li>
          <li><NavLink to="/architecture">Architecture</NavLink></li>
          <li><NavLink to="/training">Training</NavLink></li>
          <li><NavLink to="/results">Results</NavLink></li>
          <li><NavLink to="/datasets">Datasets</NavLink></li>
        </ul>
        <NavLink to="/results" className="navbar-cta">Try Diagnostic</NavLink>
      </nav>
      <main>
        <Outlet />
      </main>
      <footer className="footer">
        <span className="footer-brand">OmniDiag</span>
        <span>Omni-Modal Medical Diagnostic Framework © 2026</span>
      </footer>
    </>
  );
};

export default Layout;
