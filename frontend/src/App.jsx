import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import Home from './pages/Home';
import Architecture from './pages/Architecture';
import Training from './pages/Training';
import Results from './pages/Results';
import Datasets from './pages/Datasets';
import './index.css';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route path="/" element={<Home />} />
          <Route path="/architecture" element={<Architecture />} />
          <Route path="/training" element={<Training />} />
          <Route path="/results" element={<Results />} />
          <Route path="/datasets" element={<Datasets />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;
