import React from 'react';

export default function Header({ 
  datasets = [], 
  selectedDataset, 
  onSelectDataset, 
  backendOnline, 
  isAssessing, 
  onRunAssessment 
}) {
  return (
    <header style={{
      background: 'var(--color-dark-chocolate)',
      color: 'var(--color-cream)',
      borderBottom: '3px solid var(--color-burgundy)',
      padding: '20px 32px',
      boxShadow: 'var(--shadow-md)'
    }}>
      <div style={{
        maxWidth: '1400px',
        margin: '0 auto',
        display: 'flex',
        flexWrap: 'wrap',
        alignItems: 'center',
        justifyContent: 'space-between',
        gap: '20px'
      }}>
        {/* Branding */}
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <div style={{
              width: '36px',
              height: '36px',
              borderRadius: '8px',
              background: 'var(--color-burgundy)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              border: '1px solid var(--color-warm-beige)'
            }}>
              <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="var(--color-cream)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
                <path d="m9 12 2 2 4-4"/>
              </svg>
            </div>
            <div>
              <h1 style={{ 
                margin: 0, 
                fontSize: '1.65rem', 
                color: 'var(--color-cream)',
                letterSpacing: '1px'
              }}>
                FAIRGUARD AI
              </h1>
              <p style={{ 
                margin: 0, 
                fontSize: '0.82rem', 
                color: 'var(--color-warm-beige)',
                letterSpacing: '0.5px'
              }}>
                RESPONSIBLE AI AUDITING, EXPLAINABILITY & GOVERNANCE PLATFORM
              </p>
            </div>
          </div>
        </div>

        {/* Controls */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px', flexWrap: 'wrap' }}>
          {/* Status Badge */}
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            background: 'rgba(216, 195, 165, 0.15)',
            border: '1px solid var(--color-coffee-brown)',
            padding: '6px 14px',
            borderRadius: '20px',
            fontSize: '0.75rem',
            fontWeight: 600,
            letterSpacing: '0.5px',
            color: 'var(--color-warm-beige)'
          }}>
            <span style={{
              width: '8px',
              height: '8px',
              borderRadius: '50%',
              background: backendOnline ? 'var(--color-warm-beige)' : 'var(--color-burgundy)',
              boxShadow: backendOnline ? '0 0 8px var(--color-warm-beige)' : 'none'
            }} />
            {backendOnline ? 'FASTAPI REST BACKEND ACTIVE' : 'LOCAL ENGINE STANDALONE'}
          </div>

          {/* Dataset Selector */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span style={{ fontSize: '0.8rem', color: 'var(--color-warm-beige)', fontWeight: 600 }}>
              DATASET:
            </span>
            <select
              value={selectedDataset}
              onChange={(e) => onSelectDataset(e.target.value)}
              style={{
                background: 'var(--color-cream)',
                color: 'var(--color-dark-chocolate)',
                border: '1px solid var(--color-warm-beige)',
                borderRadius: '6px',
                padding: '6px 12px',
                fontSize: '0.85rem',
                fontWeight: 600,
                cursor: 'pointer'
              }}
            >
              {datasets.map(d => (
                <option key={d.id} value={d.id}>{d.name}</option>
              ))}
            </select>
          </div>

          {/* Action Button */}
          <button
            onClick={onRunAssessment}
            disabled={isAssessing}
            style={{
              background: isAssessing ? 'var(--color-coffee-brown)' : 'var(--color-burgundy)',
              color: 'var(--color-cream)',
              border: '1px solid var(--color-warm-beige)',
              padding: '8px 18px',
              borderRadius: '6px',
              fontSize: '0.85rem',
              fontWeight: 700,
              letterSpacing: '0.5px',
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              cursor: isAssessing ? 'not-allowed' : 'pointer'
            }}
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <path d="M21.5 2v6h-6M21.34 15.57a10 10 0 1 1-.57-8.38l5.67-5.67"/>
            </svg>
            {isAssessing ? 'RUNNING AUDIT...' : 'RE-RUN AUDIT'}
          </button>
        </div>
      </div>
    </header>
  );
}
