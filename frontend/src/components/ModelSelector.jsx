import React from 'react';

export default function ModelSelector({ models = [], activeModel, onSelectModel }) {
  return (
    <div style={{
      display: 'flex',
      flexWrap: 'wrap',
      alignItems: 'center',
      gap: '12px',
      marginBottom: '24px',
      background: 'var(--color-cream)',
      border: '1px solid var(--color-warm-beige)',
      padding: '12px 18px',
      borderRadius: '10px'
    }}>
      <span style={{
        fontFamily: 'var(--font-heading)',
        fontSize: '0.85rem',
        fontWeight: 700,
        color: 'var(--color-dark-chocolate)',
        letterSpacing: '0.5px'
      }}>
        ACTIVE AUDIT TARGET:
      </span>

      <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
        {models.map(m => {
          const isSelected = activeModel === m.id;
          return (
            <button
              key={m.id}
              onClick={() => onSelectModel(m.id)}
              style={{
                background: isSelected ? 'var(--color-burgundy)' : 'var(--color-warm-beige)',
                color: isSelected ? 'var(--color-cream)' : 'var(--color-dark-chocolate)',
                border: `1px solid ${isSelected ? 'var(--color-burgundy)' : 'var(--color-coffee-brown)'}`,
                padding: '6px 14px',
                borderRadius: '6px',
                fontSize: '0.82rem',
                fontWeight: 700,
                cursor: 'pointer',
                transition: 'all 0.15s ease'
              }}
            >
              {m.name}
            </button>
          );
        })}
      </div>
    </div>
  );
}
