import React from 'react';

export default function TabNavigation({ tabs, activeTab, onSelectTab }) {
  return (
    <div style={{
      display: 'flex',
      gap: '8px',
      overflowX: 'auto',
      marginBottom: '24px',
      borderBottom: '2px solid var(--color-warm-beige)',
      paddingBottom: '8px'
    }}>
      {tabs.map(tab => {
        const isActive = activeTab === tab.id;
        return (
          <button
            key={tab.id}
            onClick={() => onSelectTab(tab.id)}
            style={{
              background: isActive ? 'var(--color-burgundy)' : 'transparent',
              color: isActive ? 'var(--color-cream)' : 'var(--color-dark-chocolate)',
              border: isActive ? '1px solid var(--color-burgundy)' : '1px solid transparent',
              borderRadius: '8px',
              padding: '10px 18px',
              fontSize: '0.88rem',
              fontWeight: 700,
              letterSpacing: '0.5px',
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              whiteSpace: 'nowrap',
              cursor: 'pointer',
              transition: 'all 0.15s ease'
            }}
          >
            <span>{tab.icon}</span>
            <span>{tab.label}</span>
          </button>
        );
      })}
    </div>
  );
}
