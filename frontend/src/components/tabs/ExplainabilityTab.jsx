import React, { useState } from 'react';

export default function ExplainabilityTab({ data }) {
  const [selectedSample, setSelectedSample] = useState(0);

  const globalImportance = data?.feature_importance || [
    { feature: 'capital-gain', importance: 0.28 },
    { feature: 'education-num', importance: 0.22 },
    { feature: 'age', importance: 0.16 },
    { feature: 'hours-per-week', importance: 0.12 },
    { feature: 'marital-status', importance: 0.09 },
    { feature: 'occupation', importance: 0.06 },
    { feature: 'relationship', importance: 0.04 },
    { feature: 'fnlwgt', importance: 0.03 }
  ];

  const samples = [
    {
      id: '#4102',
      profile: 'Age: 48, Education: Bachelors, Hours: 45/wk, Capital Gain: $5,000',
      pred_prob: 0.78,
      base_val: 0.24,
      contributions: [
        { feature: 'capital-gain = $5,000', delta: +0.28, positive: true },
        { feature: 'education-num = 13 (Bachelors)', delta: +0.18, positive: true },
        { feature: 'age = 48', delta: +0.11, positive: true },
        { feature: 'hours-per-week = 45', delta: +0.05, positive: true },
        { feature: 'marital-status = Never-married', delta: -0.08, positive: false }
      ]
    },
    {
      id: '#7819',
      profile: 'Age: 22, Education: HS-grad, Hours: 35/wk, Capital Gain: $0',
      pred_prob: 0.12,
      base_val: 0.24,
      contributions: [
        { feature: 'age = 22', delta: -0.07, positive: false },
        { feature: 'education-num = 9 (HS-grad)', delta: -0.05, positive: false },
        { feature: 'capital-gain = $0', delta: -0.02, positive: false },
        { feature: 'hours-per-week = 35', delta: +0.02, positive: true }
      ]
    }
  ];

  const currentSample = samples[selectedSample] || samples[0];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      {/* Global Feature Importance Card */}
      <div className="vintage-card">
        <div style={{ marginBottom: '18px', paddingBottom: '14px', borderBottom: '1px solid var(--color-warm-beige)' }}>
          <h3 style={{ fontSize: '1.15rem', color: 'var(--color-dark-chocolate)' }}>
            GLOBAL FEATURE IMPORTANCE (MEAN |SHAP| ATTRIBUTION)
          </h3>
          <p style={{ margin: 0, fontSize: '0.82rem', color: 'var(--color-coffee-brown)' }}>
            Ranked model impact across test population. Identifies key decision drivers and checks for proxy variables.
          </p>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
          {globalImportance.map((item, idx) => {
            const pct = (item.importance / globalImportance[0].importance) * 100;
            return (
              <div key={idx}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.84rem', marginBottom: '4px' }}>
                  <span style={{ fontWeight: 600, color: 'var(--color-dark-chocolate)' }}>
                    {idx + 1}. {item.feature}
                  </span>
                  <span style={{ fontWeight: 700, color: 'var(--color-coffee-brown)' }}>
                    {item.importance.toFixed(3)}
                  </span>
                </div>
                <div style={{ width: '100%', height: '10px', background: 'var(--color-warm-beige)', borderRadius: '5px', overflow: 'hidden' }}>
                  <div style={{
                    width: `${pct}%`,
                    height: '100%',
                    background: idx < 2 ? 'var(--color-burgundy)' : 'var(--color-coffee-brown)'
                  }} />
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Local Sample Waterfall Section */}
      <div className="vintage-card">
        <div style={{
          display: 'flex',
          flexWrap: 'wrap',
          justifyContent: 'space-between',
          alignItems: 'center',
          gap: '12px',
          marginBottom: '20px',
          paddingBottom: '14px',
          borderBottom: '1px solid var(--color-warm-beige)'
        }}>
          <div>
            <h3 style={{ fontSize: '1.15rem', color: 'var(--color-dark-chocolate)' }}>
              LOCAL SHAP WATERFALL EXPLANATION
            </h3>
            <span style={{ fontSize: '0.82rem', color: 'var(--color-coffee-brown)' }}>
              Decomposition of individual applicant scoring decision into additive feature contributions
            </span>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span style={{ fontSize: '0.82rem', fontWeight: 700, color: 'var(--color-dark-chocolate)' }}>
              SAMPLE RECORD:
            </span>
            <select
              value={selectedSample}
              onChange={(e) => setSelectedSample(Number(e.target.value))}
              className="input-select"
            >
              {samples.map((s, idx) => (
                <option key={idx} value={idx}>{s.id} ({s.profile.split(',')[0]})</option>
              ))}
            </select>
          </div>
        </div>

        {/* Profile Pill & Score */}
        <div style={{
          background: 'var(--color-warm-beige)',
          padding: '14px 18px',
          borderRadius: '8px',
          marginBottom: '20px',
          display: 'flex',
          flexWrap: 'wrap',
          justifyContent: 'space-between',
          alignItems: 'center',
          gap: '12px'
        }}>
          <div>
            <span style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--color-coffee-brown)', display: 'block' }}>
              APPLICANT PROFILE
            </span>
            <span style={{ fontSize: '0.9rem', fontWeight: 600, color: 'var(--color-dark-chocolate)' }}>
              {currentSample.profile}
            </span>
          </div>
          <div style={{ textAlign: 'right' }}>
            <span style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--color-coffee-brown)', display: 'block' }}>
              MODEL PREDICTED PROBABILITY f(x)
            </span>
            <span style={{ fontSize: '1.4rem', fontWeight: 800, fontFamily: 'var(--font-heading)', color: 'var(--color-burgundy)' }}>
              {(currentSample.pred_prob * 100).toFixed(1)}%
            </span>
          </div>
        </div>

        {/* Contributions Waterfall List */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
          <div style={{
            display: 'flex',
            justifyContent: 'space-between',
            padding: '8px 12px',
            background: 'rgba(62, 39, 35, 0.05)',
            borderRadius: '6px',
            fontSize: '0.82rem',
            fontWeight: 700,
            color: 'var(--color-coffee-brown)'
          }}>
            <span>Base Model Population Expectation E[f(X)]</span>
            <span>{(currentSample.base_val * 100).toFixed(1)}%</span>
          </div>

          {currentSample.contributions.map((c, i) => (
            <div key={i} style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              padding: '10px 14px',
              border: `1px solid ${c.positive ? 'var(--color-burgundy)' : 'var(--color-coffee-brown)'}`,
              borderRadius: '6px',
              background: c.positive ? 'rgba(107, 44, 44, 0.08)' : 'rgba(139, 107, 74, 0.08)'
            }}>
              <span style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--color-dark-chocolate)' }}>
                {c.feature}
              </span>
              <span style={{
                fontSize: '0.85rem',
                fontWeight: 800,
                color: c.positive ? 'var(--color-burgundy)' : 'var(--color-coffee-brown)'
              }}>
                {c.delta > 0 ? `+${(c.delta * 100).toFixed(1)}%` : `${(c.delta * 100).toFixed(1)}%`}
              </span>
            </div>
          ))}

          <div style={{
            display: 'flex',
            justifyContent: 'space-between',
            padding: '12px 14px',
            background: 'var(--color-dark-chocolate)',
            color: 'var(--color-cream)',
            borderRadius: '6px',
            fontSize: '0.9rem',
            fontWeight: 800,
            marginTop: '4px'
          }}>
            <span>Final Prediction Output f(x)</span>
            <span style={{ color: 'var(--color-warm-beige)' }}>{(currentSample.pred_prob * 100).toFixed(1)}%</span>
          </div>
        </div>
      </div>
    </div>
  );
}
