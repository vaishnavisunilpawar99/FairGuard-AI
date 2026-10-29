import React, { useState } from 'react';

export default function FairnessTab({ data }) {
  const [selectedAttr, setSelectedAttr] = useState('sex');

  const fairness = data?.fairness || {
    disparate_impact_ratio: 0.312,
    demographic_parity_difference: 0.228,
    equalized_odds_difference: 0.185,
    four_fifths_rule_passed: false,
    groups: {
      sex: {
        privileged: 'Male',
        unprivileged: 'Female',
        selection_rates: { 'Male': 0.315, 'Female': 0.114 },
        tpr: { 'Male': 0.720, 'Female': 0.582 },
        fpr: { 'Male': 0.125, 'Female': 0.048 }
      },
      age_group: {
        privileged: 'Adult (>=25)',
        unprivileged: 'Young (<25)',
        selection_rates: { 'Adult (>=25)': 0.285, 'Young (<25)': 0.082 },
        tpr: { 'Adult (>=25)': 0.705, 'Young (<25)': 0.510 },
        fpr: { 'Adult (>=25)': 0.110, 'Young (<25)': 0.035 }
      }
    }
  };

  const groupData = fairness.groups?.[selectedAttr] || fairness.groups?.sex;
  const passed80 = fairness.disparate_impact_ratio >= 0.80;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      {/* Overview Cards */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))',
        gap: '16px'
      }}>
        {/* Disparate Impact Card */}
        <div className="vintage-card">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: '0.8rem', fontWeight: 700, color: 'var(--color-coffee-brown)' }}>
              DISPARATE IMPACT RATIO
            </span>
            <span className="badge" style={{
              background: passed80 ? 'var(--color-warm-beige)' : 'var(--color-burgundy)',
              color: passed80 ? 'var(--color-dark-chocolate)' : 'var(--color-cream)'
            }}>
              {passed80 ? 'PASSED 80% RULE' : 'VIOLATES 80% RULE'}
            </span>
          </div>
          <div style={{
            fontSize: '2rem',
            fontWeight: 800,
            fontFamily: 'var(--font-heading)',
            color: passed80 ? 'var(--color-dark-chocolate)' : 'var(--color-burgundy)',
            margin: '8px 0 4px 0'
          }}>
            {fairness.disparate_impact_ratio.toFixed(3)}
          </div>
          <span style={{ fontSize: '0.78rem', color: 'var(--color-coffee-brown)' }}>
            Threshold standard: ≥ 0.80 (Selection Rate Ratio: {((fairness.disparate_impact_ratio) * 100).toFixed(1)}%)
          </span>
        </div>

        {/* Demographic Parity Difference */}
        <div className="vintage-card">
          <span style={{ fontSize: '0.8rem', fontWeight: 700, color: 'var(--color-coffee-brown)' }}>
            DEMOGRAPHIC PARITY DIFF
          </span>
          <div style={{
            fontSize: '2rem',
            fontWeight: 800,
            fontFamily: 'var(--font-heading)',
            color: 'var(--color-dark-chocolate)',
            margin: '8px 0 4px 0'
          }}>
            {fairness.demographic_parity_difference.toFixed(3)}
          </div>
          <span style={{ fontSize: '0.78rem', color: 'var(--color-coffee-brown)' }}>
            Max absolute difference in acceptance rates across groups
          </span>
        </div>

        {/* Equalized Odds Difference */}
        <div className="vintage-card">
          <span style={{ fontSize: '0.8rem', fontWeight: 700, color: 'var(--color-coffee-brown)' }}>
            EQUALIZED ODDS DIFF
          </span>
          <div style={{
            fontSize: '2rem',
            fontWeight: 800,
            fontFamily: 'var(--font-heading)',
            color: 'var(--color-dark-chocolate)',
            margin: '8px 0 4px 0'
          }}>
            {fairness.equalized_odds_difference.toFixed(3)}
          </div>
          <span style={{ fontSize: '0.78rem', color: 'var(--color-coffee-brown)' }}>
            Disparity across both True Positive Rate & False Positive Rate
          </span>
        </div>
      </div>

      {/* Subgroup Analysis Section */}
      <div className="vintage-card">
        <div style={{
          display: 'flex',
          flexWrap: 'wrap',
          justifyContent: 'space-between',
          alignItems: 'center',
          gap: '12px',
          marginBottom: '20px',
          paddingBottom: '16px',
          borderBottom: '1px solid var(--color-warm-beige)'
        }}>
          <div>
            <h3 style={{ fontSize: '1.15rem', color: 'var(--color-dark-chocolate)' }}>
              SUBGROUP RATE COMPARISON
            </h3>
            <span style={{ fontSize: '0.82rem', color: 'var(--color-coffee-brown)' }}>
              Comparing Privileged ({groupData?.privileged}) vs Unprivileged ({groupData?.unprivileged}) cohorts
            </span>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span style={{ fontSize: '0.82rem', fontWeight: 700, color: 'var(--color-dark-chocolate)' }}>
              AUDIT ATTRIBUTE:
            </span>
            <select
              value={selectedAttr}
              onChange={(e) => setSelectedAttr(e.target.value)}
              className="input-select"
            >
              <option value="sex">Gender (Sex)</option>
              <option value="age_group">Age Group</option>
            </select>
          </div>
        </div>

        {/* Comparison Bars */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          {/* Selection Rates */}
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
              <span style={{ fontWeight: 700, fontSize: '0.88rem', color: 'var(--color-dark-chocolate)' }}>
                Positive Outcome / Selection Rate (P(Y_hat = 1))
              </span>
            </div>
            {Object.entries(groupData?.selection_rates || {}).map(([grp, rate], idx) => (
              <div key={idx} style={{ marginBottom: '10px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.82rem', marginBottom: '4px' }}>
                  <span style={{ fontWeight: 600, color: 'var(--color-dark-chocolate)' }}>{grp}</span>
                  <span style={{ fontWeight: 700, color: 'var(--color-coffee-brown)' }}>{(rate * 100).toFixed(1)}%</span>
                </div>
                <div style={{ width: '100%', height: '10px', background: 'var(--color-warm-beige)', borderRadius: '5px', overflow: 'hidden' }}>
                  <div style={{
                    width: `${rate * 100}%`,
                    height: '100%',
                    background: idx === 0 ? 'var(--color-dark-chocolate)' : 'var(--color-burgundy)'
                  }} />
                </div>
              </div>
            ))}
          </div>

          {/* TPR & FPR Comparisons */}
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
            gap: '20px',
            marginTop: '10px'
          }}>
            {/* TPR */}
            <div style={{ background: 'rgba(216, 195, 165, 0.25)', padding: '16px', borderRadius: '8px', border: '1px solid var(--color-warm-beige)' }}>
              <span style={{ fontSize: '0.82rem', fontWeight: 700, color: 'var(--color-dark-chocolate)', display: 'block', marginBottom: '10px' }}>
                TRUE POSITIVE RATE (EQUALITY OF OPPORTUNITY)
              </span>
              {Object.entries(groupData?.tpr || {}).map(([grp, val], idx) => (
                <div key={idx} style={{ marginBottom: '8px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', marginBottom: '2px' }}>
                    <span style={{ color: 'var(--color-dark-chocolate)' }}>{grp}</span>
                    <span style={{ fontWeight: 700, color: 'var(--color-coffee-brown)' }}>{(val * 100).toFixed(1)}%</span>
                  </div>
                  <div style={{ width: '100%', height: '8px', background: 'var(--color-warm-beige)', borderRadius: '4px', overflow: 'hidden' }}>
                    <div style={{ width: `${val * 100}%`, height: '100%', background: 'var(--color-coffee-brown)' }} />
                  </div>
                </div>
              ))}
            </div>

            {/* FPR */}
            <div style={{ background: 'rgba(216, 195, 165, 0.25)', padding: '16px', borderRadius: '8px', border: '1px solid var(--color-warm-beige)' }}>
              <span style={{ fontSize: '0.82rem', fontWeight: 700, color: 'var(--color-dark-chocolate)', display: 'block', marginBottom: '10px' }}>
                FALSE POSITIVE RATE (PREDICTIVE EQUALITY)
              </span>
              {Object.entries(groupData?.fpr || {}).map(([grp, val], idx) => (
                <div key={idx} style={{ marginBottom: '8px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', marginBottom: '2px' }}>
                    <span style={{ color: 'var(--color-dark-chocolate)' }}>{grp}</span>
                    <span style={{ fontWeight: 700, color: 'var(--color-coffee-brown)' }}>{(val * 100).toFixed(1)}%</span>
                  </div>
                  <div style={{ width: '100%', height: '8px', background: 'var(--color-warm-beige)', borderRadius: '4px', overflow: 'hidden' }}>
                    <div style={{ width: `${val * 100}%`, height: '100%', background: 'var(--color-coffee-brown)' }} />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
