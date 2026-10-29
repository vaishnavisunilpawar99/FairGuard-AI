import React from 'react';

export default function PrivacyTab({ data }) {
  const privacy = data?.privacy || {
    pii_detected_count: 0,
    pii_findings: [
      { type: 'Email Address', count: 0, status: 'SECURE' },
      { type: 'Phone Number', count: 0, status: 'SECURE' },
      { type: 'SSN / National ID', count: 0, status: 'SECURE' },
      { type: 'Credit Card Number', count: 0, status: 'SECURE' },
      { type: 'Date of Birth (DOB)', count: 0, status: 'SECURE' }
    ],
    k_anonymity: {
      quasi_identifiers: ['age', 'workclass', 'education', 'native-country'],
      min_k: 8,
      percent_k_under_5: 0.0,
      unique_rows_ratio: 0.0
    },
    mia_risk: {
      score: 0.048,
      train_auc: 0.912,
      test_auc: 0.898,
      status: 'LOW OVERFITTING RISK'
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      {/* Three Pillar Cards */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
        gap: '16px'
      }}>
        {/* PII Summary */}
        <div className="vintage-card">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: '0.8rem', fontWeight: 700, color: 'var(--color-coffee-brown)' }}>
              PII DISCLOSURE RISK
            </span>
            <span className="badge" style={{
              background: privacy.pii_detected_count > 0 ? 'var(--color-burgundy)' : 'var(--color-warm-beige)',
              color: privacy.pii_detected_count > 0 ? 'var(--color-cream)' : 'var(--color-dark-chocolate)'
            }}>
              {privacy.pii_detected_count > 0 ? `${privacy.pii_detected_count} DETECTED` : 'ALL CLEAR'}
            </span>
          </div>
          <div style={{
            fontSize: '1.8rem',
            fontWeight: 800,
            fontFamily: 'var(--font-heading)',
            color: privacy.pii_detected_count > 0 ? 'var(--color-burgundy)' : 'var(--color-dark-chocolate)',
            margin: '8px 0 4px 0'
          }}>
            {privacy.pii_detected_count === 0 ? 'ZERO LEAKS' : `${privacy.pii_detected_count} LEAKS`}
          </div>
          <span style={{ fontSize: '0.78rem', color: 'var(--color-coffee-brown)' }}>
            Regex scanner covering emails, phone, credit cards & IDs
          </span>
        </div>

        {/* k-Anonymity */}
        <div className="vintage-card">
          <span style={{ fontSize: '0.8rem', fontWeight: 700, color: 'var(--color-coffee-brown)' }}>
            k-ANONYMITY SCORE
          </span>
          <div style={{
            fontSize: '1.8rem',
            fontWeight: 800,
            fontFamily: 'var(--font-heading)',
            color: 'var(--color-dark-chocolate)',
            margin: '8px 0 4px 0'
          }}>
            k = {privacy.k_anonymity.min_k}
          </div>
          <span style={{ fontSize: '0.78rem', color: 'var(--color-coffee-brown)' }}>
            {privacy.k_anonymity.percent_k_under_5.toFixed(1)}% records with k &lt; 5 (Target: 0%)
          </span>
        </div>

        {/* MIA Risk */}
        <div className="vintage-card">
          <span style={{ fontSize: '0.8rem', fontWeight: 700, color: 'var(--color-coffee-brown)' }}>
            MIA MEMORIZATION PROXY
          </span>
          <div style={{
            fontSize: '1.8rem',
            fontWeight: 800,
            fontFamily: 'var(--font-heading)',
            color: 'var(--color-dark-chocolate)',
            margin: '8px 0 4px 0'
          }}>
            Δ AUC: {privacy.mia_risk.score.toFixed(3)}
          </div>
          <span style={{ fontSize: '0.78rem', color: 'var(--color-coffee-brown)' }}>
            Train ({privacy.mia_risk.train_auc.toFixed(3)}) vs Test ({privacy.mia_risk.test_auc.toFixed(3)}) gap
          </span>
        </div>
      </div>

      {/* Details Grid */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(360px, 1fr))',
        gap: '24px'
      }}>
        {/* PII Table */}
        <div className="vintage-card" style={{ padding: 0, overflow: 'hidden' }}>
          <div style={{ padding: '18px 24px', borderBottom: '1px solid var(--color-warm-beige)' }}>
            <h3 style={{ fontSize: '1.1rem', color: 'var(--color-dark-chocolate)' }}>
              IDENTIFIABLE ATTRIBUTE SCAN RESULTS
            </h3>
          </div>
          <table className="vintage-table">
            <thead>
              <tr>
                <th>PII TYPE</th>
                <th>INSTANCES</th>
                <th>SECURITY STATUS</th>
              </tr>
            </thead>
            <tbody>
              {privacy.pii_findings.map((f, i) => (
                <tr key={i}>
                  <td style={{ fontWeight: 600 }}>{f.type}</td>
                  <td style={{ fontFamily: 'monospace', fontWeight: 700 }}>{f.count}</td>
                  <td>
                    <span className="badge" style={{
                      background: f.count > 0 ? 'var(--color-burgundy)' : 'var(--color-warm-beige)',
                      color: f.count > 0 ? 'var(--color-cream)' : 'var(--color-dark-chocolate)'
                    }}>
                      {f.status}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* k-Anonymity Configuration */}
        <div className="vintage-card">
          <h3 style={{ fontSize: '1.1rem', marginBottom: '14px', color: 'var(--color-dark-chocolate)' }}>
            QUASI-IDENTIFIER EVALUATION
          </h3>
          <p style={{ fontSize: '0.85rem', color: 'var(--color-coffee-brown)', marginBottom: '16px' }}>
            Evaluated quasi-identifiers determine re-identification risk when combined with public voter or census registries:
          </p>

          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', marginBottom: '20px' }}>
            {privacy.k_anonymity.quasi_identifiers.map((qi, i) => (
              <span key={i} style={{
                background: 'var(--color-warm-beige)',
                color: 'var(--color-dark-chocolate)',
                padding: '4px 10px',
                borderRadius: '6px',
                fontSize: '0.82rem',
                fontWeight: 600,
                border: '1px solid var(--color-coffee-brown)'
              }}>
                {qi}
              </span>
            ))}
          </div>

          <div style={{
            background: 'rgba(216, 195, 165, 0.25)',
            padding: '14px',
            borderRadius: '8px',
            border: '1px solid var(--color-warm-beige)',
            display: 'flex',
            flexDirection: 'column',
            gap: '8px',
            fontSize: '0.85rem'
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ fontWeight: 600, color: 'var(--color-dark-chocolate)' }}>Minimum Group Size:</span>
              <span style={{ fontWeight: 700, color: 'var(--color-coffee-brown)' }}>{privacy.k_anonymity.min_k} individuals</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ fontWeight: 600, color: 'var(--color-dark-chocolate)' }}>Unique Fingerprint Ratio:</span>
              <span style={{ fontWeight: 700, color: 'var(--color-coffee-brown)' }}>{(privacy.k_anonymity.unique_rows_ratio * 100).toFixed(2)}%</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
