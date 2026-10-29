import React from 'react';

export default function DriftTab({ data }) {
  const driftMetrics = data?.drift || {
    overall_drift_detected: false,
    psi_max: 0.142,
    features_drifted_count: 1,
    feature_metrics: [
      { feature: 'age', psi: 0.042, ks_pvalue: 0.421, status: 'STABLE', alert: 'No Shift' },
      { feature: 'hours-per-week', psi: 0.068, ks_pvalue: 0.284, status: 'STABLE', alert: 'No Shift' },
      { feature: 'occupation', psi: 0.142, ks_pvalue: 0.038, status: 'MODERATE', alert: 'Covariate Shift' },
      { feature: 'education-num', psi: 0.031, ks_pvalue: 0.612, status: 'STABLE', alert: 'No Shift' },
      { feature: 'capital-gain', psi: 0.025, ks_pvalue: 0.749, status: 'STABLE', alert: 'No Shift' },
      { feature: 'workclass', psi: 0.054, ks_pvalue: 0.312, status: 'STABLE', alert: 'No Shift' }
    ]
  };

  const getStatusBadge = (status) => {
    switch (status) {
      case 'SIGNIFICANT':
        return { background: 'var(--color-burgundy)', color: 'var(--color-cream)' };
      case 'MODERATE':
        return { background: 'var(--color-coffee-brown)', color: 'var(--color-cream)' };
      default:
        return { background: 'var(--color-warm-beige)', color: 'var(--color-dark-chocolate)' };
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      {/* Summary Cards */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))',
        gap: '16px'
      }}>
        <div className="vintage-card">
          <span style={{ fontSize: '0.8rem', fontWeight: 700, color: 'var(--color-coffee-brown)' }}>
            OVERALL DRIFT STATUS
          </span>
          <div style={{
            fontSize: '1.8rem',
            fontWeight: 800,
            fontFamily: 'var(--font-heading)',
            color: driftMetrics.overall_drift_detected ? 'var(--color-burgundy)' : 'var(--color-dark-chocolate)',
            margin: '8px 0 4px 0'
          }}>
            {driftMetrics.overall_drift_detected ? 'DRIFT DETECTED' : 'POPULATION STABLE'}
          </div>
          <span style={{ fontSize: '0.78rem', color: 'var(--color-coffee-brown)' }}>
            Evaluated using PSI & KS 2-Sample Test
          </span>
        </div>

        <div className="vintage-card">
          <span style={{ fontSize: '0.8rem', fontWeight: 700, color: 'var(--color-coffee-brown)' }}>
            MAX POPULATION STABILITY INDEX (PSI)
          </span>
          <div style={{
            fontSize: '1.8rem',
            fontWeight: 800,
            fontFamily: 'var(--font-heading)',
            color: 'var(--color-dark-chocolate)',
            margin: '8px 0 4px 0'
          }}>
            {driftMetrics.psi_max.toFixed(3)}
          </div>
          <span style={{ fontSize: '0.78rem', color: 'var(--color-coffee-brown)' }}>
            Thresholds: &lt;0.10 Stable, 0.10–0.25 Moderate, &gt;0.25 Significant
          </span>
        </div>

        <div className="vintage-card">
          <span style={{ fontSize: '0.8rem', fontWeight: 700, color: 'var(--color-coffee-brown)' }}>
            FEATURES UNDER SHIFT ALERT
          </span>
          <div style={{
            fontSize: '1.8rem',
            fontWeight: 800,
            fontFamily: 'var(--font-heading)',
            color: driftMetrics.features_drifted_count > 0 ? 'var(--color-coffee-brown)' : 'var(--color-dark-chocolate)',
            margin: '8px 0 4px 0'
          }}>
            {driftMetrics.features_drifted_count} / {driftMetrics.feature_metrics.length}
          </div>
          <span style={{ fontSize: '0.78rem', color: 'var(--color-coffee-brown)' }}>
            Features requiring potential data retraining
          </span>
        </div>
      </div>

      {/* Feature-Level Table */}
      <div className="vintage-card" style={{ padding: 0, overflow: 'hidden' }}>
        <div style={{ padding: '20px 24px', borderBottom: '1px solid var(--color-warm-beige)' }}>
          <h3 style={{ fontSize: '1.15rem', color: 'var(--color-dark-chocolate)' }}>
            FEATURE COVARIATE DRIFT BREAKDOWN
          </h3>
          <span style={{ fontSize: '0.82rem', color: 'var(--color-coffee-brown)' }}>
            Detailed breakdown comparing reference baseline distribution to current evaluation cohort
          </span>
        </div>

        <table className="vintage-table">
          <thead>
            <tr>
              <th>FEATURE NAME</th>
              <th>PSI VALUE</th>
              <th>KS / CHI² P-VALUE</th>
              <th>STABILITY STATUS</th>
              <th>ACTION REQUIRED</th>
            </tr>
          </thead>
          <tbody>
            {driftMetrics.feature_metrics.map((row, i) => (
              <tr key={i}>
                <td style={{ fontWeight: 700, color: 'var(--color-dark-chocolate)' }}>{row.feature}</td>
                <td style={{ fontFamily: 'monospace', fontWeight: 600 }}>{row.psi.toFixed(3)}</td>
                <td style={{ fontFamily: 'monospace' }}>{row.ks_pvalue.toFixed(4)}</td>
                <td>
                  <span className="badge" style={getStatusBadge(row.status)}>
                    {row.status}
                  </span>
                </td>
                <td style={{ color: row.status !== 'STABLE' ? 'var(--color-burgundy)' : 'var(--color-coffee-brown)', fontWeight: 600 }}>
                  {row.alert}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
