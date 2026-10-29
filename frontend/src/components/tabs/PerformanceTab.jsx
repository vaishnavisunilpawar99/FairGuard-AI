import React from 'react';

export default function PerformanceTab({ data }) {
  const metrics = data?.metrics || {
    accuracy: 0.854,
    roc_auc: 0.902,
    f1: 0.718,
    precision: 0.742,
    recall: 0.695,
    log_loss: 0.321
  };

  const cm = data?.confusion_matrix || {
    tn: 11200,
    fp: 800,
    fn: 1400,
    tp: 2600
  };

  const metricCards = [
    { label: 'ACCURACY', val: (metrics.accuracy * 100).toFixed(1) + '%', sub: 'Correct overall predictions' },
    { label: 'ROC-AUC', val: metrics.roc_auc.toFixed(3), sub: 'Separation capability' },
    { label: 'F1 SCORE', val: metrics.f1.toFixed(3), sub: 'Harmonic mean of precision & recall' },
    { label: 'PRECISION', val: metrics.precision.toFixed(3), sub: 'True positive rate among predicted' },
    { label: 'RECALL', val: metrics.recall.toFixed(3), sub: 'Sensitivity / Detection rate' },
    { label: 'LOG LOSS', val: metrics.log_loss.toFixed(3), sub: 'Cross-entropy penalty' }
  ];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      {/* Metric Cards Grid */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
        gap: '16px'
      }}>
        {metricCards.map((m, i) => (
          <div key={i} className="vintage-card" style={{ padding: '20px' }}>
            <span style={{
              fontSize: '0.78rem',
              fontWeight: 700,
              color: 'var(--color-coffee-brown)',
              letterSpacing: '0.5px'
            }}>
              {m.label}
            </span>
            <div style={{
              fontSize: '1.8rem',
              fontWeight: 800,
              fontFamily: 'var(--font-heading)',
              color: 'var(--color-dark-chocolate)',
              margin: '8px 0 4px 0'
            }}>
              {m.val}
            </div>
            <span style={{ fontSize: '0.75rem', color: 'var(--color-coffee-brown)' }}>
              {m.sub}
            </span>
          </div>
        ))}
      </div>

      {/* Confusion Matrix & Breakdown */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(360px, 1fr))',
        gap: '24px'
      }}>
        {/* Matrix Card */}
        <div className="vintage-card">
          <h3 style={{ fontSize: '1.1rem', marginBottom: '16px', color: 'var(--color-dark-chocolate)' }}>
            CONFUSION MATRIX (DECISION THRESHOLD: 0.50)
          </h3>
          <div style={{
            display: 'grid',
            gridTemplateColumns: '1fr 1fr',
            gap: '12px'
          }}>
            <div style={{
              background: 'var(--color-warm-beige)',
              border: '1px solid var(--color-coffee-brown)',
              padding: '16px',
              borderRadius: '8px',
              textAlign: 'center'
            }}>
              <span style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--color-dark-chocolate)' }}>
                TRUE NEGATIVE (TN)
              </span>
              <div style={{ fontSize: '1.5rem', fontWeight: 800, fontFamily: 'var(--font-heading)', color: 'var(--color-dark-chocolate)', marginTop: '4px' }}>
                {cm.tn.toLocaleString()}
              </div>
            </div>

            <div style={{
              background: 'rgba(107, 44, 44, 0.1)',
              border: '1px solid var(--color-burgundy)',
              padding: '16px',
              borderRadius: '8px',
              textAlign: 'center'
            }}>
              <span style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--color-burgundy)' }}>
                FALSE POSITIVE (FP)
              </span>
              <div style={{ fontSize: '1.5rem', fontWeight: 800, fontFamily: 'var(--font-heading)', color: 'var(--color-burgundy)', marginTop: '4px' }}>
                {cm.fp.toLocaleString()}
              </div>
            </div>

            <div style={{
              background: 'rgba(107, 44, 44, 0.1)',
              border: '1px solid var(--color-burgundy)',
              padding: '16px',
              borderRadius: '8px',
              textAlign: 'center'
            }}>
              <span style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--color-burgundy)' }}>
                FALSE NEGATIVE (FN)
              </span>
              <div style={{ fontSize: '1.5rem', fontWeight: 800, fontFamily: 'var(--font-heading)', color: 'var(--color-burgundy)', marginTop: '4px' }}>
                {cm.fn.toLocaleString()}
              </div>
            </div>

            <div style={{
              background: 'var(--color-warm-beige)',
              border: '1px solid var(--color-coffee-brown)',
              padding: '16px',
              borderRadius: '8px',
              textAlign: 'center'
            }}>
              <span style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--color-dark-chocolate)' }}>
                TRUE POSITIVE (TP)
              </span>
              <div style={{ fontSize: '1.5rem', fontWeight: 800, fontFamily: 'var(--font-heading)', color: 'var(--color-dark-chocolate)', marginTop: '4px' }}>
                {cm.tp.toLocaleString()}
              </div>
            </div>
          </div>
        </div>

        {/* Diagnostic Card */}
        <div className="vintage-card">
          <h3 style={{ fontSize: '1.1rem', marginBottom: '16px', color: 'var(--color-dark-chocolate)' }}>
            CLASSIFICATION DIAGNOSTICS & CALIBRATION
          </h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '14px', fontSize: '0.88rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--color-warm-beige)', paddingBottom: '8px' }}>
              <span style={{ color: 'var(--color-dark-chocolate)', fontWeight: 600 }}>Total Verification Samples:</span>
              <span style={{ fontWeight: 700, color: 'var(--color-coffee-brown)' }}>{(cm.tn + cm.fp + cm.fn + cm.tp).toLocaleString()}</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--color-warm-beige)', paddingBottom: '8px' }}>
              <span style={{ color: 'var(--color-dark-chocolate)', fontWeight: 600 }}>Positive Base Rate:</span>
              <span style={{ fontWeight: 700, color: 'var(--color-coffee-brown)' }}>{(((cm.fn + cm.tp) / (cm.tn + cm.fp + cm.fn + cm.tp)) * 100).toFixed(1)}%</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--color-warm-beige)', paddingBottom: '8px' }}>
              <span style={{ color: 'var(--color-dark-chocolate)', fontWeight: 600 }}>False Discovery Rate (FDR):</span>
              <span style={{ fontWeight: 700, color: 'var(--color-burgundy)' }}>{((cm.fp / (cm.tp + cm.fp)) * 100).toFixed(1)}%</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--color-warm-beige)', paddingBottom: '8px' }}>
              <span style={{ color: 'var(--color-dark-chocolate)', fontWeight: 600 }}>Negative Predictive Value (NPV):</span>
              <span style={{ fontWeight: 700, color: 'var(--color-coffee-brown)' }}>{((cm.tn / (cm.tn + cm.fn)) * 100).toFixed(1)}%</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
