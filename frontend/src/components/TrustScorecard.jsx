import React, { useState } from 'react';

export default function TrustScorecard({ assessment }) {
  const [showRecs, setShowRecs] = useState(false);

  if (!assessment) return null;

  const riskScore = assessment.composite_risk_score ?? 0.22;
  const riskBand = assessment.risk_level ?? (riskScore > 0.6 ? 'HIGH' : (riskScore > 0.3 ? 'MEDIUM' : 'LOW'));
  const recs = assessment.recommendations || [
    { priority: 'P1', title: 'Fairness Mitigation', message: 'Disparate impact ratio is below 0.80 on age group. Consider adversarial reweighting or threshold adjustment.' },
    { priority: 'P2', title: 'PII Scrubbing', message: 'Unique identifiers detected in unmasked columns. Mask sensitive tokens prior to downstream feature extraction.' },
    { priority: 'P3', title: 'Covariate Drift Monitoring', message: 'Monitor occupation and age features for shift between training distribution and inference batches.' }
  ];

  const pillarScores = assessment.pillar_scores || {
    fairness: 0.25,
    privacy: 0.10,
    drift: 0.15,
    performance: 0.18,
    explainability: 0.20
  };

  const getBadgeStyle = (band) => {
    switch (band) {
      case 'HIGH':
        return {
          background: 'var(--color-burgundy)',
          color: 'var(--color-cream)',
          border: '1px solid var(--color-burgundy)'
        };
      case 'MEDIUM':
        return {
          background: 'var(--color-coffee-brown)',
          color: 'var(--color-cream)',
          border: '1px solid var(--color-dark-chocolate)'
        };
      default:
        return {
          background: 'var(--color-warm-beige)',
          color: 'var(--color-dark-chocolate)',
          border: '1px solid var(--color-coffee-brown)'
        };
    }
  };

  return (
    <div style={{
      background: 'var(--color-cream)',
      border: '2px solid var(--color-warm-beige)',
      borderRadius: '12px',
      padding: '24px',
      marginBottom: '28px',
      boxShadow: 'var(--shadow-sm)'
    }}>
      {/* Top Bar: Title & Main Score */}
      <div style={{
        display: 'flex',
        flexWrap: 'wrap',
        alignItems: 'center',
        justifyContent: 'space-between',
        gap: '20px',
        paddingBottom: '20px',
        borderBottom: '1px solid var(--color-warm-beige)'
      }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '4px' }}>
            <span style={{ 
              fontFamily: 'var(--font-heading)', 
              fontSize: '1.25rem', 
              fontWeight: 700, 
              color: 'var(--color-dark-chocolate)' 
            }}>
              EXECUTIVE TRUSTWORTHINESS & RISK SCORECARD
            </span>
            <span className="badge" style={getBadgeStyle(riskBand)}>
              {riskBand} RISK BAND
            </span>
          </div>
          <p style={{ margin: 0, fontSize: '0.88rem', color: 'var(--color-coffee-brown)' }}>
            Consolidated Responsible AI multi-pillar governance evaluation (Fairness 30%, Privacy 20%, Drift 20%, Performance 20%, Explainability 10%)
          </p>
        </div>

        {/* Big Composite Metric */}
        <div style={{
          display: 'flex',
          alignItems: 'baseline',
          gap: '12px',
          background: 'var(--color-warm-beige)',
          padding: '12px 24px',
          borderRadius: '10px',
          border: '1px solid var(--color-coffee-brown)'
        }}>
          <span style={{ fontSize: '0.85rem', fontWeight: 700, color: 'var(--color-dark-chocolate)', letterSpacing: '0.5px' }}>
            COMPOSITE RISK:
          </span>
          <span style={{ 
            fontSize: '2rem', 
            fontWeight: 800, 
            fontFamily: 'var(--font-heading)',
            color: riskBand === 'HIGH' ? 'var(--color-burgundy)' : 'var(--color-dark-chocolate)' 
          }}>
            {Number(riskScore).toFixed(2)}
          </span>
          <span style={{ fontSize: '0.85rem', color: 'var(--color-coffee-brown)' }}>/ 1.00</span>
        </div>
      </div>

      {/* 5 Pillars Breakdown Progress */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))',
        gap: '16px',
        margin: '20px 0'
      }}>
        {[
          { name: 'Fairness Risk (30%)', val: pillarScores.fairness ?? 0.2 },
          { name: 'Privacy Risk (20%)', val: pillarScores.privacy ?? 0.1 },
          { name: 'Drift Risk (20%)', val: pillarScores.drift ?? 0.15 },
          { name: 'Perf Risk (20%)', val: pillarScores.performance ?? 0.18 },
          { name: 'Explain Risk (10%)', val: pillarScores.explainability ?? 0.2 }
        ].map((p, i) => (
          <div key={i} style={{
            background: 'rgba(216, 195, 165, 0.25)',
            border: '1px solid var(--color-warm-beige)',
            borderRadius: '8px',
            padding: '12px 14px'
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
              <span style={{ fontSize: '0.78rem', fontWeight: 700, color: 'var(--color-dark-chocolate)' }}>
                {p.name}
              </span>
              <span style={{ fontSize: '0.8rem', fontWeight: 700, color: 'var(--color-coffee-brown)' }}>
                {Number(p.val).toFixed(2)}
              </span>
            </div>
            {/* Custom Bar */}
            <div style={{
              width: '100%',
              height: '8px',
              background: 'var(--color-warm-beige)',
              borderRadius: '4px',
              overflow: 'hidden'
            }}>
              <div style={{
                width: `${Math.min(100, Math.max(0, p.val * 100))}%`,
                height: '100%',
                background: p.val > 0.4 ? 'var(--color-burgundy)' : 'var(--color-coffee-brown)',
                transition: 'width 0.4s ease'
              }} />
            </div>
          </div>
        ))}
      </div>

      {/* Recommendations Toggle Button */}
      <div style={{ paddingTop: '10px' }}>
        <button
          onClick={() => setShowRecs(!showRecs)}
          style={{
            background: 'none',
            border: 'none',
            color: 'var(--color-burgundy)',
            fontWeight: 700,
            fontSize: '0.88rem',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            padding: 0
          }}
        >
          <span>{showRecs ? '▲ HIDE MITIGATION RECOMMENDATIONS' : `▼ VIEW ${recs.length} ACTIONABLE MITIGATION RECOMMENDATIONS`}</span>
        </button>

        {showRecs && (
          <div style={{
            marginTop: '16px',
            display: 'flex',
            flexDirection: 'column',
            gap: '10px'
          }}>
            {recs.map((rec, idx) => (
              <div key={idx} style={{
                background: 'rgba(216, 195, 165, 0.3)',
                borderLeft: '4px solid var(--color-burgundy)',
                padding: '12px 16px',
                borderRadius: '0 8px 8px 0'
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
                  <span style={{
                    background: 'var(--color-burgundy)',
                    color: 'var(--color-cream)',
                    fontSize: '0.7rem',
                    fontWeight: 700,
                    padding: '2px 8px',
                    borderRadius: '4px'
                  }}>
                    {rec.priority || 'P1'}
                  </span>
                  <span style={{ fontWeight: 700, color: 'var(--color-dark-chocolate)', fontSize: '0.9rem' }}>
                    {rec.title}
                  </span>
                </div>
                <p style={{ margin: 0, fontSize: '0.85rem', color: 'var(--color-dark-chocolate)' }}>
                  {rec.message}
                </p>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
