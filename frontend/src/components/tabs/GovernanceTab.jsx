import React, { useState } from 'react';

export default function GovernanceTab({ assessment, onSubmitReview, reviewHistory = [] }) {
  const [reviewer, setReviewer] = useState('Dr. S. Kulkarni (Ethics Lead)');
  const [decision, setDecision] = useState('APPROVED');
  const [notes, setNotes] = useState('Model passes disparate impact thresholds across primary cohorts. Disparate impact ratio meets acceptable bounds with minor age monitoring needed.');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [message, setMessage] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!reviewer.trim()) return;
    setIsSubmitting(true);
    setMessage('');
    try {
      await onSubmitReview({
        assessment_id: assessment?.assessment_id || 'asm_sample_001',
        reviewer,
        decision,
        notes
      });
      setMessage('✅ Human review decision successfully persisted to database audit trail.');
    } catch (err) {
      setMessage(`❌ Failed to record review: ${err.message}`);
    } finally {
      setIsSubmitting(false);
    }
  };

  const getDecisionBadge = (dec) => {
    switch (dec) {
      case 'APPROVED':
        return { background: 'var(--color-dark-chocolate)', color: 'var(--color-cream)' };
      case 'REJECTED':
        return { background: 'var(--color-burgundy)', color: 'var(--color-cream)' };
      case 'FLAGGED':
        return { background: 'var(--color-coffee-brown)', color: 'var(--color-cream)' };
      default:
        return { background: 'var(--color-warm-beige)', color: 'var(--color-dark-chocolate)' };
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      {/* Top Grid: Review Form & Report Download */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(360px, 1fr))',
        gap: '24px'
      }}>
        {/* Human Sign-off Form */}
        <div className="vintage-card">
          <h3 style={{ fontSize: '1.15rem', marginBottom: '6px', color: 'var(--color-dark-chocolate)' }}>
            HUMAN-IN-THE-LOOP COMPLIANCE SIGN-OFF
          </h3>
          <p style={{ fontSize: '0.82rem', color: 'var(--color-coffee-brown)', marginBottom: '16px' }}>
            Mandatory governance verification. All high-risk classifications trigger formal compliance escalation.
          </p>

          <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
            <div>
              <label style={{ fontSize: '0.8rem', fontWeight: 700, color: 'var(--color-dark-chocolate)', display: 'block', marginBottom: '4px' }}>
                AUDITOR / REVIEWER NAME:
              </label>
              <input
                type="text"
                value={reviewer}
                onChange={(e) => setReviewer(e.target.value)}
                className="input-text"
                required
              />
            </div>

            <div>
              <label style={{ fontSize: '0.8rem', fontWeight: 700, color: 'var(--color-dark-chocolate)', display: 'block', marginBottom: '4px' }}>
                GOVERNANCE VERDICT:
              </label>
              <select
                value={decision}
                onChange={(e) => setDecision(e.target.value)}
                className="input-select"
                style={{ width: '100%' }}
              >
                <option value="APPROVED">APPROVED (Authorized for Production)</option>
                <option value="FLAGGED">FLAGGED (Conditional with Mitigation Required)</option>
                <option value="REJECTED">REJECTED (Risk Exceeds Regulatory Thresholds)</option>
              </select>
            </div>

            <div>
              <label style={{ fontSize: '0.8rem', fontWeight: 700, color: 'var(--color-dark-chocolate)', display: 'block', marginBottom: '4px' }}>
                ETHICS & RISK JUSTIFICATION NOTES:
              </label>
              <textarea
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
                rows={3}
                className="input-text"
                style={{ resize: 'vertical' }}
              />
            </div>

            <button
              type="submit"
              disabled={isSubmitting}
              className="btn-primary"
              style={{ justifyContent: 'center', marginTop: '6px' }}
            >
              {isSubmitting ? 'RECORDING DECISION...' : 'COMMIT AUDIT SIGN-OFF'}
            </button>

            {message && (
              <div style={{
                padding: '10px 14px',
                borderRadius: '6px',
                fontSize: '0.85rem',
                fontWeight: 600,
                background: 'rgba(216, 195, 165, 0.3)',
                color: 'var(--color-dark-chocolate)',
                border: '1px solid var(--color-warm-beige)'
              }}>
                {message}
              </div>
            )}
          </form>
        </div>

        {/* Audit Report Generation & Export */}
        <div className="vintage-card">
          <h3 style={{ fontSize: '1.15rem', marginBottom: '6px', color: 'var(--color-dark-chocolate)' }}>
            AUDIT REPORT GENERATION & MODEL CARDS
          </h3>
          <p style={{ fontSize: '0.82rem', color: 'var(--color-coffee-brown)', marginBottom: '16px' }}>
            Generate compliant Model Cards (Mitchell et al.) and downloadable audit certificates.
          </p>

          <div style={{
            background: 'var(--color-warm-beige)',
            padding: '16px',
            borderRadius: '8px',
            border: '1px solid var(--color-coffee-brown)',
            marginBottom: '16px'
          }}>
            <span style={{ fontSize: '0.78rem', fontWeight: 700, color: 'var(--color-dark-chocolate)', display: 'block' }}>
              ASSESSMENT IDENTIFIER
            </span>
            <span style={{ fontFamily: 'monospace', fontWeight: 700, color: 'var(--color-dark-chocolate)', fontSize: '0.95rem' }}>
              {assessment?.assessment_id || 'asm_sample_001'}
            </span>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
            <a
              href={`http://127.0.0.1:8000/report?assessment_id=${assessment?.assessment_id || 'asm_sample_001'}&format=json`}
              target="_blank"
              rel="noreferrer"
              className="btn-secondary"
              style={{ justifyContent: 'center', textDecoration: 'none' }}
            >
              DOWNLOAD COMPLIANCE REPORT (JSON)
            </a>

            <a
              href={`http://127.0.0.1:8000/report?assessment_id=${assessment?.assessment_id || 'asm_sample_001'}&format=html`}
              target="_blank"
              rel="noreferrer"
              className="btn-dark"
              style={{ justifyContent: 'center', textDecoration: 'none' }}
            >
              EXPORT EXECUTIVE AUDIT SUMMARY (HTML)
            </a>
          </div>
        </div>
      </div>

      {/* Historical Audit Trail */}
      <div className="vintage-card" style={{ padding: 0, overflow: 'hidden' }}>
        <div style={{ padding: '20px 24px', borderBottom: '1px solid var(--color-warm-beige)' }}>
          <h3 style={{ fontSize: '1.15rem', color: 'var(--color-dark-chocolate)' }}>
            IMMUTABLE GOVERNANCE AUDIT LOG
          </h3>
          <span style={{ fontSize: '0.82rem', color: 'var(--color-coffee-brown)' }}>
            Chronological records of all human compliance actions and committee reviews
          </span>
        </div>

        <table className="vintage-table">
          <thead>
            <tr>
              <th>TIMESTAMP</th>
              <th>REVIEWER</th>
              <th>VERDICT</th>
              <th>RATIONALE & NOTES</th>
            </tr>
          </thead>
          <tbody>
            {(reviewHistory.length > 0 ? reviewHistory : [
              {
                created_at: '2026-10-28 11:45 UTC',
                reviewer: 'Dr. S. Kulkarni',
                decision: 'APPROVED',
                notes: 'Primary assessment passed 80% rule; accepted for sandbox staging.'
              },
              {
                created_at: '2026-10-27 16:30 UTC',
                reviewer: 'Compliance Officer (VIT-RAI)',
                decision: 'FLAGGED',
                notes: 'Age subgroup disparate impact was 0.31; required drift re-evaluation.'
              }
            ]).map((rev, i) => (
              <tr key={i}>
                <td style={{ fontFamily: 'monospace', fontSize: '0.8rem' }}>{rev.created_at}</td>
                <td style={{ fontWeight: 700, color: 'var(--color-dark-chocolate)' }}>{rev.reviewer}</td>
                <td>
                  <span className="badge" style={getDecisionBadge(rev.decision)}>
                    {rev.decision}
                  </span>
                </td>
                <td style={{ fontSize: '0.85rem' }}>{rev.notes}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
