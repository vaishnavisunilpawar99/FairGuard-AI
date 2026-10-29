import React, { useState, useEffect } from 'react';
import Header from './components/Header';
import TrustScorecard from './components/TrustScorecard';
import ModelSelector from './components/ModelSelector';
import TabNavigation from './components/TabNavigation';
import PerformanceTab from './components/tabs/PerformanceTab';
import FairnessTab from './components/tabs/FairnessTab';
import ExplainabilityTab from './components/tabs/ExplainabilityTab';
import DriftTab from './components/tabs/DriftTab';
import PrivacyTab from './components/tabs/PrivacyTab';
import GovernanceTab from './components/tabs/GovernanceTab';
import { api } from './api/client';

export default function App() {
  const [backendOnline, setBackendOnline] = useState(false);
  const [datasets, setDatasets] = useState([
    { id: 'adult', name: 'UCI Adult Income (48.8k Census)' },
    { id: 'german', name: 'German Credit Risk (1,000 Lending)' },
    { id: 'synthetic_biased', name: 'Synthetic Biased Suite (AC-03)' },
    { id: 'synthetic_shifted', name: 'Synthetic Drift Suite (AC-04)' },
    { id: 'synthetic_pii', name: 'Synthetic PII Leak Suite (AC-05)' }
  ]);
  const [selectedDataset, setSelectedDataset] = useState('adult');

  const [models, setModels] = useState([
    { id: 'm_rf_001', name: 'Model A: Random Forest (Baseline)' },
    { id: 'm_hist_001', name: 'Model B: HistGradientBoosting (Challenger)' },
    { id: 'm_input_001', name: 'Model C: Pre-trained Benchmark' }
  ]);
  const [activeModel, setActiveModel] = useState('m_rf_001');

  const [activeTab, setActiveTab] = useState('performance');
  const [isAssessing, setIsAssessing] = useState(false);
  const [assessment, setAssessment] = useState(null);
  const [reviewHistory, setReviewHistory] = useState([]);

  // Check health and initialize
  useEffect(() => {
    async function init() {
      try {
        const health = await api.checkHealth();
        if (health && health.status === 'ok') {
          setBackendOnline(true);
        }
        // Load datasets
        try {
          const ds = await api.getDatasets();
          if (ds && ds.length > 0) setDatasets(ds);
        } catch (e) {}
        // Load models
        try {
          const m = await api.getModels();
          if (m && m.length > 0) setModels(m);
        } catch (e) {}
      } catch (err) {
        console.info('Backend running in standalone/offline mode:', err.message);
        setBackendOnline(false);
      }
      // Load initial assessment
      runAudit(selectedDataset, activeModel);
    }
    init();
  }, []);

  const runAudit = async (datasetId, modelId) => {
    setIsAssessing(true);
    try {
      if (backendOnline) {
        const res = await api.runAssessment(datasetId, modelId);
        setAssessment(res);
      } else {
        // High fidelity local state for realistic representation
        setTimeout(() => {
          setAssessment({
            assessment_id: `asm_${Date.now().toString().slice(-6)}`,
            composite_risk_score: datasetId === 'synthetic_biased' ? 0.76 : (datasetId === 'german' ? 0.38 : 0.22),
            risk_level: datasetId === 'synthetic_biased' ? 'HIGH' : (datasetId === 'german' ? 'MEDIUM' : 'LOW'),
            pillar_scores: {
              fairness: datasetId === 'synthetic_biased' ? 0.82 : 0.25,
              privacy: datasetId === 'synthetic_pii' ? 0.90 : 0.08,
              drift: datasetId === 'synthetic_shifted' ? 0.74 : 0.12,
              performance: 0.18,
              explainability: 0.20
            },
            recommendations: [
              {
                priority: 'P1',
                title: 'Fairness Disparity Mitigation',
                message: datasetId === 'synthetic_biased' 
                  ? 'CRITICAL: Severe disparate impact ratio (0.24) detected. Model deployment must be halted until reweighting.'
                  : 'Subgroup disparate impact ratio is 0.31 on age group. Evaluate threshold adjustment for younger cohort.'
              },
              {
                priority: 'P2',
                title: 'Quasi-Identifier De-identification',
                message: 'Evaluate native country and occupation attributes for k-anonymity compliance under GDPR Art. 22.'
              },
              {
                priority: 'P3',
                title: 'Continuous Monitoring Pipeline',
                message: 'Activate automated daily Kolmogorov-Smirnov drift triggers in FastAPI monitoring service.'
              }
            ]
          });
          setIsAssessing(false);
        }, 300);
        return;
      }
    } catch (err) {
      console.warn('API assessment failed, using local fallback:', err.message);
    } finally {
      setIsAssessing(false);
    }
  };

  const handleDatasetChange = (newDs) => {
    setSelectedDataset(newDs);
    runAudit(newDs, activeModel);
  };

  const handleModelChange = (newModel) => {
    setActiveModel(newModel);
    runAudit(selectedDataset, newModel);
  };

  const handleReviewSubmit = async (reviewData) => {
    try {
      if (backendOnline) {
        await api.submitReview(
          reviewData.assessment_id,
          reviewData.reviewer,
          reviewData.decision,
          reviewData.notes
        );
      }
      setReviewHistory(prev => [
        {
          created_at: new Date().toISOString().replace('T', ' ').slice(0, 19) + ' UTC',
          reviewer: reviewData.reviewer,
          decision: reviewData.decision,
          notes: reviewData.notes
        },
        ...prev
      ]);
    } catch (err) {
      throw err;
    }
  };

  const tabs = [
    { id: 'performance', label: '1. CLASSIFICATION & METRICS', icon: '📊' },
    { id: 'fairness', label: '2. FAIRNESS & DISPARITY', icon: '⚖️' },
    { id: 'explainability', label: '3. EXPLAINABILITY (SHAP)', icon: '🔍' },
    { id: 'drift', label: '4. DATA DRIFT (PSI/KS)', icon: '🌊' },
    { id: 'privacy', label: '5. PRIVACY & PII SCAN', icon: '🔒' },
    { id: 'governance', label: '6. HUMAN GOVERNANCE', icon: '🏛️' }
  ];

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      {/* Top Header */}
      <Header
        datasets={datasets}
        selectedDataset={selectedDataset}
        onSelectDataset={handleDatasetChange}
        backendOnline={backendOnline}
        isAssessing={isAssessing}
        onRunAssessment={() => runAudit(selectedDataset, activeModel)}
      />

      {/* Main Content Area */}
      <main className="app-container" style={{ flex: 1, width: '100%' }}>
        {/* Executive Scorecard Banner */}
        <TrustScorecard assessment={assessment} />

        {/* Model Selector Bar */}
        <ModelSelector
          models={models}
          activeModel={activeModel}
          onSelectModel={handleModelChange}
        />

        {/* Tabs Bar */}
        <TabNavigation
          tabs={tabs}
          activeTab={activeTab}
          onSelectTab={setActiveTab}
        />

        {/* Tab Content Views */}
        <div>
          {activeTab === 'performance' && <PerformanceTab data={assessment} />}
          {activeTab === 'fairness' && <FairnessTab data={assessment} />}
          {activeTab === 'explainability' && <ExplainabilityTab data={assessment} />}
          {activeTab === 'drift' && <DriftTab data={assessment} />}
          {activeTab === 'privacy' && <PrivacyTab data={assessment} />}
          {activeTab === 'governance' && (
            <GovernanceTab
              assessment={assessment}
              onSubmitReview={handleReviewSubmit}
              reviewHistory={reviewHistory}
            />
          )}
        </div>
      </main>

      {/* Footer */}
      <footer style={{
        background: 'var(--color-dark-chocolate)',
        color: 'var(--color-warm-beige)',
        textAlign: 'center',
        padding: '20px',
        fontSize: '0.8rem',
        borderTop: '2px solid var(--color-burgundy)',
        marginTop: '40px'
      }}>
        FairGuard AI Platform • Department of Artificial Intelligence & Data Science (Project 23) • Strict "Vintage Soul" Governance Edition
      </footer>
    </div>
  );
}
