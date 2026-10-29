const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

async function fetchJSON(endpoint, options = {}) {
  const url = `${API_BASE_URL}${endpoint}`;
  try {
    const res = await fetch(url, {
      headers: {
        'Content-Type': 'application/json',
        ...(options.headers || {})
      },
      ...options
    });
    if (!res.ok) {
      let errorDetail = `HTTP Error ${res.status}`;
      try {
        const errJson = await res.json();
        errorDetail = errJson.detail || errJson.error || errorDetail;
      } catch (e) {}
      throw new Error(errorDetail);
    }
    return await res.json();
  } catch (err) {
    console.warn(`API call failed to ${endpoint}:`, err.message);
    throw err;
  }
}

export const api = {
  checkHealth: () => fetchJSON('/health'),
  getDatasets: () => fetchJSON('/datasets'),
  getModels: () => fetchJSON('/models'),
  getAssessment: (id) => fetchJSON(`/assessments/${id}`),
  runAssessment: (datasetId, modelId) => fetchJSON('/assessments', {
    method: 'POST',
    body: JSON.stringify({ dataset_id: datasetId, model_id: modelId })
  }),
  getFairness: (datasetId, modelId) => fetchJSON('/fairness', {
    method: 'POST',
    body: JSON.stringify({ dataset_id: datasetId, model_id: modelId })
  }),
  getExplanations: (datasetId, modelId) => fetchJSON('/explanations', {
    method: 'POST',
    body: JSON.stringify({ dataset_id: datasetId, model_id: modelId })
  }),
  getPrivacy: (datasetId, modelId) => fetchJSON('/privacy', {
    method: 'POST',
    body: JSON.stringify({ dataset_id: datasetId, model_id: modelId })
  }),
  getDrift: (datasetId, modelId) => fetchJSON('/drift', {
    method: 'POST',
    body: JSON.stringify({ dataset_id: datasetId, model_id: modelId })
  }),
  getPerformance: (datasetId, modelId) => fetchJSON('/performance', {
    method: 'POST',
    body: JSON.stringify({ dataset_id: datasetId, model_id: modelId })
  }),
  submitReview: (assessmentId, reviewer, decision, notes) => fetchJSON('/review', {
    method: 'POST',
    body: JSON.stringify({
      assessment_id: assessmentId,
      reviewer: reviewer,
      decision: decision,
      notes: notes
    })
  }),
  getModelHistory: (modelId) => fetchJSON(`/models/${modelId}/history`),
  getReportUrl: (assessmentId, format = 'json') => `${API_BASE_URL}/report?assessment_id=${assessmentId}&format=${format}`
};
