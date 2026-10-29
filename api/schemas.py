"""Pydantic Request & Response Schemas - FairGuard AI REST API
Compliant with SRS Section 8 (API Specification) and FR-14.
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "FairGuard AI Assessment API"
    version: str = "1.0.0"


class DatasetResponse(BaseModel):
    dataset_id: str
    name: str
    source: Optional[str] = None
    row_count: int
    quality_summary: Dict[str, Any]


class ModelResponse(BaseModel):
    model_id: str
    name: str
    version: str
    framework: str
    mlflow_run_id: Optional[str] = None


class AssessmentCreate(BaseModel):
    model_id: str = Field(..., description="ID of the registered model to evaluate")
    dataset_id: str = Field(..., description="ID of the registered dataset to assess against")
    sensitive_attribute: str = Field("sex", description="Target sensitive attribute (e.g. sex, race, age_bin)")
    thresholds: Optional[Dict[str, float]] = Field(
        default_factory=lambda: {"fairness": 0.10, "psi": 0.25, "di": 0.80},
        description="Configurable evaluation thresholds"
    )


class AssessmentResponse(BaseModel):
    assessment_id: str
    status: str
    risk_score: Optional[float] = None
    risk_level: Optional[str] = None


class AssessmentDetailResponse(BaseModel):
    assessment_id: str
    model_id: str
    dataset_id: str
    sensitive_attribute: str
    status: str
    risk_score: float
    risk_level: str
    requires_auditor_signoff: bool
    created_at: Optional[str] = None
    results: Dict[str, Any]
    recommendations: List[Dict[str, Any]]
    reviews: List[Dict[str, Any]]


class FairnessResponse(BaseModel):
    assessment_id: str
    sensitive_attribute: str
    demographic_parity_diff: float
    equalized_odds_diff: float
    disparate_impact: float
    error_rate_disparity: float
    di_flag: bool
    by_group: Dict[str, Any]
    overall: Dict[str, Any]
    small_groups: List[str]


class ExplanationResponse(BaseModel):
    assessment_id: str
    global_importance: Dict[str, float]
    proxy_warning_flag: bool
    local_explanation: Optional[Dict[str, Any]] = None


class PrivacyResponse(BaseModel):
    assessment_id: str
    overall_privacy_risk: str
    privacy_risk_score: float
    flagged_pii_columns: List[str]
    pii_details: Dict[str, Any]
    k_anonymity: Dict[str, Any]
    membership_inference: Dict[str, Any]


class DriftResponse(BaseModel):
    assessment_id: str
    share_drifted: float
    max_psi: float
    status: str
    drifted_features_count: int
    total_features_count: int
    per_feature: Dict[str, Any]


class PerformanceResponse(BaseModel):
    assessment_id: str
    accuracy: float
    auc: float
    f1: float
    precision: float
    recall: float
    confusion_matrix: List[List[int]]


class ReviewCreate(BaseModel):
    reviewer: str = Field(..., description="Name or ID of human reviewer/auditor")
    decision: str = Field(..., description="Approved or Rejected")
    comment: Optional[str] = Field("", description="Review rationale or notes")


class ReviewResponse(BaseModel):
    review_id: str
    assessment_id: str
    reviewer: str
    decision: str
    comment: Optional[str] = None
    reviewed_at: Optional[str] = None


class MonitoringEvaluateRequest(BaseModel):
    model_id: str
    new_dataset_id: str


class MonitoringResponse(BaseModel):
    run_id: str
    model_id: str
    new_data_ref: str
    drift_results: Dict[str, Any]
    updated_risk: Dict[str, Any]


class ModelHistoryResponse(BaseModel):
    model: Dict[str, Any]
    all_versions: List[Dict[str, Any]]
    assessments: List[Dict[str, Any]]
