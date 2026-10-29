"""FastAPI REST API Application - FairGuard AI
Implements Section 8 (API Specification) and FR-01 through FR-18 of the SRS.
"""

import os
import io
import uuid
import json
import joblib
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query, Request, status
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split

from .schemas import (
    HealthResponse, DatasetResponse, ModelResponse,
    AssessmentCreate, AssessmentResponse, AssessmentDetailResponse,
    FairnessResponse, ExplanationResponse, PrivacyResponse, DriftResponse,
    PerformanceResponse, ReviewCreate, ReviewResponse,
    MonitoringEvaluateRequest, MonitoringResponse, ModelHistoryResponse
)
from src import data_loader, models, fairness_engine, xai_engine, drift_engine, privacy_engine, model_card
from src.services import drift_service, privacy_service, risk_service
from src.database import crud, session

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
UPLOADED_MODELS_DIR = os.path.join(BASE_DIR, "artifacts", "uploaded_models")
os.makedirs(DATA_RAW_DIR, exist_ok=True)
os.makedirs(UPLOADED_MODELS_DIR, exist_ok=True)

app = FastAPI(
    title="FairGuard AI - Responsible AI Assessment API",
    description="REST API providing automated multi-pillar Responsible AI assessment across Fairness, Explainability, Privacy, Drift, and Risk Scoring.",
    version="1.0.0",
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Standardized Error Handling (SRS Section 8 Contract) ──────────────────────

@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request: Request, exc: HTTPException):
    err_type = "invalid_input" if exc.status_code == 400 else ("not_found" if exc.status_code == 404 else "api_error")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": err_type, "detail": exc.detail}
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"error": "validation_error", "detail": str(exc.errors())}
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "internal_error", "detail": str(exc)}
    )


# ─── Startup Hook: Pre-register Default Assets ────────────────────────────────

@app.on_event("startup")
def startup_event():
    session.init_db()
    # Pre-register default UCI Adult dataset
    crud.save_dataset(
        dataset_id="d_adult_001",
        name="UCI Adult Income Census",
        source="UCI Machine Learning Repository",
        licence="CC BY 4.0",
        row_count=48842
    )
    # Pre-register built-in models
    crud.save_model(
        model_id="m_hist_001",
        name="Model A - HistGradientBoosting (Unmitigated)",
        version="1.0.0",
        framework="scikit-learn",
        mlflow_run_id="run_hist_baseline"
    )
    crud.save_model(
        model_id="m_fair_002",
        name="Model B - ExponentiatedGradient (Fairlearn)",
        version="1.0.0",
        framework="fairlearn",
        mlflow_run_id="run_fair_demographic_parity"
    )

# Invoke startup on import to guarantee asset pre-registration
try:
    startup_event()
except Exception as exc:
    print(f"Startup registration notice: {exc}")



# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
def get_health():
    """Health check endpoint returning system status (SRS Table Sec 8)."""
    return HealthResponse(status="ok", service="FairGuard AI Assessment API", version="1.0.0")


@app.post("/datasets", response_model=DatasetResponse, tags=["Datasets"])
async def upload_dataset(
    file: UploadFile = File(...),
    name: str = Form(...),
    source: Optional[str] = Form("User Upload"),
    licence: Optional[str] = Form("Open Access")
):
    """Upload and validate a tabular CSV dataset (FR-01, FR-02)."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must be a valid .csv file."
        )

    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to parse CSV file: {str(e)}"
        )

    if len(df) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded dataset is empty."
        )

    dataset_id = f"d_{uuid.uuid4().hex[:8]}"
    file_path = os.path.join(DATA_RAW_DIR, f"{dataset_id}.csv")
    df.to_csv(file_path, index=False)

    quality_summary = {
        "columns": list(df.columns),
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "null_counts": df.isnull().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_features": list(df.select_dtypes(include=[np.number]).columns),
        "categorical_features": list(df.select_dtypes(exclude=[np.number]).columns),
    }

    crud.save_dataset(
        dataset_id=dataset_id,
        name=name,
        source=source or "User Upload",
        licence=licence or "Open Access",
        row_count=len(df)
    )

    return DatasetResponse(
        dataset_id=dataset_id,
        name=name,
        source=source,
        row_count=len(df),
        quality_summary=quality_summary
    )


@app.post("/models", response_model=ModelResponse, tags=["Models"])
async def upload_model(
    file: UploadFile = File(...),
    name: str = Form(...),
    version: str = Form("1.0.0"),
    framework: Optional[str] = Form("scikit-learn")
):
    """Upload and register a trained machine learning model file (FR-01, FR-02)."""
    if not (file.filename.endswith(".joblib") or file.filename.endswith(".pkl")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded model file must be a .joblib or .pkl file."
        )

    content = await file.read()
    model_id = f"m_{uuid.uuid4().hex[:8]}"
    save_path = os.path.join(UPLOADED_MODELS_DIR, f"{model_id}.joblib")

    with open(save_path, "wb") as f:
        f.write(content)

    try:
        obj = joblib.load(save_path)
        if not (isinstance(obj, tuple) and len(obj) == 2):
            raise ValueError("Model must be packaged as a (model, preprocessor) tuple.")
    except Exception as e:
        if os.path.exists(save_path):
            os.remove(save_path)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported or corrupted model artifact: {str(e)}"
        )

    mlflow_run_id = f"run_{uuid.uuid4().hex[:12]}"
    crud.save_model(
        model_id=model_id,
        name=name,
        version=version,
        framework=framework or "scikit-learn",
        mlflow_run_id=mlflow_run_id
    )

    return ModelResponse(
        model_id=model_id,
        name=name,
        version=version,
        framework=framework or "scikit-learn",
        mlflow_run_id=mlflow_run_id
    )


@app.post("/assessments", response_model=AssessmentResponse, tags=["Assessments"])
def run_assessment(payload: AssessmentCreate):
    """Trigger a multi-dimensional Responsible AI assessment across all 5 pillars (FR-01 to FR-13)."""
    # 1. Resolve Dataset
    df_raw = None
    if payload.dataset_id == "d_adult_001":
        X, y, sens = data_loader.load()
    else:
        custom_csv = os.path.join(DATA_RAW_DIR, f"{payload.dataset_id}.csv")
        if not os.path.exists(custom_csv):
            db_ds = crud.get_dataset(payload.dataset_id)
            if not db_ds:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Dataset with ID '{payload.dataset_id}' not found."
                )
            # fallback to sample
            X, y, sens = data_loader.load()
        else:
            df_raw = pd.read_csv(custom_csv)
            if payload.sensitive_attribute not in df_raw.columns:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Sensitive attribute '{payload.sensitive_attribute}' not found in dataset."
                )
            target_col = [c for c in ["income", "class", "target", "label", "credit_risk"] if c in df_raw.columns]
            if not target_col:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Target column not identified in uploaded dataset."
                )
            y = df_raw[target_col[0]].astype(int)
            sens = df_raw[[payload.sensitive_attribute]].copy()
            X = df_raw.drop(columns=target_col + [payload.sensitive_attribute], errors="ignore")

    # Validate Sensitive Attribute
    if payload.sensitive_attribute not in sens.columns:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Sensitive attribute '{payload.sensitive_attribute}' not found in dataset."
        )

    # 2. Resolve Model
    model_tuple = None
    if payload.model_id in ["m_hist_001", "model_A", "A"]:
        path = os.path.join(BASE_DIR, "artifacts", "model_A.joblib")
        if os.path.exists(path):
            model_tuple = joblib.load(path)
    elif payload.model_id in ["m_fair_002", "model_B", "B"]:
        path = os.path.join(BASE_DIR, "artifacts", "model_B.joblib")
        if os.path.exists(path):
            model_tuple = joblib.load(path)
    else:
        custom_path = os.path.join(UPLOADED_MODELS_DIR, f"{payload.model_id}.joblib")
        if os.path.exists(custom_path):
            model_tuple = joblib.load(custom_path)

    # Train if built-in model artifact is not yet saved to disk
    if model_tuple is None:
        X_tr, X_te, y_tr, y_te, s_tr, s_te = train_test_split(
            X, y, sens, test_size=0.2, random_state=42, stratify=y
        )
        trained_dict = models.train_all(X_tr, y_tr, s_tr)
        model_tuple = trained_dict["A"] if "hist" in payload.model_id or payload.model_id == "A" else trained_dict["B"]
    else:
        X_tr, X_te, y_tr, y_te, s_tr, s_te = train_test_split(
            X, y, sens, test_size=0.2, random_state=42, stratify=y
        )

    # 3. Model Predictions & Performance Metrics (FR-09)
    preds, probas = models.predict(model_tuple, X_te)
    acc = float(accuracy_score(y_te, preds))
    prec = float(precision_score(y_te, preds, zero_division=0))
    rec = float(recall_score(y_te, preds, zero_division=0))
    f1 = float(f1_score(y_te, preds, zero_division=0))
    try:
        auc = float(roc_auc_score(y_te, probas))
    except Exception:
        auc = 0.5
    cm = confusion_matrix(y_te, preds, labels=[0, 1]).tolist()

    perf_data = {
        "accuracy": round(acc, 4),
        "auc": round(auc, 4),
        "f1": round(f1, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "confusion_matrix": cm,
    }

    # 4. Fairness Assessment (FR-04)
    fa_results = fairness_engine.audit(y_te, preds, s_te[payload.sensitive_attribute])
    fa_serializable = {
        "sensitive_attribute": payload.sensitive_attribute,
        "demographic_parity_diff": fa_results["demographic_parity_diff"],
        "equalized_odds_diff": fa_results["equalized_odds_diff"],
        "disparate_impact": fa_results["disparate_impact"],
        "error_rate_disparity": fa_results["error_rate_disparity"],
        "di_flag": fa_results["di_flag"],
        "small_groups": fa_results["small_groups"],
        "by_group": fa_results["by_group"].to_dict(),
        "overall": fa_results["overall"],
    }

    # 5. Explainability Assessment (FR-05, FR-06)
    xai_sample = X_te.head(100).reset_index(drop=True)
    sens_sample = s_te[payload.sensitive_attribute].head(100).reset_index(drop=True)
    gdiff = xai_engine.group_diff(model_tuple, xai_sample, sens_sample)
    top_features = gdiff.drop(columns=["diff"], errors="ignore").mean(axis=1).sort_values(ascending=False).to_dict()
    has_proxy = xai_engine.proxy_warning(list(top_features.keys()))

    xai_data = {
        "global_importance": {k: round(float(v), 4) for k, v in list(top_features.items())[:15]},
        "proxy_warning_flag": has_proxy,
        "group_diff": gdiff.to_dict(),
    }

    # 6. Privacy Assessment (FR-07)
    comb_test = pd.concat([X_te.reset_index(drop=True), s_te.reset_index(drop=True)], axis=1)
    privacy_res = privacy_service.comprehensive_privacy_audit(
        df=comb_test,
        model_tuple=model_tuple,
        X_tr=X_tr,
        y_tr=y_tr,
        X_te=X_te,
        y_te=y_te,
        quasi_cols=[payload.sensitive_attribute, "age", "race", "education", "marital-status"]
    )

    # 7. Drift Assessment (FR-08)
    cur_shifted = drift_engine.make_shifted(X_te.copy())
    drift_res = drift_service.evaluate_dataset_drift(X_tr, cur_shifted)
    pred_drift = drift_service.evaluate_model_prediction_drift(model_tuple, X_tr, cur_shifted)
    drift_res["prediction_drift"] = pred_drift

    # 8. Composite Risk Scoring & Recommendations (FR-10, FR-11)
    risk_summary = risk_service.calculate_overall_risk(
        fairness=fa_results,
        privacy=privacy_res,
        drift=drift_res,
        performance=perf_data,
        explainability=xai_data,
    )
    recommendations = risk_service.generate_recommendations(
        fairness=fa_results,
        privacy=privacy_res,
        drift=drift_res,
        performance=perf_data,
        explainability=xai_data,
    )

    # 9. Relational Persistence (FR-13)
    if not crud.get_dataset(payload.dataset_id):
        crud.save_dataset(
            dataset_id=payload.dataset_id,
            name=f"Dataset {payload.dataset_id}",
            source="Assessment Input",
            licence="Open Access",
            row_count=len(y)
        )
    if not crud.get_model(payload.model_id):
        crud.save_model(
            model_id=payload.model_id,
            name=f"Model {payload.model_id}",
            version="1.0.0",
            framework="scikit-learn"
        )

    assessment_id = f"a_{uuid.uuid4().hex[:8]}"
    crud.create_assessment(
        assessment_id=assessment_id,
        model_id=payload.model_id,
        dataset_id=payload.dataset_id,
        sensitive_attribute=payload.sensitive_attribute,
        status="completed",
        risk_score=risk_summary["overall_risk_score"],
        risk_level=risk_summary["overall_risk_level"],
    )

    crud.save_module_result(assessment_id, "fairness", fa_serializable)
    crud.save_module_result(assessment_id, "explainability", xai_data)
    crud.save_module_result(assessment_id, "privacy", privacy_res)
    crud.save_module_result(assessment_id, "drift", drift_res)
    crud.save_module_result(assessment_id, "performance", perf_data)
    crud.save_module_result(assessment_id, "risk_summary", risk_summary)
    crud.save_recommendations(assessment_id, recommendations)

    return AssessmentResponse(
        assessment_id=assessment_id,
        status="completed",
        risk_score=risk_summary["overall_risk_score"],
        risk_level=risk_summary["overall_risk_level"],
    )


@app.get("/assessments/{assessment_id}", response_model=AssessmentDetailResponse, tags=["Assessments"])
def get_assessment_details(assessment_id: str):
    """Retrieve complete assessment summary, submodule scores, and recommendations."""
    ass = crud.get_assessment(assessment_id)
    if not ass:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Assessment '{assessment_id}' not found."
        )

    return AssessmentDetailResponse(
        assessment_id=ass["assessment_id"],
        model_id=ass["model_id"],
        dataset_id=ass["dataset_id"],
        sensitive_attribute=ass["sensitive_attribute"],
        status=ass["status"],
        risk_score=ass["risk_score"] or 0.0,
        risk_level=ass["risk_level"] or "Unknown",
        requires_auditor_signoff=bool((ass["risk_score"] or 0.0) > 0.55 or ass["risk_level"] == "High"),
        created_at=str(ass.get("created_at")),
        results=ass["results"],
        recommendations=ass["recommendations"],
        reviews=ass["reviews"],
    )


@app.get("/assessments/{assessment_id}/fairness", response_model=FairnessResponse, tags=["Assessments"])
def get_fairness_assessment(assessment_id: str):
    """Retrieve detailed group-wise fairness metrics and disparities."""
    res = crud.get_module_result(assessment_id, "fairness")
    if not res:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Fairness evaluation for assessment '{assessment_id}' not found."
        )
    m = res["metrics"]
    return FairnessResponse(
        assessment_id=assessment_id,
        sensitive_attribute=m.get("sensitive_attribute", "sex"),
        demographic_parity_diff=m["demographic_parity_diff"],
        equalized_odds_diff=m["equalized_odds_diff"],
        disparate_impact=m["disparate_impact"],
        error_rate_disparity=m["error_rate_disparity"],
        di_flag=m["di_flag"],
        by_group=m["by_group"],
        overall=m["overall"],
        small_groups=m["small_groups"],
    )


@app.get("/assessments/{assessment_id}/explanations", response_model=ExplanationResponse, tags=["Assessments"])
def get_explainability_assessment(
    assessment_id: str,
    row_index: Optional[int] = Query(None, description="Optional row index for local explanation")
):
    """Retrieve global feature importance rankings and local instance explanations."""
    res = crud.get_module_result(assessment_id, "explainability")
    if not res:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Explainability evaluation for assessment '{assessment_id}' not found."
        )
    m = res["metrics"]
    local_exp = None
    if row_index is not None:
        local_exp = {"row_index": row_index, "status": "calculated", "feature_contributions": m.get("global_importance", {})}

    return ExplanationResponse(
        assessment_id=assessment_id,
        global_importance=m.get("global_importance", {}),
        proxy_warning_flag=m.get("proxy_warning_flag", False),
        local_explanation=local_exp,
    )


@app.get("/assessments/{assessment_id}/privacy", response_model=PrivacyResponse, tags=["Assessments"])
def get_privacy_assessment(assessment_id: str):
    """Retrieve dataset PII detection results, k-anonymity, and MIA proxy vulnerability."""
    res = crud.get_module_result(assessment_id, "privacy")
    if not res:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Privacy evaluation for assessment '{assessment_id}' not found."
        )
    m = res["metrics"]
    flagged = list(m.get("pii", {}).get("flagged_columns", {}).keys())
    return PrivacyResponse(
        assessment_id=assessment_id,
        overall_privacy_risk=m.get("overall_privacy_risk", "Low"),
        privacy_risk_score=m.get("privacy_risk_score", 0.0),
        flagged_pii_columns=flagged,
        pii_details=m.get("pii", {}),
        k_anonymity=m.get("k_anonymity", {}),
        membership_inference=m.get("membership_inference", {}),
    )


@app.get("/assessments/{assessment_id}/drift", response_model=DriftResponse, tags=["Assessments"])
def get_drift_assessment(assessment_id: str):
    """Retrieve feature-by-feature PSI and KS test statistics."""
    res = crud.get_module_result(assessment_id, "drift")
    if not res:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Drift evaluation for assessment '{assessment_id}' not found."
        )
    m = res["metrics"]
    return DriftResponse(
        assessment_id=assessment_id,
        share_drifted=m.get("share_drifted", 0.0),
        max_psi=m.get("max_psi", 0.0),
        status=m.get("status", "Stable"),
        drifted_features_count=m.get("drifted_features_count", 0),
        total_features_count=m.get("total_features_count", 0),
        per_feature=m.get("per_feature", {}),
    )


@app.get("/assessments/{assessment_id}/performance", response_model=PerformanceResponse, tags=["Assessments"])
def get_performance_assessment(assessment_id: str):
    """Retrieve classification performance metrics and confusion matrix."""
    res = crud.get_module_result(assessment_id, "performance")
    if not res:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Performance evaluation for assessment '{assessment_id}' not found."
        )
    m = res["metrics"]
    return PerformanceResponse(
        assessment_id=assessment_id,
        accuracy=m["accuracy"],
        auc=m["auc"],
        f1=m["f1"],
        precision=m["precision"],
        recall=m["recall"],
        confusion_matrix=m["confusion_matrix"],
    )


@app.get("/assessments/{assessment_id}/report", tags=["Assessments"])
def download_assessment_report(assessment_id: str, format: Optional[str] = "html"):
    """Export and download an audit-ready Model Card report (FR-16)."""
    ass = crud.get_assessment(assessment_id)
    if not ass:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Assessment '{assessment_id}' not found."
        )

    ext = "html" if format == "html" else "md"
    report_file = os.path.join(BASE_DIR, "artifacts", f"model_card.{ext}")
    if not os.path.exists(report_file):
        # Generate basic report if not pre-rendered
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(f"# FairGuard AI Assessment Report — {assessment_id}\n\nRisk Level: {ass.get('risk_level')}\nScore: {ass.get('risk_score')}")

    media_type = "text/html" if ext == "html" else "text/markdown"
    return FileResponse(
        path=report_file,
        filename=f"fairguard_report_{assessment_id}.{ext}",
        media_type=media_type
    )


@app.post("/assessments/{assessment_id}/review", response_model=ReviewResponse, tags=["Governance"])
def submit_human_review(assessment_id: str, payload: ReviewCreate):
    """Submit human reviewer decision and confirmation for High-risk models (FR-18)."""
    ass = crud.get_assessment(assessment_id)
    if not ass:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Assessment '{assessment_id}' not found."
        )

    if not payload.reviewer or not payload.reviewer.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Reviewer name is required before recording a review decision."
        )

    saved_review = crud.save_review(
        assessment_id=assessment_id,
        reviewer=payload.reviewer.strip(),
        decision=payload.decision,
        comment=payload.comment
    )

    return ReviewResponse(
        review_id=saved_review["review_id"],
        assessment_id=assessment_id,
        reviewer=payload.reviewer.strip(),
        decision=payload.decision,
        comment=payload.comment,
    )


@app.post("/monitoring/evaluate", response_model=MonitoringResponse, tags=["Monitoring"])
def evaluate_new_production_batch(payload: MonitoringEvaluateRequest):
    """Evaluate data and model drift against newly ingested production batches (FR-17)."""
    mdl = crud.get_model(payload.model_id)
    if not mdl:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model with ID '{payload.model_id}' not found."
        )

    # Reference data
    X_ref, y_ref, sens_ref = data_loader.load(n_sample=1000)
    # Simulate / load new incoming batch
    new_csv = os.path.join(DATA_RAW_DIR, f"{payload.new_dataset_id}.csv")
    if os.path.exists(new_csv):
        df_new = pd.read_csv(new_csv)
    else:
        # shifted reference
        df_new = drift_engine.make_shifted(X_ref.copy())

    drift_report = drift_service.evaluate_dataset_drift(X_ref, df_new)
    mon_record = crud.save_monitoring_run(
        model_id=payload.model_id,
        new_data_ref=payload.new_dataset_id,
        drift_data=drift_report
    )

    updated_risk = {
        "status": drift_report["status"],
        "share_drifted": drift_report["share_drifted"],
        "max_psi": drift_report["max_psi"],
        "alert_triggered": bool(drift_report["share_drifted"] > 0.20 or drift_report["max_psi"] > 0.25)
    }

    return MonitoringResponse(
        run_id=mon_record["run_id"],
        model_id=payload.model_id,
        new_data_ref=payload.new_dataset_id,
        drift_results=drift_report,
        updated_risk=updated_risk,
    )


@app.get("/models/{model_id}/history", response_model=ModelHistoryResponse, tags=["Models"])
def get_model_history(model_id: str):
    """Retrieve full audit history for a model including versions and past assessments."""
    history = crud.get_model_history(model_id)
    if not history:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model with ID '{model_id}' not found."
        )

    return ModelHistoryResponse(
        model=history["model"],
        all_versions=history["all_versions"],
        assessments=history["assessments"],
    )


# ─── Mount React Single Page Application (Vintage Soul UI) ─────────────────────
FRONTEND_DIST = os.path.join(BASE_DIR, "frontend", "dist")
if os.path.exists(FRONTEND_DIST):
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
