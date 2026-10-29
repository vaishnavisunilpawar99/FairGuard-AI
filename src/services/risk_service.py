"""Unified Multi-Pillar Risk Scoring & Recommendation Engine - FairGuard AI
Implements Section 7.3, FR-10, FR-11, and FR-18 of the SRS.
Combines Fairness (30%), Privacy (20%), Drift (20%), Performance (20%),
and Explainability (10%) into an interpretable risk level and actionable guidance.
"""

from typing import Dict, Any, List, Optional


DEFAULT_WEIGHTS = {
    "fairness": 0.30,
    "privacy": 0.20,
    "drift": 0.20,
    "performance": 0.20,
    "explainability": 0.10,
}


def compute_fairness_risk(fairness_dict: Dict[str, Any]) -> float:
    """Normalize fairness metrics (DPD, EOD, DI) into a risk score [0, 1]."""
    dpd = float(fairness_dict.get("demographic_parity_diff", 0.0))
    eod = float(fairness_dict.get("equalized_odds_diff", 0.0))
    di = float(fairness_dict.get("disparate_impact", 1.0))

    # DPD risk: 0 at <=0.05, 1.0 at >=0.25
    dpd_risk = min(1.0, max(0.0, (dpd - 0.05) / 0.20))
    # EOD risk: 0 at <=0.05, 1.0 at >=0.25
    eod_risk = min(1.0, max(0.0, (eod - 0.05) / 0.20))
    # DI risk: 0 at >=0.80, 1.0 at <=0.50
    di_risk = min(1.0, max(0.0, (0.80 - di) / 0.30)) if di < 0.80 else 0.0

    return round(float(0.4 * dpd_risk + 0.3 * eod_risk + 0.3 * di_risk), 4)


def compute_privacy_risk(privacy_dict: Dict[str, Any]) -> float:
    """Normalize privacy metrics (PII, k-anonymity, MIA) into a risk score [0, 1]."""
    if "privacy_risk_score" in privacy_dict:
        return float(privacy_dict["privacy_risk_score"])
    
    score = 0.0
    # Check k-anonymity
    k_anon = privacy_dict.get("k_anonymity", {})
    if k_anon.get("risk") == "High" or k_anon.get("min_k") == 1:
        score += 0.4
    elif k_anon.get("risk") == "Medium":
        score += 0.2

    # Check MIA
    mia = privacy_dict.get("membership_inference", {})
    if mia.get("risk") == "High":
        score += 0.4
    elif mia.get("risk") == "Medium":
        score += 0.2

    # Check PII
    pii = privacy_dict.get("pii", {})
    if pii.get("pii_risk_level") == "High":
        score += 0.5
    elif pii.get("pii_risk_level") == "Medium":
        score += 0.25

    return round(min(1.0, score), 4)


def compute_drift_risk(drift_dict: Dict[str, Any]) -> float:
    """Normalize data and model drift metrics into a risk score [0, 1]."""
    share_drifted = float(drift_dict.get("share_drifted", 0.0))
    max_psi = float(drift_dict.get("max_psi", 0.0))
    
    # PSI risk: 0 at <0.10, 1.0 at >0.30
    psi_risk = min(1.0, max(0.0, (max_psi - 0.10) / 0.20)) if max_psi > 0.10 else 0.0
    # Share drifted: 0 at 0, 1.0 at 0.50
    share_risk = min(1.0, share_drifted / 0.50)

    model_drift = 0.5 if drift_dict.get("prediction_drift", {}).get("model_drift_detected", False) else 0.0

    return round(min(1.0, 0.4 * psi_risk + 0.3 * share_risk + 0.3 * model_drift), 4)


def compute_performance_risk(metrics_dict: Dict[str, Any]) -> float:
    """Normalize conventional performance (accuracy, F1, AUC) into a risk score [0, 1]."""
    acc = float(metrics_dict.get("accuracy", 0.8))
    f1 = float(metrics_dict.get("f1", 0.6))
    auc = float(metrics_dict.get("auc", 0.8))

    # Error risk (1 - accuracy)
    err_risk = min(1.0, max(0.0, 1.0 - acc))
    # Low F1 penalty
    f1_risk = min(1.0, max(0.0, (0.75 - f1) / 0.75)) if f1 < 0.75 else 0.0
    # Low AUC penalty
    auc_risk = min(1.0, max(0.0, (0.80 - auc) / 0.80)) if auc < 0.80 else 0.0

    return round(float(0.4 * err_risk + 0.3 * f1_risk + 0.3 * auc_risk), 4)


def compute_explainability_risk(xai_dict: Dict[str, Any]) -> float:
    """Evaluate proxy-feature leakage or unexplainability into a risk score [0, 1]."""
    has_proxy = bool(xai_dict.get("proxy_warning_flag", False))
    return 0.70 if has_proxy else 0.10


def calculate_overall_risk(
    fairness: Dict[str, Any],
    privacy: Dict[str, Any],
    drift: Dict[str, Any],
    performance: Dict[str, Any],
    explainability: Dict[str, Any],
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Calculate the unified composite risk score and determine the risk band."""
    w = weights or DEFAULT_WEIGHTS

    f_risk = compute_fairness_risk(fairness)
    p_risk = compute_privacy_risk(privacy)
    d_risk = compute_drift_risk(drift)
    m_risk = compute_performance_risk(performance)
    x_risk = compute_explainability_risk(explainability)

    composite_score = (
        w["fairness"] * f_risk
        + w["privacy"] * p_risk
        + w["drift"] * d_risk
        + w["performance"] * m_risk
        + w["explainability"] * x_risk
    )
    composite_score = round(float(composite_score), 4)

    if composite_score > 0.55:
        risk_level = "High"
        requires_review = True
    elif composite_score >= 0.28:
        risk_level = "Medium"
        requires_review = False
    else:
        risk_level = "Low"
        requires_review = False

    return {
        "overall_risk_score": composite_score,
        "overall_risk_level": risk_level,
        "requires_auditor_signoff": requires_review,
        "submodule_scores": {
            "fairness_risk": f_risk,
            "privacy_risk": p_risk,
            "drift_risk": d_risk,
            "performance_risk": m_risk,
            "explainability_risk": x_risk,
        },
        "weights_applied": w,
    }


def generate_recommendations(
    fairness: Dict[str, Any],
    privacy: Dict[str, Any],
    drift: Dict[str, Any],
    performance: Dict[str, Any],
    explainability: Dict[str, Any]
) -> List[Dict[str, str]]:
    """Generate prioritized actionable recommendations as required by FR-11."""
    recs = []

    # 1. Fairness recommendations
    di = fairness.get("disparate_impact", 1.0)
    dpd = fairness.get("demographic_parity_diff", 0.0)
    eod = fairness.get("equalized_odds_diff", 0.0)

    if di < 0.80:
        recs.append({
            "module": "Fairness",
            "severity": "High",
            "text": f"Disparate Impact is {di:.4f} (<0.80), violating the Four-Fifths Rule. Apply post-processing threshold adjustments (EqualizedOdds) or in-processing fair reductions (ExponentiatedGradient).",
        })
    if dpd > 0.10:
        recs.append({
            "module": "Fairness",
            "severity": "Medium",
            "text": f"Demographic Parity Difference is {dpd:.4f} (>0.10 threshold). The model selects demographic groups at significantly unequal rates.",
        })
    if eod > 0.10:
        recs.append({
            "module": "Fairness",
            "severity": "Medium",
            "text": f"Equalized Odds Difference is {eod:.4f} (>0.10). Error rates (FPR and TPR) diverge between protected groups.",
        })

    # 2. Privacy recommendations
    pii = privacy.get("pii", {})
    if pii.get("flagged_columns"):
        cols = list(pii["flagged_columns"].keys())
        recs.append({
            "module": "Privacy",
            "severity": "High",
            "text": f"Potential PII detected in column(s): {', '.join(cols[:4])}. Mask, pseudonymize, or drop these columns before model deployment.",
        })
    k_anon = privacy.get("k_anonymity", {})
    if k_anon.get("min_k") == 1:
        recs.append({
            "module": "Privacy",
            "severity": "High",
            "text": "Dataset has records with k=1 anonymity (uniquely identifiable). Generalize or bin continuous quasi-identifiers (such as exact age or zipcode).",
        })
    mia = privacy.get("membership_inference", {})
    if mia.get("risk") == "High":
        recs.append({
            "module": "Privacy",
            "severity": "Medium",
            "text": f"High Membership Inference Attack (MIA) risk detected (Train/Test AUC gap {mia.get('auc_gap', 0):.4f}). Increase regularization or pruning to reduce model overfitting.",
        })

    # 3. Drift recommendations
    share_drifted = drift.get("share_drifted", 0.0)
    max_psi = drift.get("max_psi", 0.0)
    if max_psi > 0.25 or share_drifted > 0.25:
        recs.append({
            "module": "Drift",
            "severity": "High",
            "text": f"Significant feature drift detected (Max PSI = {max_psi:.4f}, {share_drifted*100:.1f}% drifted). Inspect production data pipeline and trigger model retraining.",
        })
    elif max_psi >= 0.10:
        recs.append({
            "module": "Drift",
            "severity": "Low",
            "text": f"Moderate feature drift detected (Max PSI = {max_psi:.4f}). Set up automated continuous monitoring alerts for new batch ingestion.",
        })

    # 4. Explainability recommendations
    if explainability.get("proxy_warning_flag", False):
        recs.append({
            "module": "Explainability",
            "severity": "Medium",
            "text": "Potential proxy feature (e.g. marital status or relationship) ranks among the top 10 model drivers. Review this feature with domain and legal experts.",
        })

    # 5. Performance recommendations
    acc = performance.get("accuracy", 1.0)
    f1 = performance.get("f1", 1.0)
    if acc < 0.75 or f1 < 0.50:
        recs.append({
            "module": "Performance",
            "severity": "Medium",
            "text": f"Baseline classification metrics are low (Accuracy: {acc:.2f}, F1: {f1:.2f}). Consider hyperparameter tuning or balancing techniques.",
        })

    if not recs:
        recs.append({
            "module": "General",
            "severity": "Low",
            "text": "All metrics are within acceptable thresholds. Maintain ongoing monitoring on production inference traffic.",
        })

    return recs
