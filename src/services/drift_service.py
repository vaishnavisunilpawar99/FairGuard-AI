"""Drift Detection Service - FairGuard AI
Implements Population Stability Index (PSI), Kolmogorov-Smirnov (KS) test,
Chi-square test, and prediction-distribution drift as per SRS Section 7.3 & FR-08.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any, Optional, Tuple


def calculate_psi_numeric(
    expected: np.ndarray,
    actual: np.ndarray,
    num_buckets: int = 10,
    epsilon: float = 1e-4
) -> Tuple[float, str]:
    """Calculate Population Stability Index (PSI) for continuous numeric feature.
    
    Thresholds:
      PSI < 0.10: Stable (No significant shift)
      0.10 <= PSI <= 0.25: Moderate Drift
      PSI > 0.25: Significant Drift
    """
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    
    if len(expected) == 0 or len(actual) == 0:
        return 0.0, "Stable"

    # Determine quantile bucket edges on reference (expected) data
    percentiles = np.linspace(0, 100, num_buckets + 1)
    bucket_edges = np.percentile(expected, percentiles)
    # Deduplicate edges in case of identical values
    bucket_edges = np.unique(bucket_edges)
    
    if len(bucket_edges) < 2:
        return 0.0, "Stable"
        
    bucket_edges[0] = -np.inf
    bucket_edges[-1] = np.inf

    # Bucket counts
    exp_counts, _ = np.histogram(expected, bins=bucket_edges)
    act_counts, _ = np.histogram(actual, bins=bucket_edges)

    # Fractions with smoothing epsilon
    exp_pct = np.maximum(exp_counts / len(expected), epsilon)
    act_pct = np.maximum(act_counts / len(actual), epsilon)

    psi_val = float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))
    psi_val = max(0.0, round(psi_val, 4))

    if psi_val < 0.10:
        category = "Stable"
    elif psi_val <= 0.25:
        category = "Moderate"
    else:
        category = "Significant"

    return psi_val, category


def calculate_psi_categorical(
    expected: pd.Series,
    actual: pd.Series,
    epsilon: float = 1e-4
) -> Tuple[float, str]:
    """Calculate Population Stability Index (PSI) for discrete categorical feature."""
    exp_counts = expected.astype(str).value_counts()
    act_counts = actual.astype(str).value_counts()

    categories = list(set(exp_counts.index).union(set(act_counts.index)))
    if not categories:
        return 0.0, "Stable"

    n_exp = len(expected)
    n_act = len(actual)

    psi_val = 0.0
    for cat in categories:
        p_exp = max(exp_counts.get(cat, 0) / n_exp, epsilon)
        p_act = max(act_counts.get(cat, 0) / n_act, epsilon)
        psi_val += (p_act - p_exp) * np.log(p_act / p_exp)

    psi_val = max(0.0, round(float(psi_val), 4))

    if psi_val < 0.10:
        category = "Stable"
    elif psi_val <= 0.25:
        category = "Moderate"
    else:
        category = "Significant"

    return psi_val, category


def calculate_ks_test(
    expected: np.ndarray,
    actual: np.ndarray
) -> Tuple[float, float, bool]:
    """Perform two-sample Kolmogorov-Smirnov test for numeric distribution shift.
    
    Returns (statistic, p_value, is_drifted).
    Flagged if p_value < 0.05.
    """
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    if len(expected) == 0 or len(actual) == 0:
        return 0.0, 1.0, False

    stat_res = stats.ks_2samp(expected, actual)
    ks_stat = round(float(stat_res.statistic), 4)
    p_val = round(float(stat_res.pvalue), 6)
    drifted = bool(p_val < 0.05)
    return ks_stat, p_val, drifted


def calculate_chi2_test(
    expected: pd.Series,
    actual: pd.Series
) -> Tuple[float, float, bool]:
    """Perform Chi-square test for categorical distribution shift."""
    exp_counts = expected.astype(str).value_counts()
    act_counts = actual.astype(str).value_counts()
    cats = list(set(exp_counts.index).union(set(act_counts.index)))

    if len(cats) < 2:
        return 0.0, 1.0, False

    table = [
        [exp_counts.get(c, 0) for c in cats],
        [act_counts.get(c, 0) for c in cats]
    ]
    try:
        chi2, p_val, _, _ = stats.chi2_contingency(table)
        drifted = bool(p_val < 0.05)
        return round(float(chi2), 4), round(float(p_val), 6), drifted
    except Exception:
        return 0.0, 1.0, False


def evaluate_dataset_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    psi_threshold: float = 0.25
) -> Dict[str, Any]:
    """Perform comprehensive drift analysis comparing reference and current dataframes."""
    per_feature = {}
    drifted_count = 0
    common_cols = [c for c in reference_df.columns if c in current_df.columns]
    
    max_psi = 0.0

    for col in common_cols:
        ref_s = reference_df[col].dropna()
        cur_s = current_df[col].dropna()
        
        is_numeric = pd.api.types.is_numeric_dtype(ref_s) and len(ref_s.unique()) > 5

        if is_numeric:
            ref_arr = ref_s.to_numpy(dtype=float)
            cur_arr = cur_s.to_numpy(dtype=float)
            psi_val, psi_cat = calculate_psi_numeric(ref_arr, cur_arr)
            ks_stat, p_val, ks_drift = calculate_ks_test(ref_arr, cur_arr)
            
            # Drift flagged if PSI is significant (> threshold) OR KS test p < 0.05 with noticeable shift
            is_drifted = bool(psi_val >= psi_threshold or (ks_drift and psi_val >= 0.10))
            
            per_feature[col] = {
                "type": "numeric",
                "stattest": "KS test + PSI",
                "psi": psi_val,
                "psi_category": psi_cat,
                "ks_statistic": ks_stat,
                "p_value": p_val,
                "drift_detected": is_drifted,
            }
        else:
            psi_val, psi_cat = calculate_psi_categorical(ref_s, cur_s)
            chi2_stat, p_val, chi2_drift = calculate_chi2_test(ref_s, cur_s)
            
            is_drifted = bool(psi_val >= psi_threshold or (chi2_drift and psi_val >= 0.10))
            
            per_feature[col] = {
                "type": "categorical",
                "stattest": "Chi-Square + PSI",
                "psi": psi_val,
                "psi_category": psi_cat,
                "chi2_statistic": chi2_stat,
                "p_value": p_val,
                "drift_detected": is_drifted,
            }

        max_psi = max(max_psi, psi_val)
        if is_drifted:
            drifted_count += 1

    total_cols = len(common_cols)
    share_drifted = round(drifted_count / total_cols, 4) if total_cols > 0 else 0.0

    overall_status = "Stable"
    if share_drifted > 0.3 or max_psi > 0.25:
        overall_status = "Significant Drift"
    elif share_drifted > 0.1 or max_psi > 0.10:
        overall_status = "Moderate Drift"

    return {
        "share_drifted": share_drifted,
        "drifted_features_count": drifted_count,
        "total_features_count": total_cols,
        "max_psi": round(max_psi, 4),
        "status": overall_status,
        "per_feature": per_feature,
    }


def evaluate_model_prediction_drift(
    model_tuple: Any,
    reference_X: pd.DataFrame,
    current_X: pd.DataFrame
) -> Dict[str, Any]:
    """Evaluate drift in the model's output prediction probabilities and classifications."""
    model, prep = model_tuple
    try:
        ref_t = prep.transform(reference_X)
        cur_t = prep.transform(current_X)
        
        if hasattr(model, "predict_proba"):
            ref_probs = model.predict_proba(ref_t)[:, 1]
            cur_probs = model.predict_proba(cur_t)[:, 1]
        elif hasattr(model, "_pmf_predict"):
            pmf_ref = model._pmf_predict(ref_t)
            pmf_cur = model._pmf_predict(cur_t)
            ref_probs = np.array(pmf_ref)[:, 1]
            cur_probs = np.array(pmf_cur)[:, 1]
        else:
            ref_probs = model.predict(ref_t).astype(float)
            cur_probs = model.predict(cur_t).astype(float)

        psi_val, psi_cat = calculate_psi_numeric(ref_probs, cur_probs)
        ks_stat, p_val, ks_drift = calculate_ks_test(ref_probs, cur_probs)

        return {
            "prediction_psi": psi_val,
            "prediction_psi_category": psi_cat,
            "prediction_ks_stat": ks_stat,
            "prediction_p_value": p_val,
            "model_drift_detected": bool(psi_val >= 0.25 or (ks_drift and psi_val >= 0.10)),
            "ref_mean_score": round(float(np.mean(ref_probs)), 4),
            "cur_mean_score": round(float(np.mean(cur_probs)), 4),
        }
    except Exception as exc:
        return {
            "prediction_psi": 0.0,
            "prediction_psi_category": "Error",
            "model_drift_detected": False,
            "error": str(exc)
        }
