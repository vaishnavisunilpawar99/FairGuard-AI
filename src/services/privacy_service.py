"""Privacy & PII Assessment Service - FairGuard AI
Implements Rule-Based PII Detection, Quasi-Identifier Analysis,
k-Anonymity Screening, and Membership Inference Attack (MIA) Proxy.
Compliant with SRS Section 7.3 & FR-07.
"""

import re
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from sklearn.metrics import roc_auc_score

# Regex patterns for value-based PII detection
PII_PATTERNS = {
    "email": re.compile(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"),
    "phone": re.compile(r"^(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}$"),
    "ssn": re.compile(r"^\d{3}-\d{2}-\d{4}$"),
    "credit_card": re.compile(r"^\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}$"),
    "ip_address": re.compile(r"^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$"),
    "date_of_birth": re.compile(r"^(?:\d{4}[-/]\d{2}[-/]\d{2}|\d{2}[-/]\d{2}[-/]\d{4})$"),
}

# Header heuristics for column name matching
PII_HEADER_KEYWORDS = {
    "email": ["email", "e-mail", "mail_id", "email_address"],
    "phone": ["phone", "cell", "telephone", "mobile", "contact_no", "contact_number"],
    "ssn_national_id": ["ssn", "social_security", "national_id", "aadhar", "passport", "tax_id"],
    "name": ["name", "full_name", "first_name", "last_name", "surname", "fname", "lname"],
    "dob": ["dob", "birth_date", "birthdate", "date_of_birth"],
    "address": ["address", "street", "residence", "city", "zipcode", "postal_code", "postcode"],
    "financial": ["credit_card", "card_number", "bank_account", "account_no", "iban", "cvv"],
    "ip_address": ["ip_address", "client_ip", "ip", "ipv4"],
}

DEFAULT_QUASI_IDENTIFIERS = [
    "age", "sex", "gender", "race", "ethnicity", "education",
    "marital-status", "zipcode", "occupation", "native-country"
]


def detect_pii_columns(df: pd.DataFrame, sample_size: int = 500) -> Dict[str, Any]:
    """Scan dataframe headers and cell values for Personally Identifiable Information (PII)."""
    flagged_columns = {}
    sample_df = df.sample(min(len(df), sample_size), random_state=42) if len(df) > 0 else df

    for col in df.columns:
        col_lower = str(col).lower().replace(" ", "_").replace("-", "_")
        matched_categories = []

        # 1. Header heuristic check
        for category, keywords in PII_HEADER_KEYWORDS.items():
            for kw in keywords:
                if kw == col_lower or (f"_{kw}" in col_lower) or (f"{kw}_" in col_lower):
                    matched_categories.append(f"Header keyword matched: '{kw}' ({category})")
                    break

        # 2. Content regex matching on string / object values
        sample_vals = sample_df[col].dropna().astype(str).str.strip()
        if len(sample_vals) > 0:
            for pattern_name, regex in PII_PATTERNS.items():
                match_count = sum(bool(regex.match(val)) for val in sample_vals.head(100))
                if match_count >= max(2, int(len(sample_vals.head(100)) * 0.10)):
                    matched_categories.append(f"Content format matched: {pattern_name} ({match_count}% matches)")

        # 3. Check for unique identifier columns (e.g. ID, UUID)
        n_unique = df[col].nunique()
        total_rows = len(df)
        if total_rows > 50 and n_unique == total_rows and ("id" in col_lower or "code" in col_lower):
            matched_categories.append("Unique identifier: 100% unique row values with ID-like name")

        if matched_categories:
            flagged_columns[col] = {
                "column": col,
                "findings": matched_categories,
                "severity": "High" if any("email" in m or "ssn" in m or "phone" in m or "financial" in m for m in matched_categories) else "Medium",
            }

    # Overall PII Risk
    high_count = sum(1 for v in flagged_columns.values() if v["severity"] == "High")
    med_count = sum(1 for v in flagged_columns.values() if v["severity"] == "Medium")

    if high_count > 0:
        pii_risk_level = "High"
        pii_risk_score = min(1.0, 0.6 + 0.1 * high_count)
    elif med_count > 0:
        pii_risk_level = "Medium"
        pii_risk_score = min(0.6, 0.3 + 0.05 * med_count)
    else:
        pii_risk_level = "Low"
        pii_risk_score = 0.0

    return {
        "pii_risk_level": pii_risk_level,
        "pii_risk_score": round(pii_risk_score, 4),
        "flagged_columns_count": len(flagged_columns),
        "flagged_columns": flagged_columns,
    }


def calculate_k_anonymity(df: pd.DataFrame, quasi_cols: Optional[List[str]] = None) -> Dict[str, Any]:
    """Calculate k-anonymity score, proportion of rows with k < 5, and uniqueness percentage."""
    if quasi_cols is None:
        quasi_cols = DEFAULT_QUASI_IDENTIFIERS

    available_cols = [c for c in quasi_cols if c in df.columns]
    if not available_cols:
        return {
            "min_k": -1,
            "pct_rows_k_lt_5": 0.0,
            "unique_pct": 0.0,
            "quasi_identifiers_evaluated": [],
            "risk": "Low",
        }

    sub = pd.DataFrame({c: df[c].astype(str).values for c in available_cols})
    grp = sub.groupby(available_cols).size().reset_index(name="count")
    merged = sub.merge(grp, on=available_cols, how="left")
    counts = merged["count"].values

    min_k = int(counts.min()) if len(counts) > 0 else 0
    pct_lt5 = round(float((counts < 5).mean()) * 100, 2)
    unique_pct = round(float((counts == 1).mean()) * 100, 2)

    if unique_pct > 5.0 or min_k == 1:
        risk = "High"
    elif unique_pct >= 1.0 or pct_lt5 > 10.0:
        risk = "Medium"
    else:
        risk = "Low"

    return {
        "min_k": min_k,
        "pct_rows_k_lt_5": pct_lt5,
        "unique_pct": unique_pct,
        "quasi_identifiers_evaluated": available_cols,
        "risk": risk,
    }


def evaluate_mia_risk(
    model_tuple: Any,
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_te: pd.DataFrame,
    y_te: pd.Series
) -> Dict[str, Any]:
    """Estimate Membership Inference Attack (MIA) vulnerability via train/test confidence and AUC gaps."""
    model, prep = model_tuple
    try:
        Xtr_t = prep.transform(X_tr)
        Xte_t = prep.transform(X_te)
    except Exception as e:
        return {"risk": "Unknown", "error": f"Preprocessing failed: {e}"}

    try:
        conf_tr = model.predict_proba(Xtr_t)
        conf_te = model.predict_proba(Xte_t)
        tr_conf = float(conf_tr.max(axis=1).mean())
        te_conf = float(conf_te.max(axis=1).mean())
        tr_auc = float(roc_auc_score(y_tr, conf_tr[:, 1]))
        te_auc = float(roc_auc_score(y_te, conf_te[:, 1]))
    except Exception:
        preds_tr = model.predict(Xtr_t).astype(float)
        preds_te = model.predict(Xte_t).astype(float)
        tr_conf = float(preds_tr.mean())
        te_conf = float(preds_te.mean())
        try:
            tr_auc = float(roc_auc_score(y_tr, preds_tr))
            te_auc = float(roc_auc_score(y_te, preds_te))
        except Exception:
            tr_auc, te_auc = 0.5, 0.5

    auc_gap = round(tr_auc - te_auc, 4)
    conf_gap = round(tr_conf - te_conf, 4)

    if auc_gap < 0.03:
        risk = "Low"
    elif auc_gap < 0.08:
        risk = "Medium"
    else:
        risk = "High"

    return {
        "train_auc": round(tr_auc, 4),
        "test_auc": round(te_auc, 4),
        "auc_gap": auc_gap,
        "train_confidence": round(tr_conf, 4),
        "test_confidence": round(te_conf, 4),
        "confidence_gap": conf_gap,
        "risk": risk,
    }


def comprehensive_privacy_audit(
    df: pd.DataFrame,
    model_tuple: Optional[Any] = None,
    X_tr: Optional[pd.DataFrame] = None,
    y_tr: Optional[pd.Series] = None,
    X_te: Optional[pd.DataFrame] = None,
    y_te: Optional[pd.Series] = None,
    quasi_cols: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Execute complete privacy assessment covering PII detection, k-anonymity, and MIA proxy."""
    pii_results = detect_pii_columns(df)
    k_anon = calculate_k_anonymity(df, quasi_cols)

    mia_results = {}
    if model_tuple and X_tr is not None and y_tr is not None and X_te is not None and y_te is not None:
        mia_results = evaluate_mia_risk(model_tuple, X_tr, y_tr, X_te, y_te)

    # Composite privacy risk score
    k_risk_score = 0.8 if k_anon["risk"] == "High" else (0.4 if k_anon["risk"] == "Medium" else 0.1)
    mia_risk_score = 0.8 if mia_results.get("risk") == "High" else (0.4 if mia_results.get("risk") == "Medium" else 0.1)
    pii_score = pii_results["pii_risk_score"]

    composite_privacy_score = round(0.4 * pii_score + 0.3 * k_risk_score + 0.3 * mia_risk_score, 4)
    
    if composite_privacy_score > 0.5 or pii_results["pii_risk_level"] == "High":
        overall_risk = "High"
    elif composite_privacy_score > 0.25:
        overall_risk = "Medium"
    else:
        overall_risk = "Low"

    return {
        "overall_privacy_risk": overall_risk,
        "privacy_risk_score": composite_privacy_score,
        "pii": pii_results,
        "k_anonymity": k_anon,
        "membership_inference": mia_results,
    }
