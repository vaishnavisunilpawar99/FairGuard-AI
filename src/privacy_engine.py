import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from .services.privacy_service import detect_pii_columns, comprehensive_privacy_audit


def k_anonymity(df: pd.DataFrame, quasi: list = None) -> dict:
    if quasi is None:
        quasi = ["age", "sex", "race", "education", "marital-status"]
    # Flatten any duplicate column names and ensure 1-D columns only
    df = df.copy()
    df.columns = [str(c) for c in df.columns]
    # Drop duplicate columns keeping first
    df = df.loc[:, ~df.columns.duplicated()]
    cols = [c for c in quasi if c in df.columns]
    if not cols:
        return {"min_k": -1, "pct_rows_k_lt_5": -1}
    # Ensure all quasi columns are 1-D Series (not DataFrames)
    sub = pd.DataFrame({c: df[c].astype(str).values for c in cols})
    grp = sub.groupby(cols).size().reset_index(name="count")
    merged = sub.merge(grp, on=cols, how="left")
    counts = merged["count"].values
    min_k = int(counts.min()) if len(counts) > 0 else 0
    pct_lt5 = round(float((counts < 5).mean()) * 100, 2)
    unique_pct = round(float((counts == 1).mean()) * 100, 2)
    return {"min_k": min_k, "pct_rows_k_lt_5": pct_lt5,
            "unique_pct": unique_pct, "risk": privacy_risk(unique_pct)}


def privacy_risk(unique_pct: float) -> str:
    if unique_pct < 1:
        return "Low"
    if unique_pct <= 5:
        return "Medium"
    return "High"


def mia_proxy(model_tuple, X_tr: pd.DataFrame, y_tr: pd.Series,
              X_te: pd.DataFrame, y_te: pd.Series) -> dict:
    model, prep = model_tuple
    Xtr_t = prep.transform(X_tr)
    Xte_t = prep.transform(X_te)
    try:
        conf_tr = model.predict_proba(Xtr_t)
        conf_te = model.predict_proba(Xte_t)
        tr_conf = conf_tr.max(axis=1).mean()
        te_conf = conf_te.max(axis=1).mean()
        tr_auc = roc_auc_score(y_tr, conf_tr[:, 1])
        te_auc = roc_auc_score(y_te, conf_te[:, 1])
    except Exception:
        preds_tr = model.predict(Xtr_t).astype(float)
        preds_te = model.predict(Xte_t).astype(float)
        tr_conf = float(preds_tr.mean())
        te_conf = float(preds_te.mean())
        try:
            tr_auc = roc_auc_score(y_tr, preds_tr)
            te_auc = roc_auc_score(y_te, preds_te)
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
