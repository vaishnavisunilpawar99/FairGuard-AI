import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

try:
    from fairlearn.metrics import MetricFrame, selection_rate, false_positive_rate, false_negative_rate, true_positive_rate
except ImportError:
    MetricFrame = None


def _fdr(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return fp / (fp + tp) if (fp + tp) > 0 else 0.0


def _error_rate(y_true, y_pred):
    return float(np.mean(y_true != y_pred))


def audit(y_true: pd.Series, y_pred: np.ndarray, sens_col: pd.Series) -> dict:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    sf = np.array(sens_col)

    if MetricFrame is not None:
        mf = MetricFrame(
            metrics={
                "selection_rate": selection_rate,
                "TPR": true_positive_rate,
                "FPR": false_positive_rate,
                "FNR": false_negative_rate,
                "FDR": _fdr,
                "error_rate": _error_rate,
            },
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sf,
        )
        by_group = mf.by_group.copy()
        counts = pd.Series(sf).value_counts().reindex(by_group.index).fillna(0).astype(int)
        by_group.insert(0, "group_count", counts.values)
        by_group.insert(1, "small_group", counts.values < 100)
        overall = mf.overall.to_dict()
    else:
        # Pure Python / Scikit-learn fallback (exact mathematical equivalent)
        groups = np.unique(sf)
        group_rows = []
        for g in groups:
            mask = (sf == g)
            yt_g = y_true[mask]
            yp_g = y_pred[mask]
            g_count = int(np.sum(mask))
            if g_count == 0:
                continue
            
            sel_rate = float(np.mean(yp_g == 1))
            cm = confusion_matrix(yt_g, yp_g, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0
            fdr = fp / (fp + tp) if (fp + tp) > 0 else 0.0
            err = float(np.mean(yt_g != yp_g))

            group_rows.append({
                "group": g,
                "group_count": g_count,
                "small_group": g_count < 100,
                "selection_rate": sel_rate,
                "TPR": tpr,
                "FPR": fpr,
                "FNR": fnr,
                "FDR": fdr,
                "error_rate": err,
            })
        by_group = pd.DataFrame(group_rows).set_index("group")
        counts = by_group["group_count"]
        cm_all = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm_all.ravel()
        overall = {
            "selection_rate": float(np.mean(y_pred == 1)),
            "TPR": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
            "FPR": fp / (fp + tn) if (fp + tn) > 0 else 0.0,
            "FNR": fn / (tp + fn) if (tp + fn) > 0 else 0.0,
            "FDR": fp / (fp + tp) if (fp + tp) > 0 else 0.0,
            "error_rate": float(np.mean(y_true != y_pred)),
        }

    sr = by_group["selection_rate"]
    max_sr = sr.max()
    min_sr = sr.min()
    di = min_sr / max_sr if max_sr > 0 else 1.0

    tpr = by_group["TPR"]
    fpr = by_group["FPR"]
    opportunity_gap = float(tpr.max() - tpr.min())
    eod = float(opportunity_gap + abs(fpr.max() - fpr.min()))

    er = by_group["error_rate"]
    dpd = float(sr.max() - sr.min())
    dpr = float(min_sr / max_sr) if max_sr > 0 else 1.0

    return {
        "by_group": by_group,
        "overall": overall,
        "demographic_parity_diff": round(dpd, 4),
        "demographic_parity_ratio": round(dpr, 4),
        "equalized_odds_diff": round(eod, 4),
        "equal_opportunity_gap": round(opportunity_gap, 4),
        "group_counts": counts.to_dict(),
        "small_groups": counts[counts < 100].index.astype(str).tolist(),
        "disparate_impact": round(di, 4),
        "di_flag": di < 0.8,
        "error_rate_disparity": round(float(er.max() - er.min()), 4),
    }


def status(metric: str, val: float) -> str:
    thresholds = {
        "demographic_parity_diff": [(0.1, "Pass"), (0.2, "Warning")],
        "equalized_odds_diff": [(0.1, "Pass"), (0.2, "Warning")],
        "disparate_impact": [(0.8, None), (0.7, None)],
        "error_rate_disparity": [(0.1, "Pass"), (0.2, "Warning")],
    }
    if metric == "disparate_impact":
        if val >= 0.8:
            return "Pass"
        elif val >= 0.7:
            return "Warning"
        else:
            return "Fail"
    if metric not in thresholds:
        return "N/A"
    for thresh, label in thresholds[metric]:
        if val <= thresh:
            return label
    return "Fail"


def intersectional(y_true, y_pred, sens_df: pd.DataFrame) -> pd.DataFrame:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    sens_df = sens_df.copy().reset_index(drop=True)
    if "sex" in sens_df.columns and "race" in sens_df.columns:
        combo = sens_df["sex"].astype(str) + " x " + sens_df["race"].astype(str)
    else:
        combo = sens_df.iloc[:, 0].astype(str) + " x " + sens_df.iloc[:, 1].astype(str)

    res = audit(y_true, y_pred, combo)
    return res["by_group"]


def confusion_by_group(y_true, y_pred, sens_col: pd.Series):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    sf = np.array(sens_col)
    groups = pd.Series(sf).value_counts().head(4).index.tolist()

    fig, axes = plt.subplots(1, len(groups), figsize=(3.5 * len(groups), 3))
    if len(groups) == 1:
        axes = [axes]

    for ax, g in zip(axes, groups):
        mask = (sf == g)
        cm = confusion_matrix(y_true[mask], y_pred[mask], labels=[0, 1])
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(f"{g} (n={mask.sum()})", fontsize=9)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["<=50K", ">50K"], fontsize=8)
        ax.set_yticklabels(["<=50K", ">50K"], fontsize=8)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=9)

    plt.tight_layout()
    return fig
