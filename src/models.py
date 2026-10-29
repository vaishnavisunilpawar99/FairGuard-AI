import os
import joblib
# MLflow 3 blocks the legacy file store unless explicitly opted in. This
# project keeps its tracking data under artifacts/mlruns for compatibility.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

try:
    import mlflow
except ImportError:
    mlflow = None

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

try:
    from fairlearn.reductions import ExponentiatedGradient, DemographicParity
except ImportError:
    ExponentiatedGradient, DemographicParity = None, None

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")
MLRUNS_DIR = os.path.join(ARTIFACT_DIR, "mlruns")

CAT_COLS = [
    "workclass", "education", "marital-status", "occupation",
    "relationship", "native-country",
]
NUM_COLS = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]


def _build_preprocessor(X: pd.DataFrame):
    cat_present = [c for c in CAT_COLS if c in X.columns]
    num_present = [c for c in NUM_COLS if c in X.columns]
    transformers = []
    if cat_present:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_present))
    if num_present:
        transformers.append(("num", StandardScaler(), num_present))
    return ColumnTransformer(transformers, remainder="drop")


def _get_feature_names(preprocessor, X):
    names = []
    for name, trans, cols in preprocessor.transformers_:
        if name == "cat":
            names.extend(trans.get_feature_names_out(cols).tolist())
        else:
            names.extend(cols)
    return names


def train_all(X_tr: pd.DataFrame, y_tr: pd.Series, s_tr: pd.DataFrame) -> dict:
    os.makedirs(MLRUNS_DIR, exist_ok=True)
    if mlflow:
        try:
            mlflow.set_tracking_uri(f"file:{MLRUNS_DIR}")
            mlflow.set_experiment("fairguard")
        except Exception as e:
            print(f"MLflow setup notice: {e}")

    prep_A = _build_preprocessor(X_tr)
    X_tr_t = prep_A.fit_transform(X_tr)

    prep_B = _build_preprocessor(X_tr)
    X_tr_tb = prep_B.fit_transform(X_tr)

    sensitive_col = s_tr["sex"].astype(str)

    model_A = HistGradientBoostingClassifier(max_iter=100, random_state=42)
    model_A.fit(X_tr_t, y_tr)

    if ExponentiatedGradient is not None:
        base_lr = LogisticRegression(max_iter=1000, solver="lbfgs", C=0.5)
        model_B = ExponentiatedGradient(base_lr, constraints=DemographicParity(), max_iter=30, nu=0.05)
        model_B.fit(X_tr_tb, y_tr, sensitive_features=sensitive_col)
    else:
        # Fallback fair baseline: balanced LogisticRegression with group weight balancing
        sample_weights = np.ones(len(y_tr))
        for group in sensitive_col.unique():
            idx = (sensitive_col == group)
            sample_weights[idx] = 1.0 / (idx.mean() + 1e-4)
        sample_weights /= sample_weights.mean()
        model_B = LogisticRegression(max_iter=1000, solver="lbfgs", C=0.5, class_weight="balanced")
        model_B.fit(X_tr_tb, y_tr, sample_weight=sample_weights)

    if mlflow:
        try:
            with mlflow.start_run(run_name="Model_A_Unmitigated"):
                mlflow.log_param("model_type", "HistGradientBoosting")
                mlflow.log_param("mitigated", False)

            with mlflow.start_run(run_name="Model_B_FairnessMitigated"):
                mlflow.log_param("model_type", "ExponentiatedGradient_LR" if ExponentiatedGradient else "Balanced_LR")
                mlflow.log_param("mitigated", True)
                mlflow.log_param("constraint", "DemographicParity")
        except Exception as exc:
            print(f"MLflow tracking unavailable; continuing without run logging: {exc}")

    joblib.dump((model_A, prep_A), os.path.join(ARTIFACT_DIR, "model_A.joblib"))
    joblib.dump((model_B, prep_B), os.path.join(ARTIFACT_DIR, "model_B.joblib"))

    return {"A": (model_A, prep_A), "B": (model_B, prep_B)}


def predict(model_tuple, X: pd.DataFrame) -> tuple:
    model, prep = model_tuple
    Xt = prep.transform(X)
    preds = model.predict(Xt)
    try:
        proba = model.predict_proba(Xt)[:, 1]
    except AttributeError:
        try:
            pmf = model._pmf_predict(Xt)
            if hasattr(pmf, "iloc"):
                proba = pmf.iloc[:, 1].values
            else:
                proba = np.array(pmf)[:, 1]
        except Exception:
            proba = preds.astype(float)
    return preds, proba


def metrics(y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    acc = accuracy_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_proba)
    except ValueError:
        auc = 0.5
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return {
        "accuracy": round(float(acc), 4),
        "auc": round(float(auc), 4),
        "f1": round(float(f1), 4),
    }
