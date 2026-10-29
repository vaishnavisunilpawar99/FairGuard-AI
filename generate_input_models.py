"""
Generates sample input models in the input/ folder for upload testing.
Each file is a (model, preprocessor) tuple saved with joblib.
Run once: venv\Scripts\python.exe generate_input_models.py
"""
import os, sys
os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")
sys.path.insert(0, os.path.dirname(__file__))

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_loader import load

INPUT_DIR = os.path.join(os.path.dirname(__file__), "input")
os.makedirs(INPUT_DIR, exist_ok=True)

CAT_COLS = ["workclass","education","marital-status","occupation",
            "relationship","native-country"]
NUM_COLS = ["age","fnlwgt","education-num","capital-gain","capital-loss","hours-per-week"]

def build_prep(X):
    cat = [c for c in X.columns if X[c].dtype == "object" or X[c].dtype.name == "category"]
    num = [c for c in X.columns if c not in cat]
    t = []
    if cat: t.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat))
    if num: t.append(("num", StandardScaler(), num))
    return ColumnTransformer(t, remainder="drop")

def save_model(name, model, prep, X_tr, y_tr):
    prep.fit(X_tr)
    Xt = prep.transform(X_tr)
    model.fit(Xt, y_tr)
    path = os.path.join(INPUT_DIR, f"{name}.joblib")
    joblib.dump((model, prep), path)
    print(f"  Saved -> {path}")

print("Loading Adult Census data...")
X, y, sens = load(n_sample=20000)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

MODELS = {
    "random_forest":        RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1),
    "logistic_regression":  LogisticRegression(max_iter=500, C=1.0, solver="lbfgs", random_state=42),
    "decision_tree":        DecisionTreeClassifier(max_depth=6, random_state=42),
    "gradient_boosting":    GradientBoostingClassifier(n_estimators=80, max_depth=4, random_state=42),
    "linear_svc_calibrated": CalibratedClassifierCV(LinearSVC(max_iter=2000, random_state=42), cv=3),
}

print(f"Training & saving {len(MODELS)} Adult models to input/ ...")
for name, clf in MODELS.items():
    print(f"  [{name}]")
    prep = build_prep(X_tr)
    save_model(name, clf, prep, X_tr, y_tr)

# Also generate German Credit sample models
try:
    from src.german_credit import load_german_credit
    print("\nLoading German Credit data...")
    X_gc, y_gc, _ = load_german_credit()
    X_gc_tr, _, y_gc_tr, _ = train_test_split(X_gc, y_gc, test_size=0.2, random_state=42, stratify=y_gc)
    
    gc_models = {
        "german_credit_random_forest": RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
        "german_credit_logistic_regression": LogisticRegression(max_iter=500, random_state=42),
    }
    print(f"Training & saving {len(gc_models)} German Credit models to input/ ...")
    for name, clf in gc_models.items():
        print(f"  [{name}]")
        prep = build_prep(X_gc_tr)
        save_model(name, clf, prep, X_gc_tr, y_gc_tr)
except Exception as e:
    print(f"German credit model generation skipped: {e}")

print("\nDone! Files in input/:")
for f in sorted(os.listdir(INPUT_DIR)):
    size = os.path.getsize(os.path.join(INPUT_DIR, f))
    print(f"  {f:45s}  {size/1024:.1f} KB")

