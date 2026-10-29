import os
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from . import german_credit

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DATA_PATH = os.path.join(DATA_DIR, "adult_income.csv")
SENSITIVE_COLS = ["sex", "race", "age_bin"]
CAT_COLS = [
    "workclass", "education", "marital-status", "occupation",
    "relationship", "native-country", "sex", "race",
]
NUM_COLS = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]


def _bin_age(age: pd.Series) -> pd.Series:
    return pd.cut(age, bins=[0, 30, 50, 100], labels=["<30", "30-50", ">50"])


def load(dataset_name: str = "adult", n_sample: int = 20000, random_state: int = 42):
    """Load dataset by name: 'adult', 'german', 'synthetic_biased', 'synthetic_shifted', or 'synthetic_pii'."""
    if dataset_name.lower() in ["german", "german_credit"]:
        return german_credit.load_german_credit()

    if dataset_name.startswith("synthetic_"):
        file_path = os.path.join(DATA_DIR, f"{dataset_name}.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            target_col = [c for c in ["target", "income", "credit_risk"] if c in df.columns][0]
            sens_col = [c for c in ["group", "sex", "age"] if c in df.columns][0]
            y = df[target_col].astype(int)
            sens = df[[sens_col]].copy()
            X = df.drop(columns=[target_col, sens_col], errors="ignore")
            return X, y, sens

    # Default: UCI Adult Income
    if os.getenv("FAIRGUARD_SAMPLE"):
        n_sample = min(n_sample, 1000)
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
    else:
        raw = fetch_openml("adult", version=2, as_frame=True)
        df = raw.frame.copy()
        df["income"] = (raw.target.astype(str).str.strip().str.replace(".", "", regex=False)
                        .isin([">=50K", ">50K"])).astype(int)
        df.dropna(inplace=True)
        df.to_csv(DATA_PATH, index=False)

    df.dropna(inplace=True)
    df = df.sample(min(n_sample, len(df)), random_state=random_state).reset_index(drop=True)

    if "income" not in df.columns:
        if "class" in df.columns:
            df["income"] = df["class"].astype(str).str.strip().str.replace(".", "", regex=False).isin([">=50K", ">50K"]).astype(int)
        else:
            raise ValueError("No income/class column found")

    df["age_bin"] = _bin_age(df["age"].astype(float))
    df["sex"] = df["sex"].astype(str).str.strip()
    df["race"] = df["race"].astype(str).str.strip()

    y = df["income"].astype(int)
    sens = df[SENSITIVE_COLS].copy()

    # Sensitive columns remain available in sens for auditing only.
    drop_cols = ["income", "age_bin", "sex", "race"]
    if "class" in df.columns:
        drop_cols.append("class")
    X = df.drop(columns=drop_cols, errors="ignore")

    return X, y, sens


def profile(X: pd.DataFrame, y: pd.Series, sens: pd.DataFrame) -> dict:
    out = {}
    out["class_balance"] = y.value_counts(normalize=True).to_dict()
    out["n_rows"] = len(X)
    out["n_features"] = X.shape[1]
    for col in sens.columns:
        grp = pd.DataFrame({"y": y, "s": sens[col]})
        out[f"group_counts_{col}"] = grp["s"].value_counts().to_dict()
        out[f"group_base_rates_{col}"] = grp.groupby("s", observed=True)["y"].mean().to_dict()
    return out
