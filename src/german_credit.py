"""German Credit Dataset Loader & Generator - FairGuard AI
Implements Secondary Benchmark Dataset as defined in SRS Section 6.1.
Attributes: 1,000 records, 20 attributes, credit risk binary classification (1: Good, 0: Bad).
Sensitive Attributes: age_group, personal_status_sex, foreign_worker.
"""

import os
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
GERMAN_CSV_PATH = os.path.join(DATA_DIR, "german_credit.csv")


def generate_german_credit_data(n_samples: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """Generate realistic synthetic German Credit dataset matching UCI structure."""
    np.random.seed(random_state)

    checking_status = np.random.choice(["<0 DM", "0<=X<200 DM", ">=200 DM", "no checking"], n_samples, p=[0.27, 0.27, 0.06, 0.40])
    duration = np.random.randint(6, 72, n_samples)
    credit_history = np.random.choice(["critical", "delayed", "existing paid", "all paid", "no credit"], n_samples, p=[0.29, 0.09, 0.53, 0.05, 0.04])
    purpose = np.random.choice(["car", "furniture", "radio/tv", "appliances", "repairs", "education", "business"], n_samples)
    credit_amount = np.random.lognormal(mean=7.8, sigma=0.7, size=n_samples).astype(int).clip(250, 18500)
    savings = np.random.choice(["<100 DM", "100<=X<500 DM", "500<=X<1000 DM", ">=1000 DM", "unknown"], n_samples, p=[0.60, 0.10, 0.06, 0.05, 0.19])
    employment = np.random.choice(["unemployed", "<1 yr", "1<=X<4 yrs", "4<=X<7 yrs", ">=7 yrs"], n_samples, p=[0.06, 0.17, 0.34, 0.17, 0.26])
    installment_rate = np.random.choice([1, 2, 3, 4], n_samples, p=[0.14, 0.23, 0.16, 0.47])
    
    # Sensitive attribute: personal status and sex
    personal_status_sex = np.random.choice(["male:single", "male:married/widowed", "female:single", "female:divorced"], n_samples, p=[0.55, 0.14, 0.26, 0.05])
    sex = np.where(pd.Series(personal_status_sex).str.startswith("female"), "Female", "Male")
    
    # Age & sensitive age grouping
    age = np.random.randint(19, 75, n_samples)
    age_group = np.where(age < 25, "Young (<25)", "Adult (>=25)")
    
    housing = np.random.choice(["rent", "own", "for free"], n_samples, p=[0.18, 0.71, 0.11])
    existing_credits = np.random.choice([1, 2, 3, 4], n_samples, p=[0.63, 0.33, 0.03, 0.01])
    job = np.random.choice(["unemployed", "unskilled", "skilled", "management"], n_samples, p=[0.02, 0.20, 0.63, 0.15])
    foreign_worker = np.random.choice(["yes", "no"], n_samples, p=[0.96, 0.04])

    # Ground truth credit risk (1: Good risk, 0: Bad risk)
    # Influenced by history, savings, duration, and slight historical bias on age
    risk_score = (
        (pd.Series(savings).isin([">=1000 DM", "500<=X<1000 DM"])).astype(int) * 0.8
        + (pd.Series(checking_status).isin([">=200 DM", "no checking"])).astype(int) * 0.7
        - (duration > 36).astype(int) * 0.6
        - (credit_amount > 8000).astype(int) * 0.5
        + (age >= 25).astype(int) * 0.3
        + np.random.normal(0, 0.5, n_samples)
    )
    credit_risk = (risk_score > 0.0).astype(int)

    df = pd.DataFrame({
        "checking_status": checking_status,
        "duration": duration,
        "credit_history": credit_history,
        "purpose": purpose,
        "credit_amount": credit_amount,
        "savings": savings,
        "employment": employment,
        "installment_rate": installment_rate,
        "personal_status_sex": personal_status_sex,
        "sex": sex,
        "age": age,
        "age_group": age_group,
        "housing": housing,
        "existing_credits": existing_credits,
        "job": job,
        "foreign_worker": foreign_worker,
        "credit_risk": credit_risk,
    })
    return df


def load_german_credit():
    """Load or generate the German Credit dataset and split into X, y, sens."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(GERMAN_CSV_PATH):
        df = pd.read_csv(GERMAN_CSV_PATH)
    else:
        df = generate_german_credit_data()
        df.to_csv(GERMAN_CSV_PATH, index=False)

    y = df["credit_risk"].astype(int)
    sens = df[["sex", "age_group", "foreign_worker"]].copy()
    X = df.drop(columns=["credit_risk", "sex", "age_group", "foreign_worker"], errors="ignore")
    return X, y, sens
