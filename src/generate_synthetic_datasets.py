"""Synthetic Test Dataset Generator - FairGuard AI
Generates controlled benchmark datasets for acceptance testing (AC-03, AC-04, AC-05, T-05, T-07, T-09).
"""

import os
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)


def generate_all_synthetic():
    np.random.seed(42)
    n = 1000

    # 1. Biased Dataset (AC-03 / T-05)
    # Deliberate severe bias: Group A has 85% positive label, Group B has 15% positive label
    group = np.random.choice(["Group_A", "Group_B"], n, p=[0.5, 0.5])
    prob_positive = np.where(group == "Group_A", 0.85, 0.15)
    biased_target = (np.random.uniform(0, 1, n) < prob_positive).astype(int)

    df_biased = pd.DataFrame({
        "feature_1": np.random.normal(50, 10, n),
        "feature_2": np.random.normal(100, 15, n),
        "category": np.random.choice(["Low", "Med", "High"], n),
        "group": group,
        "target": biased_target,
    })
    df_biased.to_csv(os.path.join(DATA_DIR, "synthetic_biased.csv"), index=False)

    # 2. Shifted Dataset (AC-04 / T-07)
    # Mean shifted by +2.5 standard deviations to force PSI > 0.25 and KS p < 0.001
    df_shifted = pd.DataFrame({
        "age": np.random.normal(65, 8, n),  # shifted from normal ~35
        "hours_per_week": np.random.normal(55, 5, n), # shifted from ~40
        "workclass": np.random.choice(["Private", "Government"], n, p=[0.1, 0.9]), # inverted proportion
        "sex": np.random.choice(["Male", "Female"], n),
        "income": np.random.choice([0, 1], n),
    })
    df_shifted.to_csv(os.path.join(DATA_DIR, "synthetic_shifted.csv"), index=False)

    # 3. PII-Injected Dataset (AC-05 / T-09)
    df_pii = pd.DataFrame({
        "user_uuid": [f"usr_{i:05d}" for i in range(n)],
        "full_name": [f"Person {i}" for i in range(n)],
        "email_address": [f"client_{i}@enterprise.org" for i in range(n)],
        "phone_number": [f"+1-555-01{i:02d}" for i in range(n)],
        "social_security_number": [f"987-65-{i:04d}" for i in range(n)],
        "age": np.random.randint(20, 70, n),
        "score": np.random.uniform(0, 100, n),
        "target": np.random.choice([0, 1], n),
    })
    df_pii.to_csv(os.path.join(DATA_DIR, "synthetic_pii.csv"), index=False)

    print("Synthetic datasets generated in data/ folder:")
    print(" - data/synthetic_biased.csv")
    print(" - data/synthetic_shifted.csv")
    print(" - data/synthetic_pii.csv")


if __name__ == "__main__":
    generate_all_synthetic()
