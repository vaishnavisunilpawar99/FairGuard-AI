"""Unit and Integration Tests for Phase 1 & Phase 2 - FairGuard AI
Validates PSI/KS drift engine, PII detection, risk scoring & recommendations,
and relational database persistence (SRS Sections 6.4, 7.3, FR-07, FR-08, FR-10, FR-11, FR-13).
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.services import drift_service, privacy_service, risk_service
from src.database import crud, session


class TestPhase1AndPhase2(unittest.TestCase):

    def setUp(self):
        session.init_db()

    # ─── 1. Drift Service Tests ───────────────────────────────────────────────

    def test_psi_and_ks_unshifted(self):
        """Unshifted data should have low PSI and no KS drift (Test Scenario T-08)."""
        np.random.seed(42)
        ref = np.random.normal(50, 5, 1000)
        cur = np.random.normal(50, 5, 1000)
        psi, cat = drift_service.calculate_psi_numeric(ref, cur)
        stat, pval, drifted = drift_service.calculate_ks_test(ref, cur)

        self.assertLess(psi, 0.10, f"Expected stable PSI < 0.10, got {psi}")
        self.assertEqual(cat, "Stable")
        self.assertFalse(drifted, "Unshifted data should not be flagged as drifted")

    def test_psi_and_ks_shifted(self):
        """Artificially shifted data should trigger significant PSI and KS drift (Test Scenario T-07)."""
        np.random.seed(42)
        ref = np.random.normal(50, 5, 1000)
        cur = np.random.normal(65, 5, 1000)
        psi, cat = drift_service.calculate_psi_numeric(ref, cur)
        stat, pval, drifted = drift_service.calculate_ks_test(ref, cur)

        self.assertGreater(psi, 0.25, f"Expected significant PSI > 0.25, got {psi}")
        self.assertEqual(cat, "Significant")
        self.assertTrue(drifted, "Shifted data must be flagged as drifted")
        self.assertLess(pval, 0.05, "p-value must be < 0.05 for shifted distributions")

    def test_dataset_drift_evaluation(self):
        df_ref = pd.DataFrame({
            "age": np.random.normal(40, 10, 500),
            "hours": np.random.normal(40, 5, 500),
            "workclass": np.random.choice(["Private", "Gov", "Self-emp"], 500)
        })
        df_cur = df_ref.copy()
        df_cur["age"] = df_cur["age"] + 15  # shift age significantly

        report = drift_service.evaluate_dataset_drift(df_ref, df_cur)
        self.assertIn("share_drifted", report)
        self.assertIn("max_psi", report)
        self.assertTrue(report["per_feature"]["age"]["drift_detected"])

    # ─── 2. Privacy & PII Tests ───────────────────────────────────────────────

    def test_clean_dataset_pii(self):
        """Clean tabular dataset should return zero PII flags."""
        clean_df = pd.DataFrame({
            "age": [25, 45, 60],
            "capital_gain": [1000, 0, 5000],
            "hours_per_week": [40, 35, 50]
        })
        res = privacy_service.detect_pii_columns(clean_df)
        self.assertEqual(res["pii_risk_level"], "Low")
        self.assertEqual(res["flagged_columns_count"], 0)

    def test_pii_injection_detection(self):
        """Dataset with emails and phone numbers must be flagged (Test Scenario T-09)."""
        pii_df = pd.DataFrame({
            "client_email": [f"user_{i}@company.com" for i in range(100)],
            "phone_number": ["555-123-4567"] * 100,
            "ssn_number": ["123-45-6789"] * 100,
            "age": [30] * 100
        })
        res = privacy_service.detect_pii_columns(pii_df)
        self.assertEqual(res["pii_risk_level"], "High")
        self.assertIn("client_email", res["flagged_columns"])
        self.assertIn("phone_number", res["flagged_columns"])
        self.assertIn("ssn_number", res["flagged_columns"])

    def test_k_anonymity_screening(self):
        df = pd.DataFrame({
            "age": [20, 20, 20, 30, 40],
            "sex": ["M", "M", "M", "F", "F"]
        })
        res = privacy_service.calculate_k_anonymity(df, ["age", "sex"])
        self.assertEqual(res["min_k"], 1)  # age 30 and 40 appear once
        self.assertEqual(res["risk"], "High")

    # ─── 3. Risk & Recommendation Tests ───────────────────────────────────────

    def test_multi_pillar_risk_and_recommendations(self):
        fair = {"demographic_parity_diff": 0.22, "equalized_odds_diff": 0.18, "disparate_impact": 0.55}
        priv = {"privacy_risk_score": 0.7, "pii": {"flagged_columns": {"email": {}}}}
        drft = {"share_drifted": 0.35, "max_psi": 0.40}
        perf = {"accuracy": 0.78, "f1": 0.60, "auc": 0.75}
        xai = {"proxy_warning_flag": True}

        risk_res = risk_service.calculate_overall_risk(fair, priv, drft, perf, xai)
        self.assertEqual(risk_res["overall_risk_level"], "High")
        self.assertTrue(risk_res["requires_auditor_signoff"])

        recs = risk_service.generate_recommendations(fair, priv, drft, perf, xai)
        modules = [r["module"] for r in recs]
        self.assertIn("Fairness", modules)
        self.assertIn("Privacy", modules)
        self.assertIn("Drift", modules)
        self.assertIn("Explainability", modules)

    # ─── 4. Database CRUD Persistence Tests ───────────────────────────────────

    def test_database_crud_flow(self):
        import uuid
        uid = uuid.uuid4().hex[:6]
        ds_id = f"ds_test_{uid}"
        m_id = f"m_test_{uid}"
        a_id = f"a_test_{uid}"

        # 1. Dataset
        ds = crud.save_dataset(ds_id, "German Credit", "UCI", "Open", 1000)
        self.assertEqual(ds["dataset_id"], ds_id)

        # 2. Model
        mdl = crud.save_model(m_id, "XGBoost_Credit", "2.1.0", "xgboost", "run_abc")
        self.assertEqual(mdl["model_id"], m_id)

        # 3. Assessment
        ass = crud.create_assessment(a_id, m_id, ds_id, "age", "completed", 0.68, "High")
        self.assertEqual(ass["risk_level"], "High")

        # 4. Module Results
        crud.save_module_result(a_id, "fairness", {"dpd": 0.15})
        crud.save_module_result(a_id, "drift", {"psi": 0.28})

        # 5. Recommendations
        crud.save_recommendations(a_id, [{"severity": "High", "text": "Fix drift on credit score"}])

        # 6. Human Review (FR-18)
        rev = crud.save_review(a_id, "Auditor Jane", "Approved", "Mitigation plan approved.")
        self.assertEqual(rev["decision"], "Approved")

        # 7. Verification of retrieval
        retrieved = crud.get_assessment(a_id)
        self.assertEqual(retrieved["assessment_id"], a_id)
        self.assertIn("fairness", retrieved["results"])
        self.assertEqual(len(retrieved["recommendations"]), 1)
        self.assertEqual(len(retrieved["reviews"]), 1)

        # 8. Monitoring Run
        mon = crud.save_monitoring_run(m_id, "batch_credit_oct", {"drift_share": 0.05})
        runs = crud.get_monitoring_runs(m_id)
        self.assertTrue(any(r["run_id"] == mon["run_id"] for r in runs))


if __name__ == "__main__":
    unittest.main()
