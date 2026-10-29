"""API Integration Test Suite - FairGuard AI
Tests the 14 REST endpoints, status codes, and error contracts defined in SRS Section 8 & Section 11.
"""

import os
import sys
import unittest
import io
import pandas as pd
from fastapi.testclient import TestClient

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from api.main import app
from src.database import session


class TestFairGuardAPI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        session.init_db()
        cls.client = TestClient(app)

    # ─── System & Health ──────────────────────────────────────────────────────

    def test_01_health_endpoint(self):
        """Verify GET /health returns 200 OK with service metadata."""
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("FairGuard AI", data["service"])

    # ─── Datasets Upload & Validation ─────────────────────────────────────────

    def test_02_dataset_upload_non_csv(self):
        """Test Scenario T-02: Uploading a non-CSV file must return 400 with structured error."""
        fake_file = io.BytesIO(b"Hello World")
        resp = self.client.post(
            "/datasets",
            files={"file": ("test.txt", fake_file, "text/plain")},
            data={"name": "Invalid Text File"}
        )
        self.assertEqual(resp.status_code, 400)
        data = resp.json()
        self.assertEqual(data["error"], "invalid_input")
        self.assertIn("must be a valid .csv file", data["detail"])

    def test_03_dataset_upload_valid_csv(self):
        """Verify POST /datasets with valid CSV returns 200, row count, and quality summary."""
        csv_content = io.StringIO("age,sex,income\n25,Male,0\n35,Female,1\n45,Female,0\n")
        resp = self.client.post(
            "/datasets",
            files={"file": ("small_test.csv", io.BytesIO(csv_content.getvalue().encode()), "text/csv")},
            data={"name": "Small Test Dataset", "source": "Unit Test"}
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("d_", data["dataset_id"])
        self.assertEqual(data["row_count"], 3)
        self.assertIn("null_counts", data["quality_summary"])

    # ─── Models Upload & Validation ───────────────────────────────────────────

    def test_04_model_upload_invalid_file(self):
        """Test Scenario T-04: Uploading non-joblib/pkl must return 400 error."""
        fake_model = io.BytesIO(b"not a joblib")
        resp = self.client.post(
            "/models",
            files={"file": ("model.txt", fake_model, "text/plain")},
            data={"name": "Bad Model", "version": "1.0"}
        )
        self.assertEqual(resp.status_code, 400)
        data = resp.json()
        self.assertEqual(data["error"], "invalid_input")
        self.assertIn("must be a .joblib or .pkl file", data["detail"])

    # ─── Assessment Execution & Validation ────────────────────────────────────

    def test_05_assessment_invalid_sensitive_attribute(self):
        """Test Scenario T-03: Missing sensitive attribute must return 400 and no crash."""
        resp = self.client.post(
            "/assessments",
            json={
                "model_id": "m_hist_001",
                "dataset_id": "d_adult_001",
                "sensitive_attribute": "non_existent_attribute_xyz"
            }
        )
        self.assertEqual(resp.status_code, 400)
        data = resp.json()
        self.assertEqual(data["error"], "invalid_input")
        self.assertIn("not found in dataset", data["detail"])

    def test_06_assessment_execution_full(self):
        """Test Scenario T-01: Valid dataset & model triggers full 5-pillar assessment."""
        os.environ["FAIRGUARD_SAMPLE"] = "1"
        resp = self.client.post(
            "/assessments",
            json={
                "model_id": "m_hist_001",
                "dataset_id": "d_adult_001",
                "sensitive_attribute": "sex",
                "thresholds": {"fairness": 0.10, "psi": 0.25}
            }
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("a_", data["assessment_id"])
        self.assertEqual(data["status"], "completed")
        self.assertIn(data["risk_level"], ["Low", "Medium", "High"])
        self.assertIsNotNone(data["risk_score"])

        # Store assessment ID for subsequent sub-module tests
        TestFairGuardAPI.test_assessment_id = data["assessment_id"]

    # ─── Assessment Submodule Retrieval ───────────────────────────────────────

    def test_07_get_assessment_details(self):
        """Test Scenario T-13: Retrieve complete stored assessment by ID."""
        ass_id = getattr(self, "test_assessment_id", None)
        self.assertIsNotNone(ass_id, "Prior assessment test must succeed first")

        resp = self.client.get(f"/assessments/{ass_id}")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["assessment_id"], ass_id)
        self.assertIn("fairness", data["results"])
        self.assertIn("drift", data["results"])
        self.assertIn("privacy", data["results"])
        self.assertIn("performance", data["results"])
        self.assertGreaterEqual(len(data["recommendations"]), 1)

    def test_08_get_assessment_missing_id(self):
        """Test Scenario T-12: Request to missing assessment ID returns 404."""
        resp = self.client.get("/assessments/a_non_existent_999")
        self.assertEqual(resp.status_code, 404)
        data = resp.json()
        self.assertEqual(data["error"], "not_found")

    def test_09_submodule_endpoints(self):
        """Verify dedicated submodule endpoints (/fairness, /drift, /privacy, /performance, /explanations)."""
        ass_id = getattr(self, "test_assessment_id", None)
        self.assertIsNotNone(ass_id)

        # 1. Fairness
        f_resp = self.client.get(f"/assessments/{ass_id}/fairness")
        self.assertEqual(f_resp.status_code, 200)
        self.assertIn("demographic_parity_diff", f_resp.json())

        # 2. Drift
        d_resp = self.client.get(f"/assessments/{ass_id}/drift")
        self.assertEqual(d_resp.status_code, 200)
        self.assertIn("share_drifted", d_resp.json())
        self.assertIn("max_psi", d_resp.json())

        # 3. Privacy
        p_resp = self.client.get(f"/assessments/{ass_id}/privacy")
        self.assertEqual(p_resp.status_code, 200)
        self.assertIn("k_anonymity", p_resp.json())

        # 4. Performance
        m_resp = self.client.get(f"/assessments/{ass_id}/performance")
        self.assertEqual(m_resp.status_code, 200)
        self.assertIn("accuracy", m_resp.json())
        self.assertIn("confusion_matrix", m_resp.json())

        # 5. Explanations
        x_resp = self.client.get(f"/assessments/{ass_id}/explanations?row_index=0")
        self.assertEqual(x_resp.status_code, 200)
        self.assertIn("global_importance", x_resp.json())

    # ─── Report Export ────────────────────────────────────────────────────────

    def test_10_download_report(self):
        """Test Scenario T-14: Downloadable audit-ready report."""
        ass_id = getattr(self, "test_assessment_id", None)
        self.assertIsNotNone(ass_id)
        resp = self.client.get(f"/assessments/{ass_id}/report?format=html")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("text/html", resp.headers.get("content-type", ""))

    # ─── Governance Review ────────────────────────────────────────────────────

    def test_11_human_review(self):
        """Test Scenario T-15: Reviewer decision stored and retrieved."""
        ass_id = getattr(self, "test_assessment_id", None)
        self.assertIsNotNone(ass_id)

        # Missing reviewer name returns 400
        bad_rev = self.client.post(
            f"/assessments/{ass_id}/review",
            json={"reviewer": "   ", "decision": "Approved"}
        )
        self.assertEqual(bad_rev.status_code, 400)

        # Valid reviewer
        good_rev = self.client.post(
            f"/assessments/{ass_id}/review",
            json={
                "reviewer": "Meet Shah (Lead)",
                "decision": "Approved",
                "comment": "Performance and fairness trade-offs accepted."
            }
        )
        self.assertEqual(good_rev.status_code, 200)
        data = good_rev.json()
        self.assertEqual(data["reviewer"], "Meet Shah (Lead)")
        self.assertEqual(data["decision"], "Approved")

    # ─── Continuous Monitoring & History ──────────────────────────────────────

    def test_12_monitoring_and_history(self):
        """Verify POST /monitoring/evaluate and GET /models/{id}/history."""
        mon_resp = self.client.post(
            "/monitoring/evaluate",
            json={"model_id": "m_hist_001", "new_dataset_id": "batch_oct_2026"}
        )
        self.assertEqual(mon_resp.status_code, 200)
        mon_data = mon_resp.json()
        self.assertIn("drift_results", mon_data)

        hist_resp = self.client.get("/models/m_hist_001/history")
        self.assertEqual(hist_resp.status_code, 200)
        hist_data = hist_resp.json()
        self.assertIn("all_versions", hist_data)
        self.assertIn("assessments", hist_data)


if __name__ == "__main__":
    unittest.main()
