"""Database CRUD Operations - FairGuard AI
Implements transactional persistence for datasets, models, assessments,
module results, recommendations, reviews, and monitoring runs (FR-13).
"""

import json
import uuid
from typing import List, Dict, Any, Optional
from .session import get_db_connection, is_postgres


# ─── Datasets ─────────────────────────────────────────────────────────────────

def save_dataset(
    dataset_id: str,
    name: str,
    source: str,
    licence: str,
    row_count: int
) -> Dict[str, Any]:
    with get_db_connection() as conn:
        cursor = conn.cursor()
        query = """
        INSERT INTO datasets (dataset_id, name, source, licence, row_count)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (dataset_id) DO UPDATE SET
            name = EXCLUDED.name,
            source = EXCLUDED.source,
            licence = EXCLUDED.licence,
            row_count = EXCLUDED.row_count;
        """ if is_postgres() else """
        INSERT INTO datasets (dataset_id, name, source, licence, row_count)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT (dataset_id) DO UPDATE SET
            name = excluded.name,
            source = excluded.source,
            licence = excluded.licence,
            row_count = excluded.row_count;
        """
        cursor.execute(query, (dataset_id, name, source, licence, row_count))
    return get_dataset(dataset_id)


def get_dataset(dataset_id: str) -> Optional[Dict[str, Any]]:
    with get_db_connection() as conn:
        cursor = conn.cursor()
        query = "SELECT * FROM datasets WHERE dataset_id = %s;" if is_postgres() else "SELECT * FROM datasets WHERE dataset_id = ?;"
        cursor.execute(query, (dataset_id,))
        row = cursor.fetchone()
        if not row:
            return None
        return dict(row)


def list_datasets() -> List[Dict[str, Any]]:
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM datasets ORDER BY created_at DESC;")
        return [dict(r) for r in cursor.fetchall()]


# ─── Models ───────────────────────────────────────────────────────────────────

def save_model(
    model_id: str,
    name: str,
    version: str,
    framework: str,
    mlflow_run_id: Optional[str] = None
) -> Dict[str, Any]:
    with get_db_connection() as conn:
        cursor = conn.cursor()
        query = """
        INSERT INTO models (model_id, name, version, framework, mlflow_run_id)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (model_id) DO UPDATE SET
            name = EXCLUDED.name,
            version = EXCLUDED.version,
            framework = EXCLUDED.framework,
            mlflow_run_id = EXCLUDED.mlflow_run_id;
        """ if is_postgres() else """
        INSERT INTO models (model_id, name, version, framework, mlflow_run_id)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT (model_id) DO UPDATE SET
            name = excluded.name,
            version = excluded.version,
            framework = excluded.framework,
            mlflow_run_id = excluded.mlflow_run_id;
        """
        cursor.execute(query, (model_id, name, version, framework, mlflow_run_id))
    return get_model(model_id)


def get_model(model_id: str) -> Optional[Dict[str, Any]]:
    with get_db_connection() as conn:
        cursor = conn.cursor()
        query = "SELECT * FROM models WHERE model_id = %s;" if is_postgres() else "SELECT * FROM models WHERE model_id = ?;"
        cursor.execute(query, (model_id,))
        row = cursor.fetchone()
        return dict(row) if row else None


def list_models() -> List[Dict[str, Any]]:
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM models ORDER BY created_at DESC;")
        return [dict(r) for r in cursor.fetchall()]


def get_model_history(model_id: str) -> Dict[str, Any]:
    """Retrieve model metadata, all associated versions, and past assessments."""
    model = get_model(model_id)
    if not model:
        return {}

    with get_db_connection() as conn:
        cursor = conn.cursor()
        # Find all models with same name
        query_m = "SELECT * FROM models WHERE name = %s ORDER BY created_at DESC;" if is_postgres() else "SELECT * FROM models WHERE name = ? ORDER BY created_at DESC;"
        cursor.execute(query_m, (model["name"],))
        versions = [dict(r) for r in cursor.fetchall()]

        # Past assessments for this model
        query_a = "SELECT * FROM assessments WHERE model_id = %s ORDER BY created_at DESC;" if is_postgres() else "SELECT * FROM assessments WHERE model_id = ? ORDER BY created_at DESC;"
        cursor.execute(query_a, (model_id,))
        assessments = [dict(r) for r in cursor.fetchall()]

        return {
            "model": model,
            "all_versions": versions,
            "assessments": assessments,
        }


# ─── Assessments ──────────────────────────────────────────────────────────────

def create_assessment(
    assessment_id: str,
    model_id: str,
    dataset_id: str,
    sensitive_attribute: str,
    status: str = "pending",
    risk_score: Optional[float] = None,
    risk_level: Optional[str] = None
) -> Dict[str, Any]:
    with get_db_connection() as conn:
        cursor = conn.cursor()
        query = """
        INSERT INTO assessments (assessment_id, model_id, dataset_id, sensitive_attribute, status, risk_score, risk_level)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (assessment_id) DO UPDATE SET
            model_id = EXCLUDED.model_id,
            dataset_id = EXCLUDED.dataset_id,
            sensitive_attribute = EXCLUDED.sensitive_attribute,
            status = EXCLUDED.status,
            risk_score = EXCLUDED.risk_score,
            risk_level = EXCLUDED.risk_level;
        """ if is_postgres() else """
        INSERT INTO assessments (assessment_id, model_id, dataset_id, sensitive_attribute, status, risk_score, risk_level)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (assessment_id) DO UPDATE SET
            model_id = excluded.model_id,
            dataset_id = excluded.dataset_id,
            sensitive_attribute = excluded.sensitive_attribute,
            status = excluded.status,
            risk_score = excluded.risk_score,
            risk_level = excluded.risk_level;
        """
        cursor.execute(query, (assessment_id, model_id, dataset_id, sensitive_attribute, status, risk_score, risk_level))
    return get_assessment(assessment_id)


def update_assessment(
    assessment_id: str,
    status: str,
    risk_score: Optional[float] = None,
    risk_level: Optional[str] = None
):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        query = """
        UPDATE assessments
        SET status = %s, risk_score = %s, risk_level = %s
        WHERE assessment_id = %s;
        """ if is_postgres() else """
        UPDATE assessments
        SET status = ?, risk_score = ?, risk_level = ?
        WHERE assessment_id = ?;
        """
        cursor.execute(query, (status, risk_score, risk_level, assessment_id))


def get_assessment(assessment_id: str) -> Optional[Dict[str, Any]]:
    with get_db_connection() as conn:
        cursor = conn.cursor()
        query = "SELECT * FROM assessments WHERE assessment_id = %s;" if is_postgres() else "SELECT * FROM assessments WHERE assessment_id = ?;"
        cursor.execute(query, (assessment_id,))
        row = cursor.fetchone()
        if not row:
            return None
        res = dict(row)
        res["results"] = get_all_module_results(assessment_id)
        res["recommendations"] = get_recommendations(assessment_id)
        res["reviews"] = get_reviews(assessment_id)
        return res


def list_assessments() -> List[Dict[str, Any]]:
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM assessments ORDER BY created_at DESC;")
        return [dict(r) for r in cursor.fetchall()]


# ─── Module Results ───────────────────────────────────────────────────────────

def save_module_result(
    assessment_id: str,
    module: str,
    metrics: Dict[str, Any]
):
    result_id = f"res_{uuid.uuid4().hex[:12]}"
    metrics_str = json.dumps(metrics, default=str)
    with get_db_connection() as conn:
        cursor = conn.cursor()
        # Delete prior result for same module if re-running
        del_q = "DELETE FROM module_results WHERE assessment_id = %s AND module = %s;" if is_postgres() else "DELETE FROM module_results WHERE assessment_id = ? AND module = ?;"
        cursor.execute(del_q, (assessment_id, module))

        ins_q = """
        INSERT INTO module_results (result_id, assessment_id, module, metrics_json)
        VALUES (%s, %s, %s, %s);
        """ if is_postgres() else """
        INSERT INTO module_results (result_id, assessment_id, module, metrics_json)
        VALUES (?, ?, ?, ?);
        """
        cursor.execute(ins_q, (result_id, assessment_id, module, metrics_str))


def get_module_result(assessment_id: str, module: str) -> Optional[Dict[str, Any]]:
    with get_db_connection() as conn:
        cursor = conn.cursor()
        query = "SELECT * FROM module_results WHERE assessment_id = %s AND module = %s;" if is_postgres() else "SELECT * FROM module_results WHERE assessment_id = ? AND module = ?;"
        cursor.execute(query, (assessment_id, module))
        row = cursor.fetchone()
        if not row:
            return None
        d = dict(row)
        d["metrics"] = json.loads(d["metrics_json"])
        return d


def get_all_module_results(assessment_id: str) -> Dict[str, Any]:
    with get_db_connection() as conn:
        cursor = conn.cursor()
        query = "SELECT * FROM module_results WHERE assessment_id = %s;" if is_postgres() else "SELECT * FROM module_results WHERE assessment_id = ?;"
        cursor.execute(query, (assessment_id,))
        out = {}
        for row in cursor.fetchall():
            d = dict(row)
            out[d["module"]] = json.loads(d["metrics_json"])
        return out


# ─── Recommendations ──────────────────────────────────────────────────────────

def save_recommendations(
    assessment_id: str,
    recommendations: List[Dict[str, str]]
):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        del_q = "DELETE FROM recommendations WHERE assessment_id = %s;" if is_postgres() else "DELETE FROM recommendations WHERE assessment_id = ?;"
        cursor.execute(del_q, (assessment_id,))

        ins_q = "INSERT INTO recommendations (rec_id, assessment_id, severity, text) VALUES (%s, %s, %s, %s);" if is_postgres() else "INSERT INTO recommendations (rec_id, assessment_id, severity, text) VALUES (?, ?, ?, ?);"
        for rec in recommendations:
            rec_id = f"rec_{uuid.uuid4().hex[:12]}"
            cursor.execute(ins_q, (rec_id, assessment_id, rec.get("severity", "Medium"), rec.get("text", "")))


def get_recommendations(assessment_id: str) -> List[Dict[str, Any]]:
    with get_db_connection() as conn:
        cursor = conn.cursor()
        query = "SELECT * FROM recommendations WHERE assessment_id = %s;" if is_postgres() else "SELECT * FROM recommendations WHERE assessment_id = ?;"
        cursor.execute(query, (assessment_id,))
        return [dict(r) for r in cursor.fetchall()]


# ─── Reviews ──────────────────────────────────────────────────────────────────

def save_review(
    assessment_id: str,
    reviewer: str,
    decision: str,
    comment: Optional[str] = None
) -> Dict[str, Any]:
    review_id = f"rev_{uuid.uuid4().hex[:12]}"
    with get_db_connection() as conn:
        cursor = conn.cursor()
        ins_q = """
        INSERT INTO reviews (review_id, assessment_id, reviewer, decision, comment)
        VALUES (%s, %s, %s, %s, %s);
        """ if is_postgres() else """
        INSERT INTO reviews (review_id, assessment_id, reviewer, decision, comment)
        VALUES (?, ?, ?, ?, ?);
        """
        cursor.execute(ins_q, (review_id, assessment_id, reviewer, decision, comment or ""))
    return {"review_id": review_id, "assessment_id": assessment_id, "reviewer": reviewer, "decision": decision, "comment": comment}


def get_reviews(assessment_id: str) -> List[Dict[str, Any]]:
    with get_db_connection() as conn:
        cursor = conn.cursor()
        query = "SELECT * FROM reviews WHERE assessment_id = %s ORDER BY reviewed_at DESC;" if is_postgres() else "SELECT * FROM reviews WHERE assessment_id = ? ORDER BY reviewed_at DESC;"
        cursor.execute(query, (assessment_id,))
        return [dict(r) for r in cursor.fetchall()]


# ─── Continuous Monitoring Runs ───────────────────────────────────────────────

def save_monitoring_run(
    model_id: str,
    new_data_ref: str,
    drift_data: Dict[str, Any]
) -> Dict[str, Any]:
    run_id = f"mon_{uuid.uuid4().hex[:12]}"
    drift_str = json.dumps(drift_data, default=str)
    with get_db_connection() as conn:
        cursor = conn.cursor()
        ins_q = """
        INSERT INTO monitoring_runs (run_id, model_id, new_data_ref, drift_json)
        VALUES (%s, %s, %s, %s);
        """ if is_postgres() else """
        INSERT INTO monitoring_runs (run_id, model_id, new_data_ref, drift_json)
        VALUES (?, ?, ?, ?);
        """
        cursor.execute(ins_q, (run_id, model_id, new_data_ref, drift_str))
    return {"run_id": run_id, "model_id": model_id, "new_data_ref": new_data_ref}


def get_monitoring_runs(model_id: str) -> List[Dict[str, Any]]:
    with get_db_connection() as conn:
        cursor = conn.cursor()
        query = "SELECT * FROM monitoring_runs WHERE model_id = %s ORDER BY run_at DESC;" if is_postgres() else "SELECT * FROM monitoring_runs WHERE model_id = ? ORDER BY run_at DESC;"
        cursor.execute(query, (model_id,))
        out = []
        for r in cursor.fetchall():
            d = dict(r)
            d["drift_results"] = json.loads(d["drift_json"])
            out.append(d)
        return out
