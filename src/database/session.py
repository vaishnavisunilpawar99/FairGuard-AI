"""Database Session & Connection Management - FairGuard AI
Supports PostgreSQL for containerized / production deployments
and SQLite for frictionless local execution and testing.
"""

import os
import sqlite3
from contextlib import contextmanager
from typing import Generator

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
SQLITE_DB_PATH = os.path.join(ARTIFACTS_DIR, "fairguard.db")

DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{SQLITE_DB_PATH}")


def is_postgres() -> bool:
    return DATABASE_URL.startswith("postgresql://") or DATABASE_URL.startswith("postgres://")


@contextmanager
def get_db_connection():
    """Context manager yielding an active database connection."""
    if is_postgres():
        import psycopg2
        import psycopg2.extras
        conn = psycopg2.connect(DATABASE_URL)
        conn.autocommit = False
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    else:
        conn = sqlite3.connect(SQLITE_DB_PATH)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()


def init_db():
    """Create all 7 core tables if they do not exist."""
    schema_sql = """
    CREATE TABLE IF NOT EXISTS datasets (
        dataset_id VARCHAR(64) PRIMARY KEY,
        name VARCHAR(255) NOT NULL,
        source VARCHAR(255),
        licence VARCHAR(100),
        row_count INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS models (
        model_id VARCHAR(64) PRIMARY KEY,
        name VARCHAR(255) NOT NULL,
        version VARCHAR(64) NOT NULL,
        framework VARCHAR(100),
        mlflow_run_id VARCHAR(128),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS assessments (
        assessment_id VARCHAR(64) PRIMARY KEY,
        model_id VARCHAR(64) NOT NULL,
        dataset_id VARCHAR(64) NOT NULL,
        sensitive_attribute VARCHAR(100) NOT NULL,
        status VARCHAR(64) NOT NULL,
        risk_score REAL,
        risk_level VARCHAR(32),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (model_id) REFERENCES models (model_id) ON DELETE CASCADE,
        FOREIGN KEY (dataset_id) REFERENCES datasets (dataset_id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS module_results (
        result_id VARCHAR(64) PRIMARY KEY,
        assessment_id VARCHAR(64) NOT NULL,
        module VARCHAR(64) NOT NULL,
        metrics_json TEXT NOT NULL,
        FOREIGN KEY (assessment_id) REFERENCES assessments (assessment_id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS recommendations (
        rec_id VARCHAR(64) PRIMARY KEY,
        assessment_id VARCHAR(64) NOT NULL,
        severity VARCHAR(32) NOT NULL,
        text TEXT NOT NULL,
        FOREIGN KEY (assessment_id) REFERENCES assessments (assessment_id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS reviews (
        review_id VARCHAR(64) PRIMARY KEY,
        assessment_id VARCHAR(64) NOT NULL,
        reviewer VARCHAR(255) NOT NULL,
        decision VARCHAR(64) NOT NULL,
        comment TEXT,
        reviewed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (assessment_id) REFERENCES assessments (assessment_id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS monitoring_runs (
        run_id VARCHAR(64) PRIMARY KEY,
        model_id VARCHAR(64) NOT NULL,
        new_data_ref VARCHAR(255) NOT NULL,
        drift_json TEXT NOT NULL,
        run_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (model_id) REFERENCES models (model_id) ON DELETE CASCADE
    );
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        if is_postgres():
            cursor.execute(schema_sql)
        else:
            conn.executescript(schema_sql)


# Initialize schema on module load
try:
    init_db()
except Exception as e:
    print(f"Database initialization warning: {e}")
