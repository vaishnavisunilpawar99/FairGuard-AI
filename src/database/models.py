"""Database Schema Definitions - FairGuard AI
Defines ORM mappings and data structures for the 7 core tables (SRS Section 6.4).
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any

try:
    from sqlalchemy.orm import declarative_base, relationship
    from sqlalchemy import Column, String, Integer, Float, Text, DateTime, ForeignKey
    
    Base = declarative_base()

    class DatasetORM(Base):
        __tablename__ = "datasets"
        dataset_id = Column(String(64), primary_key=True)
        name = Column(String(255), nullable=False)
        source = Column(String(255))
        licence = Column(String(100))
        row_count = Column(Integer)
        created_at = Column(DateTime, default=datetime.utcnow)

        assessments = relationship("AssessmentORM", back_populates="dataset", cascade="all, delete-orphan")

    class ModelORM(Base):
        __tablename__ = "models"
        model_id = Column(String(64), primary_key=True)
        name = Column(String(255), nullable=False)
        version = Column(String(64), nullable=False)
        framework = Column(String(100))
        mlflow_run_id = Column(String(128))
        created_at = Column(DateTime, default=datetime.utcnow)

        assessments = relationship("AssessmentORM", back_populates="model", cascade="all, delete-orphan")
        monitoring_runs = relationship("MonitoringRunORM", back_populates="model", cascade="all, delete-orphan")

    class AssessmentORM(Base):
        __tablename__ = "assessments"
        assessment_id = Column(String(64), primary_key=True)
        model_id = Column(String(64), ForeignKey("models.model_id", ondelete="CASCADE"), nullable=False)
        dataset_id = Column(String(64), ForeignKey("datasets.dataset_id", ondelete="CASCADE"), nullable=False)
        sensitive_attribute = Column(String(100), nullable=False)
        status = Column(String(64), nullable=False)
        risk_score = Column(Float)
        risk_level = Column(String(32))
        created_at = Column(DateTime, default=datetime.utcnow)

        model = relationship("ModelORM", back_populates="assessments")
        dataset = relationship("DatasetORM", back_populates="assessments")
        module_results = relationship("ModuleResultORM", back_populates="assessment", cascade="all, delete-orphan")
        recommendations = relationship("RecommendationORM", back_populates="assessment", cascade="all, delete-orphan")
        reviews = relationship("ReviewORM", back_populates="assessment", cascade="all, delete-orphan")

    class ModuleResultORM(Base):
        __tablename__ = "module_results"
        result_id = Column(String(64), primary_key=True)
        assessment_id = Column(String(64), ForeignKey("assessments.assessment_id", ondelete="CASCADE"), nullable=False)
        module = Column(String(64), nullable=False)
        metrics_json = Column(Text, nullable=False)

        assessment = relationship("AssessmentORM", back_populates="module_results")

    class RecommendationORM(Base):
        __tablename__ = "recommendations"
        rec_id = Column(String(64), primary_key=True)
        assessment_id = Column(String(64), ForeignKey("assessments.assessment_id", ondelete="CASCADE"), nullable=False)
        severity = Column(String(32), nullable=False)
        text = Column(Text, nullable=False)

        assessment = relationship("AssessmentORM", back_populates="recommendations")

    class ReviewORM(Base):
        __tablename__ = "reviews"
        review_id = Column(String(64), primary_key=True)
        assessment_id = Column(String(64), ForeignKey("assessments.assessment_id", ondelete="CASCADE"), nullable=False)
        reviewer = Column(String(255), nullable=False)
        decision = Column(String(64), nullable=False)
        comment = Column(Text)
        reviewed_at = Column(DateTime, default=datetime.utcnow)

        assessment = relationship("AssessmentORM", back_populates="reviews")

    class MonitoringRunORM(Base):
        __tablename__ = "monitoring_runs"
        run_id = Column(String(64), primary_key=True)
        model_id = Column(String(64), ForeignKey("models.model_id", ondelete="CASCADE"), nullable=False)
        new_data_ref = Column(String(255), nullable=False)
        drift_json = Column(Text, nullable=False)
        run_at = Column(DateTime, default=datetime.utcnow)

        model = relationship("ModelORM", back_populates="monitoring_runs")

except ImportError:
    Base = None


# Dataclasses for standard typed representation
@dataclass
class DatasetRecord:
    dataset_id: str
    name: str
    source: str
    licence: str
    row_count: int
    created_at: Optional[str] = None


@dataclass
class ModelRecord:
    model_id: str
    name: str
    version: str
    framework: str
    mlflow_run_id: Optional[str] = None
    created_at: Optional[str] = None


@dataclass
class AssessmentRecord:
    assessment_id: str
    model_id: str
    dataset_id: str
    sensitive_attribute: str
    status: str
    risk_score: float
    risk_level: str
    created_at: Optional[str] = None


@dataclass
class ModuleResultRecord:
    result_id: str
    assessment_id: str
    module: str
    metrics_json: str


@dataclass
class RecommendationRecord:
    rec_id: str
    assessment_id: str
    severity: str
    text: str


@dataclass
class ReviewRecord:
    review_id: str
    assessment_id: str
    reviewer: str
    decision: str
    comment: Optional[str] = None
    reviewed_at: Optional[str] = None


@dataclass
class MonitoringRunRecord:
    run_id: str
    model_id: str
    new_data_ref: str
    drift_json: str
    run_at: Optional[str] = None
