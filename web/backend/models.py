"""
MirAI Backend - Database Models
SQLAlchemy ORM models for users, assessments, and risk scores
"""
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text, JSON
from sqlalchemy.orm import relationship
from datetime import datetime
from database import Base


class User(Base):
    """User account model"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    assessments = relationship("Assessment", back_populates="user")
    sessions = relationship("UserSession", back_populates="user")


class UserSession(Base):
    """User session for authentication"""
    __tablename__ = "user_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    token = Column(String(255), unique=True, index=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="sessions")


class Assessment(Base):
    """Assessment session model"""
    __tablename__ = "assessments"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    stage = Column(Integer, default=1)  # 1 = Stage-1, 2 = Stage-2
    status = Column(String(20), default="in_progress")  # in_progress, completed
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="assessments")
    responses = relationship("QuestionResponse", back_populates="assessment")
    risk_scores = relationship("RiskScore", back_populates="assessment")
    stage2_data = relationship("Stage2Data", back_populates="assessment")
    stage3_data = relationship("Stage3Data", back_populates="assessment")


class QuestionResponse(Base):
    """Individual question response"""
    __tablename__ = "question_responses"
    
    id = Column(Integer, primary_key=True, index=True)
    assessment_id = Column(Integer, ForeignKey("assessments.id"), nullable=False)
    question_key = Column(String(50), nullable=False)  # Variable name
    answer_value = Column(String(255), nullable=False)  # User's answer
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    assessment = relationship("Assessment", back_populates="responses")


class RiskScore(Base):
    """Risk score result from ML model"""
    __tablename__ = "risk_scores"
    
    id = Column(Integer, primary_key=True, index=True)
    assessment_id = Column(Integer, ForeignKey("assessments.id"), nullable=False)
    stage = Column(Integer, default=1)
    risk_score = Column(Float, nullable=False)  # 0-100 scale
    probability = Column(Float, nullable=False)  # 0-1 probability
    risk_category = Column(String(20), nullable=False)  # low, moderate, elevated
    contributing_factors = Column(JSON, nullable=True)  # List of factors
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    assessment = relationship("Assessment", back_populates="risk_scores")


class Stage2Data(Base):
    """Stage-2 genetic risk data"""
    __tablename__ = "stage2_data"
    
    id = Column(Integer, primary_key=True, index=True)
    assessment_id = Column(Integer, ForeignKey("assessments.id"), nullable=False)
    
    # Genetic inputs
    apoe_e4_count = Column(Integer, nullable=False)  # 0, 1, or 2 copies
    polygenic_risk_score = Column(Float, nullable=True)  # PRS (optional)
    
    # Risk outputs
    stage2_probability = Column(Float, nullable=True)  # 0-1 probability
    stage2_risk_category = Column(String(20), nullable=True)  # low, moderate, high
    
    # Combined risk (Stage-1 + Stage-2)
    combined_probability = Column(Float, nullable=True)
    combined_risk_category = Column(String(20), nullable=True)
    
    # Metadata
    consent_given = Column(Integer, default=1)  # User consented to genetic screening
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    assessment = relationship("Assessment", back_populates="stage2_data")


class Stage3Data(Base):
    """Stage-3 biomarker risk data"""
    __tablename__ = "stage3_data"
    
    id = Column(Integer, primary_key=True, index=True)
    assessment_id = Column(Integer, ForeignKey("assessments.id"), nullable=False)
    
    # Biomarker inputs
    ptau217 = Column(Float, nullable=True)              # Plasma pTau-217 (pg/mL)
    ptau217_abeta42_ratio = Column(Float, nullable=True)  # pTau-217/Aβ42 ratio
    nfl = Column(Float, nullable=True)                  # Neurofilament Light (pg/mL)
    
    # Stage-3 risk outputs
    stage3_probability = Column(Float, nullable=True)   # 0-1 probability
    stage3_risk_category = Column(String(20), nullable=True)  # low, moderate, high
    
    # Final combined risk (Stage-1 + Stage-2 + Stage-3)
    final_probability = Column(Float, nullable=True)
    final_risk_category = Column(String(20), nullable=True)
    final_recommendation = Column(Text, nullable=True)
    
    # Metadata
    consent_given = Column(Integer, default=1)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    assessment = relationship("Assessment", back_populates="stage3_data")


