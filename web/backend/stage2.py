"""
MirAI Backend - Stage-2 Genetic Risk Stratification API
Full implementation for genetic risk assessment endpoints
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

from database import get_db
from models import Assessment, Stage2Data, RiskScore
from auth import get_current_user
from ml_service_stage2 import get_stage2_ml_service

router = APIRouter(prefix="/api/stage2", tags=["stage2"])


# ============ Request/Response Models ============

class Stage2SubmitRequest(BaseModel):
    """Request model for Stage-2 genetic data submission"""
    token: str
    assessment_id: int
    apoe_e4_count: int = Field(..., ge=0, le=2, description="APOE ε4 allele count (0, 1, or 2)")
    polygenic_risk_score: Optional[float] = Field(None, description="Optional PRS value")
    consent_given: bool = True


class Stage2ResultResponse(BaseModel):
    """Response model for Stage-2 results"""
    assessment_id: int
    stage2_probability: float
    stage2_risk_score: float
    stage2_risk_category: str
    feature_analysis: List[Dict[str, Any]]
    message: str


class CombinedResultResponse(BaseModel):
    """Response model for combined Stage-1 + Stage-2 results"""
    assessment_id: int
    stage1_probability: float
    stage1_risk_score: float
    stage1_risk_category: str
    stage2_probability: float
    stage2_risk_score: float
    stage2_risk_category: str
    combined_probability: float
    combined_risk_score: float
    combined_risk_category: str
    stage1_weight: float
    stage2_weight: float
    message: str


class Stage2StatusResponse(BaseModel):
    """Status response for Stage-2 availability"""
    available: bool
    ml_loaded: bool
    message: str
    description: str


# ============ Helper Functions ============

def calculate_combined_risk(stage1_prob: float, stage2_prob: float, 
                            stage1_weight: float = 0.4, stage2_weight: float = 0.6) -> dict:
    """
    Calculate combined risk using weighted fusion
    Formula: combined = (weight1 * stage1) + (weight2 * stage2)
    """
    combined_prob = (stage1_weight * stage1_prob) + (stage2_weight * stage2_prob)
    combined_prob = max(0.0, min(1.0, combined_prob))
    
    if combined_prob < 0.33:
        category = "low"
    elif combined_prob < 0.66:
        category = "moderate"
    else:
        category = "high"
    
    return {
        "probability": round(combined_prob, 4),
        "risk_score": round(combined_prob * 100, 1),
        "risk_category": category
    }


# ============ API Endpoints ============

@router.get("/status", response_model=Stage2StatusResponse)
def get_stage2_status():
    """Check if Stage-2 genetic risk stratification is available"""
    ml_service = get_stage2_ml_service()
    
    return Stage2StatusResponse(
        available=ml_service.is_loaded,
        ml_loaded=ml_service.is_loaded,
        message="Stage-2 Genetic Risk Stratification - Active" if ml_service.is_loaded 
                else "Stage-2 ML service not loaded",
        description="Stage-2 uses APOE ε4 genotype and Polygenic Risk Score (PRS) to assess genetic risk factors for Alzheimer's disease."
    )


@router.post("/submit", response_model=Stage2ResultResponse)
def submit_genetic_data(request: Stage2SubmitRequest, db: Session = Depends(get_db)):
    """
    Submit genetic data and run Stage-2 inference.
    Returns genetic risk probability and category.
    """
    user = get_current_user(request.token, db)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired session"
        )
    
    # Verify assessment exists, belongs to user, and Stage-1 is completed
    assessment = db.query(Assessment).filter(
        Assessment.id == request.assessment_id,
        Assessment.user_id == user.id
    ).first()
    
    if not assessment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Assessment not found"
        )
    
    if assessment.status != "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Stage-1 assessment must be completed before Stage-2"
        )
    
    # Check if Stage-2 already exists for this assessment
    existing = db.query(Stage2Data).filter(
        Stage2Data.assessment_id == request.assessment_id
    ).first()
    
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Stage-2 already submitted for this assessment"
        )
    
    # Run ML inference
    ml_service = get_stage2_ml_service()
    if not ml_service.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Stage-2 ML service not available"
        )
    
    inputs = {
        "APOE_e4_count": request.apoe_e4_count,
        "polygenic_risk_score": request.polygenic_risk_score
    }
    
    result = ml_service.predict_risk(inputs)
    
    # Get Stage-1 risk score for combined calculation
    stage1_risk = db.query(RiskScore).filter(
        RiskScore.assessment_id == request.assessment_id,
        RiskScore.stage == 1
    ).first()
    
    combined = None
    if stage1_risk:
        combined = calculate_combined_risk(
            stage1_risk.probability, 
            result["probability"]
        )
    
    # Store Stage-2 data
    stage2_data = Stage2Data(
        assessment_id=request.assessment_id,
        apoe_e4_count=request.apoe_e4_count,
        polygenic_risk_score=request.polygenic_risk_score,
        stage2_probability=result["probability"],
        stage2_risk_category=result["risk_category"],
        combined_probability=combined["probability"] if combined else None,
        combined_risk_category=combined["risk_category"] if combined else None,
        consent_given=1 if request.consent_given else 0
    )
    db.add(stage2_data)
    db.commit()
    db.refresh(stage2_data)
    
    # Generate message
    messages = {
        "low": "Your genetic risk profile indicates lower predisposition. Environmental and lifestyle factors remain important for overall cognitive health.",
        "moderate": "Your genetic profile shows moderate risk factors. Consider discussing genetic counseling options with your healthcare provider.",
        "high": "Your genetic profile indicates elevated risk factors. We recommend genetic counseling and regular cognitive health monitoring."
    }
    
    return Stage2ResultResponse(
        assessment_id=request.assessment_id,
        stage2_probability=result["probability"],
        stage2_risk_score=result["risk_score"],
        stage2_risk_category=result["risk_category"],
        feature_analysis=result["feature_analysis"],
        message=messages[result["risk_category"]]
    )


@router.get("/result/{assessment_id}", response_model=Stage2ResultResponse)
def get_stage2_result(assessment_id: int, token: str, db: Session = Depends(get_db)):
    """Get Stage-2 result for a completed assessment"""
    user = get_current_user(token, db)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired session"
        )
    
    # Verify assessment exists and belongs to user
    assessment = db.query(Assessment).filter(
        Assessment.id == assessment_id,
        Assessment.user_id == user.id
    ).first()
    
    if not assessment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Assessment not found"
        )
    
    # Get Stage-2 data
    stage2_data = db.query(Stage2Data).filter(
        Stage2Data.assessment_id == assessment_id
    ).first()
    
    if not stage2_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Stage-2 data not found for this assessment"
        )
    
    # Reconstruct feature analysis
    feature_analysis = [
        {
            "feature": "APOE_e4_count",
            "label": "APOE ε4 Allele Count",
            "value": stage2_data.apoe_e4_count,
            "description": f"{stage2_data.apoe_e4_count} cop{'y' if stage2_data.apoe_e4_count == 1 else 'ies'} of ε4 allele"
        }
    ]
    
    if stage2_data.polygenic_risk_score is not None:
        feature_analysis.append({
            "feature": "polygenic_risk_score",
            "label": "Polygenic Risk Score",
            "value": round(stage2_data.polygenic_risk_score, 3),
            "description": f"PRS: {stage2_data.polygenic_risk_score:.3f}"
        })
    
    messages = {
        "low": "Your genetic risk profile indicates lower predisposition.",
        "moderate": "Your genetic profile shows moderate risk factors.",
        "high": "Your genetic profile indicates elevated risk factors."
    }
    
    return Stage2ResultResponse(
        assessment_id=assessment_id,
        stage2_probability=stage2_data.stage2_probability,
        stage2_risk_score=round(stage2_data.stage2_probability * 100, 1),
        stage2_risk_category=stage2_data.stage2_risk_category,
        feature_analysis=feature_analysis,
        message=messages.get(stage2_data.stage2_risk_category, "")
    )


@router.get("/combined/{assessment_id}", response_model=CombinedResultResponse)
def get_combined_result(assessment_id: int, token: str, db: Session = Depends(get_db)):
    """Get combined Stage-1 + Stage-2 risk assessment"""
    user = get_current_user(token, db)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired session"
        )
    
    # Verify assessment
    assessment = db.query(Assessment).filter(
        Assessment.id == assessment_id,
        Assessment.user_id == user.id
    ).first()
    
    if not assessment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Assessment not found"
        )
    
    # Get Stage-1 data
    stage1_risk = db.query(RiskScore).filter(
        RiskScore.assessment_id == assessment_id,
        RiskScore.stage == 1
    ).first()
    
    if not stage1_risk:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Stage-1 results not found"
        )
    
    # Get Stage-2 data
    stage2_data = db.query(Stage2Data).filter(
        Stage2Data.assessment_id == assessment_id
    ).first()
    
    if not stage2_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Stage-2 results not found. Please complete Stage-2 assessment first."
        )
    
    # Calculate combined risk
    stage1_weight = 0.6
    stage2_weight = 0.4
    combined = calculate_combined_risk(
        stage1_risk.probability,
        stage2_data.stage2_probability,
        stage1_weight,
        stage2_weight
    )
    
    # Update Stage-2 data with combined result
    stage2_data.combined_probability = combined["probability"]
    stage2_data.combined_risk_category = combined["risk_category"]
    db.commit()
    
    # Generate message
    messages = {
        "low": "Your combined clinical and genetic assessment indicates lower overall risk. Continue maintaining healthy lifestyle habits.",
        "moderate": "Your combined assessment suggests moderate risk. Consider discussing these results with your healthcare provider for personalized recommendations.",
        "high": "Your combined assessment indicates elevated risk. We strongly recommend a comprehensive evaluation with a specialist and exploring Stage-3 biomarker screening."
    }
    
    return CombinedResultResponse(
        assessment_id=assessment_id,
        stage1_probability=stage1_risk.probability,
        stage1_risk_score=stage1_risk.risk_score,
        stage1_risk_category=stage1_risk.risk_category,
        stage2_probability=stage2_data.stage2_probability,
        stage2_risk_score=round(stage2_data.stage2_probability * 100, 1),
        stage2_risk_category=stage2_data.stage2_risk_category,
        combined_probability=combined["probability"],
        combined_risk_score=combined["risk_score"],
        combined_risk_category=combined["risk_category"],
        stage1_weight=stage1_weight,
        stage2_weight=stage2_weight,
        message=messages[combined["risk_category"]]
    )
