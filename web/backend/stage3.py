"""
MirAI Backend - Stage-3 Biomarker Risk Assessment API
Plasma biomarker evaluation for early Alzheimer's pathology detection
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

from database import get_db
from auth import get_current_user
from models import Assessment, RiskScore, Stage2Data, Stage3Data
from ml_service_stage3 import get_stage3_ml_service

router = APIRouter(prefix="/api/stage3", tags=["stage3"])


# ============ Pydantic Models ============

class Stage3StatusResponse(BaseModel):
    available: bool
    ml_loaded: bool
    message: str
    description: str
    required_biomarkers: List[str]


class Stage3SubmitRequest(BaseModel):
    """Request model for Stage-3 biomarker submission"""
    token: str
    assessment_id: int
    ptau217: Optional[float] = Field(None, description="Plasma pTau-217 (pg/mL)")
    ptau217_abeta42_ratio: Optional[float] = Field(None, description="pTau-217/Aβ42 ratio")
    nfl: Optional[float] = Field(None, description="Neurofilament Light (pg/mL)")
    consent_given: bool = True


class Stage3ResultResponse(BaseModel):
    """Response model for Stage-3 results"""
    assessment_id: int
    stage3_probability: float
    stage3_risk_score: float
    stage3_risk_category: str
    biomarkers_provided: Dict[str, Any]
    message: str


class FinalResultResponse(BaseModel):
    """Response model for final combined results"""
    assessment_id: int
    
    # Stage-1
    stage1_probability: float
    stage1_risk_score: float
    stage1_risk_category: str
    
    # Stage-2
    stage2_probability: float
    stage2_risk_score: float
    stage2_risk_category: str
    
    # Stage-3
    stage3_probability: float
    stage3_risk_score: float
    stage3_risk_category: str
    
    # Final Combined
    final_probability: float
    final_risk_score: float
    final_risk_category: str
    final_recommendation: str
    
    # Weights used
    weights: Dict[str, float]


# ============ Helper Functions ============

def calculate_final_risk(stage1_prob: float, stage2_prob: float, stage3_prob: float) -> dict:
    """
    Calculate final combined risk using weighted fusion.
    Formula: final = (0.25 * Stage-1) + (0.35 * Stage-2) + (0.40 * Stage-3)
    """
    STAGE1_WEIGHT = 0.25
    STAGE2_WEIGHT = 0.35
    STAGE3_WEIGHT = 0.40
    
    final_prob = (STAGE1_WEIGHT * stage1_prob) + \
                 (STAGE2_WEIGHT * stage2_prob) + \
                 (STAGE3_WEIGHT * stage3_prob)
    
    final_prob = max(0.0, min(1.0, final_prob))
    
    # Determine category
    if final_prob < 0.33:
        category = "low"
        recommendation = "Routine monitoring recommended. Continue healthy lifestyle practices and schedule regular check-ups with your primary care physician."
    elif final_prob < 0.66:
        category = "moderate"
        recommendation = "Annual biomarker testing recommended. Consider cognitive assessments and discuss preventive measures with your healthcare provider."
    else:
        category = "high"
        recommendation = "Neurologist consultation recommended. Advanced neuroimaging (MRI/PET) and comprehensive cognitive evaluation advised for further assessment."
    
    return {
        "probability": round(final_prob, 4),
        "risk_score": round(final_prob * 100, 1),
        "risk_category": category,
        "recommendation": recommendation,
        "weights": {
            "stage1": STAGE1_WEIGHT,
            "stage2": STAGE2_WEIGHT,
            "stage3": STAGE3_WEIGHT
        }
    }


# ============ API Endpoints ============

@router.get("/status", response_model=Stage3StatusResponse)
def get_stage3_status():
    """Check if Stage-3 biomarker assessment is available"""
    ml_service = get_stage3_ml_service()
    
    return Stage3StatusResponse(
        available=ml_service.is_loaded,
        ml_loaded=ml_service.is_loaded,
        message="Stage-3 Biomarker Screening - Active" if ml_service.is_loaded 
                else "Stage-3 ML service not loaded",
        description="Stage-3 evaluates plasma biomarkers (pTau-217, pTau-217/Aβ42 ratio, NfL) to detect early Alzheimer's pathology with high specificity.",
        required_biomarkers=["ptau217", "ptau217_abeta42_ratio", "nfl"]
    )


@router.post("/submit", response_model=Stage3ResultResponse)
def submit_biomarker_data(request: Stage3SubmitRequest, db: Session = Depends(get_db)):
    """
    Submit biomarker data and run Stage-3 inference.
    Only accessible after Stage-2 completion.
    """
    user = get_current_user(request.token, db)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired session"
        )
    
    # Verify assessment exists and belongs to user
    assessment = db.query(Assessment).filter(
        Assessment.id == request.assessment_id,
        Assessment.user_id == user.id
    ).first()
    
    if not assessment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Assessment not found"
        )
    
    # Verify Stage-1 is completed
    stage1_risk = db.query(RiskScore).filter(
        RiskScore.assessment_id == request.assessment_id,
        RiskScore.stage == 1
    ).first()
    
    if not stage1_risk:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Stage-1 assessment must be completed first"
        )
    
    # Verify Stage-2 is completed
    stage2_data = db.query(Stage2Data).filter(
        Stage2Data.assessment_id == request.assessment_id
    ).first()
    
    if not stage2_data or stage2_data.stage2_probability is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Stage-2 genetic assessment must be completed first"
        )
    
    # Check if Stage-3 already exists
    existing_stage3 = db.query(Stage3Data).filter(
        Stage3Data.assessment_id == request.assessment_id
    ).first()
    
    if existing_stage3:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Stage-3 already submitted for this assessment"
        )
    
    # Run ML inference
    ml_service = get_stage3_ml_service()
    if not ml_service.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Stage-3 ML service not available"
        )
    
    # Construct inputs for ML model (needs 5 features)
    inputs = {
        "stage2_probability": stage2_data.stage2_probability,
        "ptau217": request.ptau217,
        "nfl": request.nfl,
        # ab42 and ab40 are not in UI, so they will be None and imputed
        "ab42": None,
        "ab40": None
    }
    
    try:
        result = ml_service.predict_risk(inputs)
    except Exception as e:
        print(f"Stage-3 Inference Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error running biomarker analysis: {str(e)}"
        )
    
    # Calculate final combined risk
    final_result = calculate_final_risk(
        stage1_prob=stage1_risk.probability,
        stage2_prob=stage2_data.stage2_probability,
        stage3_prob=result["probability"]
    )
    
    # Store Stage-3 data
    stage3_data = Stage3Data(
        assessment_id=request.assessment_id,
        ptau217=request.ptau217,
        ptau217_abeta42_ratio=request.ptau217_abeta42_ratio,
        nfl=request.nfl,
        stage3_probability=result["probability"],
        stage3_risk_category=result["risk_category"],
        final_probability=final_result["probability"],
        final_risk_category=final_result["risk_category"],
        final_recommendation=final_result["recommendation"],
        consent_given=1 if request.consent_given else 0
    )
    
    db.add(stage3_data)
    
    # Update assessment stage
    assessment.stage = 3
    
    db.commit()
    db.refresh(stage3_data)
    
    return Stage3ResultResponse(
        assessment_id=request.assessment_id,
        stage3_probability=result["probability"],
        stage3_risk_score=result["risk_score"],
        stage3_risk_category=result["risk_category"],
        biomarkers_provided={
            "ptau217": request.ptau217,
            "ptau217_abeta42_ratio": request.ptau217_abeta42_ratio,
            "nfl": request.nfl
        },
        message=f"Stage-3 biomarker analysis complete. Biomarker risk: {result['risk_category']}"
    )


@router.get("/result/{assessment_id}", response_model=Stage3ResultResponse)
def get_stage3_result(assessment_id: int, token: str, db: Session = Depends(get_db)):
    """Get Stage-3 results for an assessment"""
    user = get_current_user(token, db)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired session"
        )
    
    # Get assessment
    assessment = db.query(Assessment).filter(
        Assessment.id == assessment_id,
        Assessment.user_id == user.id
    ).first()
    
    if not assessment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Assessment not found"
        )
    
    # Get Stage-3 data
    stage3_data = db.query(Stage3Data).filter(
        Stage3Data.assessment_id == assessment_id
    ).first()
    
    if not stage3_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Stage-3 data not found. Please complete biomarker screening first."
        )
    
    return Stage3ResultResponse(
        assessment_id=assessment_id,
        stage3_probability=stage3_data.stage3_probability,
        stage3_risk_score=round(stage3_data.stage3_probability * 100, 1),
        stage3_risk_category=stage3_data.stage3_risk_category,
        biomarkers_provided={
            "ptau217": stage3_data.ptau217,
            "ptau217_abeta42_ratio": stage3_data.ptau217_abeta42_ratio,
            "nfl": stage3_data.nfl
        },
        message=f"Stage-3 biomarker risk: {stage3_data.stage3_risk_category}"
    )


@router.get("/final/{assessment_id}", response_model=FinalResultResponse)
def get_final_results(assessment_id: int, token: str, db: Session = Depends(get_db)):
    """Get final combined results from all three stages"""
    user = get_current_user(token, db)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired session"
        )
    
    # Get assessment
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
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Stage-1 not completed"
        )
    
    # Get Stage-2 data
    stage2_data = db.query(Stage2Data).filter(
        Stage2Data.assessment_id == assessment_id
    ).first()
    
    if not stage2_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Stage-2 not completed"
        )
    
    # Get Stage-3 data
    stage3_data = db.query(Stage3Data).filter(
        Stage3Data.assessment_id == assessment_id
    ).first()
    
    if not stage3_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Stage-3 not completed"
        )
    
    return FinalResultResponse(
        assessment_id=assessment_id,
        
        # Stage-1
        stage1_probability=stage1_risk.probability,
        stage1_risk_score=stage1_risk.risk_score,
        stage1_risk_category=stage1_risk.risk_category,
        
        # Stage-2
        stage2_probability=stage2_data.stage2_probability,
        stage2_risk_score=round(stage2_data.stage2_probability * 100, 1),
        stage2_risk_category=stage2_data.stage2_risk_category,
        
        # Stage-3
        stage3_probability=stage3_data.stage3_probability,
        stage3_risk_score=round(stage3_data.stage3_probability * 100, 1),
        stage3_risk_category=stage3_data.stage3_risk_category,
        
        # Final
        final_probability=stage3_data.final_probability,
        final_risk_score=round(stage3_data.final_probability * 100, 1),
        final_risk_category=stage3_data.final_risk_category,
        final_recommendation=stage3_data.final_recommendation,
        
        # Weights
        weights={
            "stage1": 0.25,
            "stage2": 0.35,
            "stage3": 0.40
        }
    )
