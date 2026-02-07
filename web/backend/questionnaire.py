"""
MirAI Backend - Questionnaire Service
Question definitions, assessment management, and response handling
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime

from database import get_db
from models import User, Assessment, QuestionResponse, RiskScore
from auth import get_current_user
from ml_service import get_ml_service, FEATURE_LABELS

router = APIRouter(prefix="/api/questionnaire", tags=["questionnaire"])


# Stage-1 Questions Definition
STAGE1_QUESTIONS = [
    # Section 1: Demographics
    {
        "section": "Demographics & Basic Information",
        "section_id": 1,
        "questions": [
            {
                "key": "age",
                "text": "What is your age?",
                "type": "number",
                "min": 1,
                "max": 100,
                "required": True
            },
            {
                "key": "sex",
                "text": "What is your biological sex?",
                "type": "select",
                "options": [
                    {"value": "1", "label": "Male"},
                    {"value": "0", "label": "Female"}
                ],
                "required": True
            },
            {
                "key": "education_years",
                "text": "How many years of formal education have you completed?",
                "type": "number",
                "min": 0,
                "max": 25,
                "required": True
            },
            {
                "key": "employment_status",
                "text": "What is your current employment status?",
                "type": "select",
                "options": [
                    {"value": "0", "label": "Currently working"},
                    {"value": "1", "label": "Retired"},
                    {"value": "2", "label": "Not currently employed"}
                ],
                "required": True
            }
        ]
    },
    # Section 2: Family History
    {
        "section": "Family History",
        "section_id": 2,
        "questions": [
            {
                "key": "family_history_dementia",
                "text": "Do you have a family history of Alzheimer's disease or dementia?",
                "type": "select",
                "options": [
                    {"value": "1", "label": "Yes"},
                    {"value": "0", "label": "No"},
                    {"value": "-1", "label": "Not sure"}
                ],
                "required": True
            },
            {
                "key": "family_relation_degree",
                "text": "If yes, how closely related was/is this family member?",
                "type": "select",
                "options": [
                    {"value": "2", "label": "First-degree relative (parent, sibling)"},
                    {"value": "1", "label": "Second-degree relative (grandparent, aunt/uncle)"},
                    {"value": "0", "label": "Not applicable / No family history"}
                ],
                "required": True,
                "depends_on": {"key": "family_history_dementia", "value": "1"}
            }
        ]
    },
    # Section 3: Memory & Cognitive Changes (Subjective)
    {
        "section": "Memory & Daily Changes",
        "section_id": 3,
        "questions": [
            {
                "key": "judgment_problems",
                "text": "Have you noticed problems making decisions or poor judgment (e.g., financial decisions, planning)?",
                "type": "select",
                "options": [
                    {"value": "1", "label": "Yes, I have noticed this"},
                    {"value": "0", "label": "No, I haven't noticed this"}
                ],
                "required": True
            },
            {
                "key": "reduced_interest",
                "text": "Have you lost interest in hobbies or activities you used to enjoy?",
                "type": "select",
                "options": [
                    {"value": "1", "label": "Yes"},
                    {"value": "0", "label": "No"}
                ],
                "required": True
            },
            {
                "key": "repetition_issues",
                "text": "Do you find yourself repeating the same questions, stories, or statements?",
                "type": "select",
                "options": [
                    {"value": "1", "label": "Yes, this happens often"},
                    {"value": "0", "label": "No, or rarely"}
                ],
                "required": True
            },
            {
                "key": "learning_difficulty",
                "text": "Do you have trouble learning how to use new tools, appliances, or technology?",
                "type": "select",
                "options": [
                    {"value": "1", "label": "Yes, this is a challenge"},
                    {"value": "0", "label": "No, I can learn new things"}
                ],
                "required": True
            },
            {
                "key": "temporal_disorientation",
                "text": "Do you sometimes forget the current month, year, or day of the week?",
                "type": "select",
                "options": [
                    {"value": "1", "label": "Yes, this happens"},
                    {"value": "0", "label": "No, I stay oriented"}
                ],
                "required": True
            },
            {
                "key": "financial_difficulty",
                "text": "Do you have trouble handling complex financial tasks (e.g., paying bills, managing accounts)?",
                "type": "select",
                "options": [
                    {"value": "1", "label": "Yes, I struggle with this"},
                    {"value": "0", "label": "No, I manage fine"}
                ],
                "required": True
            },
            {
                "key": "appointment_memory",
                "text": "Do you frequently forget appointments or scheduled events?",
                "type": "select",
                "options": [
                    {"value": "1", "label": "Yes, often"},
                    {"value": "0", "label": "No, or rarely"}
                ],
                "required": True
            },
            {
                "key": "daily_memory_problems",
                "text": "Do you experience daily problems with thinking and/or memory that affect your routine?",
                "type": "select",
                "options": [
                    {"value": "1", "label": "Yes, daily challenges"},
                    {"value": "0", "label": "No significant daily issues"}
                ],
                "required": True
            }
        ]
    },
    # Section 4: Functional Daily Living
    {
        "section": "Daily Living Activities",
        "section_id": 4,
        "questions": [
            {
                "key": "adl_assistance",
                "text": "Do you need help with basic daily activities (bathing, dressing, eating)?",
                "type": "select",
                "options": [
                    {"value": "0", "label": "No, I am fully independent"},
                    {"value": "1", "label": "Sometimes need assistance"},
                    {"value": "2", "label": "Frequently need assistance"}
                ],
                "required": True
            },
            {
                "key": "navigation_difficulty",
                "text": "Do you ever have difficulty finding your way in familiar places?",
                "type": "select",
                "options": [
                    {"value": "1", "label": "Yes, this has happened"},
                    {"value": "0", "label": "No, I navigate well"}
                ],
                "required": True
            }
        ]
    },
    # Section 5: Health & Lifestyle
    {
        "section": "Health & Lifestyle Factors",
        "section_id": 5,
        "questions": [
            {
                "key": "diabetes",
                "text": "Have you been diagnosed with diabetes?",
                "type": "select",
                "options": [
                    {"value": "1", "label": "Yes"},
                    {"value": "0", "label": "No"}
                ],
                "required": True
            },
            {
                "key": "hypertension",
                "text": "Have you been diagnosed with high blood pressure (hypertension)?",
                "type": "select",
                "options": [
                    {"value": "1", "label": "Yes"},
                    {"value": "0", "label": "No"}
                ],
                "required": True
            },
            {
                "key": "stroke_history",
                "text": "Have you ever had a stroke or mini-stroke (TIA)?",
                "type": "select",
                "options": [
                    {"value": "1", "label": "Yes"},
                    {"value": "0", "label": "No"}
                ],
                "required": True
            },
            {
                "key": "physical_activity",
                "text": "How would you describe your typical physical activity level?",
                "type": "select",
                "options": [
                    {"value": "0", "label": "Sedentary (little to no exercise)"},
                    {"value": "1", "label": "Light (occasional walks, light activity)"},
                    {"value": "2", "label": "Moderate (regular exercise, 2-3 times/week)"},
                    {"value": "3", "label": "Active (frequent exercise, 4+ times/week)"}
                ],
                "required": True
            },
            {
                "key": "smoking_status",
                "text": "What is your smoking history?",
                "type": "select",
                "options": [
                    {"value": "0", "label": "Never smoked"},
                    {"value": "1", "label": "Former smoker (quit more than 1 year ago)"},
                    {"value": "2", "label": "Current smoker"}
                ],
                "required": True
            },
            {
                "key": "sleep_hours",
                "text": "On average, how many hours of sleep do you get per night?",
                "type": "number",
                "min": 2,
                "max": 14,
                "required": True
            },
            {
                "key": "depression",
                "text": "Have you been diagnosed with depression or experience persistent low mood?",
                "type": "select",
                "options": [
                    {"value": "1", "label": "Yes"},
                    {"value": "0", "label": "No"}
                ],
                "required": True
            },
            {
                "key": "head_injury",
                "text": "Have you ever had a significant head injury or concussion?",
                "type": "select",
                "options": [
                    {"value": "1", "label": "Yes"},
                    {"value": "0", "label": "No"}
                ],
                "required": True
            },
            {
                "key": "high_cholesterol",
                "text": "Do you have high cholesterol or take cholesterol-lowering medication?",
                "type": "select",
                "options": [
                    {"value": "1", "label": "Yes"},
                    {"value": "0", "label": "No"}
                ],
                "required": True
            }
        ]
    }
]


# Request/Response Models
class StartAssessmentRequest(BaseModel):
    token: str


class SubmitAnswersRequest(BaseModel):
    token: str
    assessment_id: int
    answers: Dict[str, Any]


class AssessmentResponse(BaseModel):
    id: int
    stage: int
    status: str
    created_at: datetime
    completed_at: Optional[datetime] = None


class RiskResultResponse(BaseModel):
    assessment_id: int
    risk_score: float
    probability: float
    risk_category: str
    contributing_factors: List[Dict[str, Any]]
    message: str


@router.get("/questions")
def get_questions():
    """Get all Stage-1 questionnaire questions"""
    return {
        "stage": 1,
        "title": "Stage-1 Clinical Screening",
        "description": "This questionnaire collects information about your health, lifestyle, and daily experiences to assess early indicators of cognitive health.",
        "disclaimer": "This is a research screening tool and is NOT a diagnostic test. Results should be discussed with a healthcare professional.",
        "sections": STAGE1_QUESTIONS,
        "total_questions": sum(len(s["questions"]) for s in STAGE1_QUESTIONS)
    }


@router.post("/start", response_model=AssessmentResponse)
def start_assessment(request: StartAssessmentRequest, db: Session = Depends(get_db)):
    """Start a new Stage-1 assessment"""
    user = get_current_user(request.token, db)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired session"
        )
    
    # Create new assessment
    assessment = Assessment(
        user_id=user.id,
        stage=1,
        status="in_progress"
    )
    db.add(assessment)
    db.commit()
    db.refresh(assessment)
    
    return AssessmentResponse(
        id=assessment.id,
        stage=assessment.stage,
        status=assessment.status,
        created_at=assessment.created_at,
        completed_at=assessment.completed_at
    )


@router.post("/submit", response_model=RiskResultResponse)
def submit_answers(request: SubmitAnswersRequest, db: Session = Depends(get_db)):
    """Submit all answers and get risk assessment"""
    user = get_current_user(request.token, db)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired session"
        )
    
    # Get assessment
    assessment = db.query(Assessment).filter(
        Assessment.id == request.assessment_id,
        Assessment.user_id == user.id
    ).first()
    
    if not assessment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Assessment not found"
        )
    
    if assessment.status == "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Assessment already completed"
        )
    
    # Store responses
    for key, value in request.answers.items():
        response = QuestionResponse(
            assessment_id=assessment.id,
            question_key=key,
            answer_value=str(value)
        )
        db.add(response)
    
    # Run ML inference
    ml_service = get_ml_service()
    if not ml_service.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML service not available"
        )
    
    result = ml_service.predict_risk(request.answers)
    
    # Store risk score
    risk_score = RiskScore(
        assessment_id=assessment.id,
        stage=1,
        risk_score=result["risk_score"],
        probability=result["probability"],
        risk_category=result["risk_category"],
        contributing_factors=result["contributing_factors"]
    )
    db.add(risk_score)
    
    # Mark assessment complete
    assessment.status = "completed"
    assessment.completed_at = datetime.utcnow()
    
    db.commit()
    
    # Generate result message
    messages = {
        "low": "Your Stage-1 screening indicates a lower risk profile. Continue maintaining healthy habits and consult your healthcare provider for routine check-ups.",
        "moderate": "Your Stage-1 screening suggests some factors that warrant attention. Consider discussing these results with your healthcare provider for personalized guidance.",
        "elevated": "Your Stage-1 screening indicates elevated risk factors. We recommend scheduling a consultation with a healthcare professional for a comprehensive evaluation."
    }
    
    return RiskResultResponse(
        assessment_id=assessment.id,
        risk_score=result["risk_score"],
        probability=result["probability"],
        risk_category=result["risk_category"],
        contributing_factors=result["contributing_factors"],
        message=messages[result["risk_category"]]
    )


@router.get("/result/{assessment_id}", response_model=RiskResultResponse)
def get_result(assessment_id: int, token: str, db: Session = Depends(get_db)):
    """Get risk assessment result for a completed assessment"""
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
    
    if assessment.status != "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Assessment not yet completed"
        )
    
    # Get risk score
    risk = db.query(RiskScore).filter(
        RiskScore.assessment_id == assessment_id
    ).first()
    
    if not risk:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Risk score not found"
        )
    
    messages = {
        "low": "Your Stage-1 screening indicates a lower risk profile. Continue maintaining healthy habits and consult your healthcare provider for routine check-ups.",
        "moderate": "Your Stage-1 screening suggests some factors that warrant attention. Consider discussing these results with your healthcare provider for personalized guidance.",
        "elevated": "Your Stage-1 screening indicates elevated risk factors. We recommend scheduling a consultation with a healthcare professional for a comprehensive evaluation."
    }
    
    return RiskResultResponse(
        assessment_id=assessment.id,
        risk_score=risk.risk_score,
        probability=risk.probability,
        risk_category=risk.risk_category,
        contributing_factors=risk.contributing_factors or [],
        message=messages.get(risk.risk_category, "")
    )


@router.get("/history")
def get_assessment_history(token: str, db: Session = Depends(get_db)):
    """Get user's assessment history"""
    user = get_current_user(token, db)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired session"
        )
    
    assessments = db.query(Assessment).filter(
        Assessment.user_id == user.id
    ).order_by(Assessment.created_at.desc()).all()
    
    results = []
    for a in assessments:
        risk = db.query(RiskScore).filter(RiskScore.assessment_id == a.id).first()
        results.append({
            "id": a.id,
            "stage": a.stage,
            "status": a.status,
            "created_at": a.created_at.isoformat(),
            "completed_at": a.completed_at.isoformat() if a.completed_at else None,
            "risk_score": risk.risk_score if risk else None,
            "risk_category": risk.risk_category if risk else None
        })
    
    return {"assessments": results}
