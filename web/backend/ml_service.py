"""
MirAI Backend - ML Inference Service
Loads Stage-1 XGBoost model artifacts and performs risk prediction
"""
import json
import pickle
import joblib
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import xgboost as xgb

# Path to model artifacts
MODELS_DIR = Path(__file__).parent / "models"

# Feature order expected by the XGBoost model (6 features based on model structure)
# These map to questionnaire responses that get aggregated into model features
FEATURE_ORDER = [
    "cognitive_score",      # Aggregated from memory/cognition questions
    "functional_score",     # Aggregated from ADL questions
    "age_normalized",       # Age normalized
    "family_risk_score",    # Family history risk
    "cardiovascular_score", # CV risk factors
    "lifestyle_score"       # Lifestyle factors
]

# Questionnaire questions that map to each model feature
QUESTION_TO_FEATURE_MAP = {
    "cognitive_score": [
        "judgment_problems", "reduced_interest", "repetition_issues",
        "learning_difficulty", "temporal_disorientation", "financial_difficulty",
        "appointment_memory", "daily_memory_problems"
    ],
    "functional_score": ["adl_assistance", "navigation_difficulty"],
    "age_normalized": ["age"],
    "family_risk_score": ["family_history_dementia", "family_relation_degree"],
    "cardiovascular_score": ["diabetes", "hypertension", "stroke_history", "high_cholesterol"],
    "lifestyle_score": ["physical_activity", "smoking_status", "sleep_hours", "depression", "head_injury"]
}

# Feature importance for explanations
FEATURE_IMPORTANCE = {
    "cognitive_score": 0.30,
    "functional_score": 0.15,
    "age_normalized": 0.20,
    "family_risk_score": 0.15,
    "cardiovascular_score": 0.12,
    "lifestyle_score": 0.08
}

# Human-readable feature names
FEATURE_LABELS = {
    "cognitive_score": "Cognitive Function Indicators",
    "functional_score": "Daily Living Function",
    "age_normalized": "Age Factor",
    "family_risk_score": "Family History Risk",
    "cardiovascular_score": "Cardiovascular Health",
    "lifestyle_score": "Lifestyle Factors"
}

# Question labels for explanation
QUESTION_LABELS = {
    "age": "Age",
    "sex": "Biological Sex",
    "education_years": "Education Level",
    "employment_status": "Employment Status",
    "family_history_dementia": "Family History of Dementia",
    "family_relation_degree": "Close Relative with Dementia",
    "judgment_problems": "Difficulty with Judgment/Decisions",
    "reduced_interest": "Reduced Interest in Activities",
    "repetition_issues": "Repeating Questions/Stories",
    "learning_difficulty": "Difficulty Learning New Things",
    "temporal_disorientation": "Forgetting Date/Time",
    "financial_difficulty": "Trouble with Finances",
    "appointment_memory": "Forgetting Appointments",
    "daily_memory_problems": "Daily Memory Problems",
    "adl_assistance": "Need Assistance with Daily Activities",
    "navigation_difficulty": "Getting Lost in Familiar Places",
    "diabetes": "Diabetes",
    "hypertension": "High Blood Pressure",
    "stroke_history": "History of Stroke",
    "physical_activity": "Physical Activity Level",
    "smoking_status": "Smoking History",
    "sleep_hours": "Sleep Duration",
    "depression": "Depression",
    "head_injury": "History of Head Injury",
    "high_cholesterol": "High Cholesterol"
}


class MLService:
    """Machine Learning inference service for Stage-1 risk assessment using XGBoost"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.imputer = None
        self.is_loaded = False
    
    def load_artifacts(self):
        """Load XGBoost model, scaler, and imputer from disk"""
        try:
            # Load XGBoost model from JSON
            model_path = MODELS_DIR / "stage1_model.json"
            self.model = xgb.Booster()
            self.model.load_model(str(model_path))
            
            # Load scaler
            scaler_path = MODELS_DIR / "stage1_scaler.pkl"
            try:
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
            except Exception:
                # Fallback to joblib if pickle fails
                self.scaler = joblib.load(scaler_path)
            
            # Load imputer
            imputer_path = MODELS_DIR / "stage1_imputer.pkl"
            try:
                with open(imputer_path, 'rb') as f:
                    self.imputer = pickle.load(f)
            except Exception:
                # Fallback to joblib if pickle fails
                self.imputer = joblib.load(imputer_path)
            
            self.is_loaded = True
            print("✓ ML artifacts loaded successfully (XGBoost model)")
            return True
            
        except Exception as e:
            print(f"✗ Error loading ML artifacts: {e}")
            import traceback
            traceback.print_exc()
            self.is_loaded = False
            return False
    
    def _aggregate_to_model_features(self, responses: Dict[str, Any]) -> np.ndarray:
        """
        Aggregate questionnaire responses into the 6 model features
        """
        features = []
        
        # Cognitive score: average of cognitive questions (0-1 scale)
        cognitive_questions = QUESTION_TO_FEATURE_MAP["cognitive_score"]
        cognitive_values = []
        for q in cognitive_questions:
            if q in responses:
                try:
                    cognitive_values.append(float(responses[q]))
                except (ValueError, TypeError):
                    pass
        cognitive_score = np.mean(cognitive_values) if cognitive_values else 0.0
        features.append(cognitive_score)
        
        # Functional score: average of ADL questions (0-2 scale normalized to 0-1)
        functional_questions = QUESTION_TO_FEATURE_MAP["functional_score"]
        functional_values = []
        for q in functional_questions:
            if q in responses:
                try:
                    val = float(responses[q])
                    # Normalize adl_assistance from 0-2 to 0-1
                    if q == "adl_assistance":
                        val = val / 2.0
                    functional_values.append(val)
                except (ValueError, TypeError):
                    pass
        functional_score = np.mean(functional_values) if functional_values else 0.0
        features.append(functional_score)
        
        # Age normalized (40-100 -> 0-1)
        age = float(responses.get("age", 65))
        age_normalized = (age - 40) / 60.0
        age_normalized = max(0.0, min(1.0, age_normalized))
        features.append(age_normalized)
        
        # Family risk score
        family_history = float(responses.get("family_history_dementia", 0))
        family_degree = float(responses.get("family_relation_degree", 0)) / 2.0  # 0-2 -> 0-1
        family_risk = (family_history * 0.6 + family_degree * 0.4)
        features.append(family_risk)
        
        # Cardiovascular score
        cv_questions = QUESTION_TO_FEATURE_MAP["cardiovascular_score"]
        cv_values = []
        for q in cv_questions:
            if q in responses:
                try:
                    cv_values.append(float(responses[q]))
                except (ValueError, TypeError):
                    pass
        cv_score = np.mean(cv_values) if cv_values else 0.0
        features.append(cv_score)
        
        # Lifestyle score (higher physical activity = lower risk, so invert some)
        lifestyle_values = []
        if "physical_activity" in responses:
            # Invert: 0=sedentary(high risk) -> 1, 3=active(low risk) -> 0
            pa = float(responses["physical_activity"])
            lifestyle_values.append(1.0 - (pa / 3.0))
        if "smoking_status" in responses:
            # 0=never, 1=former, 2=current -> normalize
            smoke = float(responses["smoking_status"]) / 2.0
            lifestyle_values.append(smoke)
        if "sleep_hours" in responses:
            # Optimal 7-8 hours, risk increases with deviation
            sleep = float(responses["sleep_hours"])
            sleep_risk = abs(sleep - 7.5) / 5.0  # Deviation from optimal
            lifestyle_values.append(min(1.0, sleep_risk))
        if "depression" in responses:
            lifestyle_values.append(float(responses["depression"]))
        if "head_injury" in responses:
            lifestyle_values.append(float(responses["head_injury"]))
        
        lifestyle_score = np.mean(lifestyle_values) if lifestyle_values else 0.0
        features.append(lifestyle_score)
        
        return np.array(features).reshape(1, -1)
    
    def predict_risk(self, responses: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run full inference pipeline:
        1. Aggregate questionnaire responses to model features
        2. Apply imputer for missing values
        3. Apply scaler for normalization
        4. Run XGBoost model inference
        
        Returns dict with risk_score, probability, category, contributing_factors
        """
        if not self.is_loaded:
            raise RuntimeError("ML artifacts not loaded")
        
        # Step 1: Aggregate to model features
        features = self._aggregate_to_model_features(responses)
        
        # Step 2: Apply imputer (handle missing values)
        features_imputed = self.imputer.transform(features)
        
        # Step 3: Apply scaler (normalize)
        features_scaled = self.scaler.transform(features_imputed)
        
        # Step 4: XGBoost inference
        dmatrix = xgb.DMatrix(features_scaled)
        probability = float(self.model.predict(dmatrix)[0])
        
        # Ensure probability is in valid range
        probability = max(0.0, min(1.0, probability))
        
        # Convert to risk score (0-100)
        risk_score = probability * 100
        
        # Determine risk category
        if probability < 0.3:
            risk_category = "low"
        elif probability < 0.6:
            risk_category = "moderate"
        else:
            risk_category = "elevated"
        
        # Identify contributing factors
        contributing_factors = self._identify_contributing_factors(
            responses, features_scaled[0]
        )
        
        return {
            "risk_score": round(risk_score, 1),
            "probability": round(probability, 4),
            "risk_category": risk_category,
            "contributing_factors": contributing_factors
        }
    
    def _identify_contributing_factors(
        self, 
        responses: Dict[str, Any],
        scaled_features: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Identify which factors contribute most to the risk score"""
        contributions = []
        
        for i, feature_name in enumerate(FEATURE_ORDER):
            # Calculate contribution based on scaled feature and importance
            feature_value = scaled_features[i]
            importance = FEATURE_IMPORTANCE.get(feature_name, 0.1)
            contribution = feature_value * importance
            
            # Only include significant contributions
            if contribution > 0.01:
                contributions.append({
                    "feature": feature_name,
                    "label": FEATURE_LABELS.get(feature_name, feature_name),
                    "value": round(float(feature_value), 2),
                    "contribution": round(float(contribution), 3),
                    "importance": importance
                })
        
        # Sort by contribution (highest first)
        contributions.sort(key=lambda x: x["contribution"], reverse=True)
        
        # Return top 5 contributing factors
        return contributions[:5]


# Singleton instance
ml_service = MLService()


def get_ml_service() -> MLService:
    """Get the ML service instance"""
    return ml_service
