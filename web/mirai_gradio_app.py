"""
MirAI - Early Alzheimer's Disease Risk Assessment Platform
Consolidated Gradio Application
Single-file reconstruction of all ML services, authentication, and multi-stage workflow
"""
import gradio as gr
import json
import pickle
import joblib
import numpy as np
import bcrypt
import secrets
import sqlite3
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import xgboost as xgb
from contextlib import contextmanager

# ============================================================================
# CONFIGURATION
# ============================================================================

MODELS_DIR = Path(__file__).parent / "backend" / "models"
DATABASE_PATH = Path(__file__).parent / "mirai_gradio.db"
SESSION_DURATION_HOURS = 24


# ============================================================================
# DATABASE LAYER
# ============================================================================

def init_database():
    """Initialize SQLite database with all required tables"""
    conn = sqlite3.connect(str(DATABASE_PATH))
    cursor = conn.cursor()
    
    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Sessions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            token TEXT UNIQUE NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            expires_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    
    # Assessments table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS assessments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            stage INTEGER DEFAULT 1,
            status TEXT DEFAULT 'in_progress',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            completed_at TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    
    # Question responses table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS question_responses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            assessment_id INTEGER NOT NULL,
            question_key TEXT NOT NULL,
            answer_value TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (assessment_id) REFERENCES assessments(id)
        )
    ''')
    
    # Risk scores table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS risk_scores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            assessment_id INTEGER NOT NULL,
            stage INTEGER DEFAULT 1,
            risk_score REAL NOT NULL,
            probability REAL NOT NULL,
            risk_category TEXT NOT NULL,
            contributing_factors TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (assessment_id) REFERENCES assessments(id)
        )
    ''')
    
    # Stage-2 data table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS stage2_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            assessment_id INTEGER NOT NULL,
            apoe_e4_count INTEGER NOT NULL,
            polygenic_risk_score REAL,
            stage2_probability REAL,
            stage2_risk_category TEXT,
            combined_probability REAL,
            combined_risk_category TEXT,
            consent_given INTEGER DEFAULT 1,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (assessment_id) REFERENCES assessments(id)
        )
    ''')
    
    # Stage-3 data table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS stage3_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            assessment_id INTEGER NOT NULL,
            ptau217 REAL,
            ptau217_abeta42_ratio REAL,
            nfl REAL,
            stage3_probability REAL,
            stage3_risk_category TEXT,
            final_probability REAL,
            final_risk_category TEXT,
            final_recommendation TEXT,
            consent_given INTEGER DEFAULT 1,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (assessment_id) REFERENCES assessments(id)
        )
    ''')
    
    conn.commit()
    conn.close()

@contextmanager
def get_db():
    """Context manager for database connections"""
    conn = sqlite3.connect(str(DATABASE_PATH))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


# ============================================================================
# AUTHENTICATION SERVICE
# ============================================================================

def hash_password(password: str) -> str:
    """Hash password using bcrypt"""
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

def verify_password(password: str, hashed: str) -> bool:
    """Verify password against hash"""
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def create_session_token() -> str:
    """Generate secure session token"""
    return secrets.token_urlsafe(32)

def register_user(username: str, password: str) -> tuple:
    """Register a new user"""
    if len(username) < 3:
        return False, "Username must be at least 3 characters", None
    if len(password) < 6:
        return False, "Password must be at least 6 characters", None
    
    with get_db() as conn:
        cursor = conn.cursor()
        
        # Check if username exists
        cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
        if cursor.fetchone():
            return False, "Username already exists", None
        
        # Create user
        password_hash = hash_password(password)
        cursor.execute(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            (username, password_hash)
        )
        user_id = cursor.lastrowid
        
        # Create session
        token = create_session_token()
        expires_at = (datetime.utcnow() + timedelta(hours=SESSION_DURATION_HOURS)).isoformat()
        cursor.execute(
            "INSERT INTO user_sessions (user_id, token, expires_at) VALUES (?, ?, ?)",
            (user_id, token, expires_at)
        )
        
        conn.commit()
        return True, "Registration successful", {"token": token, "user_id": user_id, "username": username}

def login_user(username: str, password: str) -> tuple:
    """Login user and create session"""
    with get_db() as conn:
        cursor = conn.cursor()
        
        cursor.execute("SELECT id, password_hash FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()
        
        if not user:
            return False, "Invalid username or password", None
        
        if not verify_password(password, user["password_hash"]):
            return False, "Invalid username or password", None
        
        # Create new session
        token = create_session_token()
        expires_at = (datetime.utcnow() + timedelta(hours=SESSION_DURATION_HOURS)).isoformat()
        cursor.execute(
            "INSERT INTO user_sessions (user_id, token, expires_at) VALUES (?, ?, ?)",
            (user["id"], token, expires_at)
        )
        
        conn.commit()
        return True, "Login successful", {"token": token, "user_id": user["id"], "username": username}

def validate_session(token: str) -> Optional[dict]:
    """Validate session token and return user info"""
    if not token:
        return None
    
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT u.id, u.username, s.expires_at 
            FROM user_sessions s 
            JOIN users u ON s.user_id = u.id 
            WHERE s.token = ?
        ''', (token,))
        result = cursor.fetchone()
        
        if result and datetime.fromisoformat(result["expires_at"]) > datetime.utcnow():
            return {"user_id": result["id"], "username": result["username"]}
        return None

def logout_user(token: str) -> bool:
    """Logout user and invalidate session"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM user_sessions WHERE token = ?", (token,))
        conn.commit()
        return True


# ============================================================================
# STAGE-1 ML SERVICE
# ============================================================================

FEATURE_ORDER = [
    "cognitive_score", "functional_score", "age_normalized",
    "family_risk_score", "cardiovascular_score", "lifestyle_score"
]

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

FEATURE_IMPORTANCE = {
    "cognitive_score": 0.30,
    "functional_score": 0.15,
    "age_normalized": 0.20,
    "family_risk_score": 0.15,
    "cardiovascular_score": 0.12,
    "lifestyle_score": 0.08
}

FEATURE_LABELS = {
    "cognitive_score": "Cognitive Function Indicators",
    "functional_score": "Daily Living Function",
    "age_normalized": "Age Factor",
    "family_risk_score": "Family History Risk",
    "cardiovascular_score": "Cardiovascular Health",
    "lifestyle_score": "Lifestyle Factors"
}

class Stage1MLService:
    """Stage-1 ML inference service using XGBoost"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.imputer = None
        self.is_loaded = False
    
    def load_artifacts(self) -> bool:
        """Load XGBoost model, scaler, and imputer from disk"""
        try:
            # Load XGBoost model
            model_path = MODELS_DIR / "stage1_model.json"
            self.model = xgb.Booster()
            self.model.load_model(str(model_path))
            
            # Load scaler
            scaler_path = MODELS_DIR / "stage1_scaler.pkl"
            try:
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
            except Exception:
                self.scaler = joblib.load(scaler_path)
            
            # Load imputer
            imputer_path = MODELS_DIR / "stage1_imputer.pkl"
            try:
                with open(imputer_path, 'rb') as f:
                    self.imputer = pickle.load(f)
            except Exception:
                self.imputer = joblib.load(imputer_path)
            
            self.is_loaded = True
            print("✓ Stage-1 ML artifacts loaded successfully")
            return True
            
        except Exception as e:
            print(f"✗ Error loading Stage-1 ML artifacts: {e}")
            self.is_loaded = False
            return False
    
    def _aggregate_to_model_features(self, responses: Dict[str, Any]) -> np.ndarray:
        """Aggregate questionnaire responses into the 6 model features"""
        features = []
        
        # Cognitive score
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
        
        # Functional score
        functional_questions = QUESTION_TO_FEATURE_MAP["functional_score"]
        functional_values = []
        for q in functional_questions:
            if q in responses:
                try:
                    val = float(responses[q])
                    if q == "adl_assistance":
                        val = val / 2.0
                    functional_values.append(val)
                except (ValueError, TypeError):
                    pass
        functional_score = np.mean(functional_values) if functional_values else 0.0
        features.append(functional_score)
        
        # Age normalized
        age = float(responses.get("age", 65))
        age_normalized = (age - 40) / 60.0
        age_normalized = max(0.0, min(1.0, age_normalized))
        features.append(age_normalized)
        
        # Family risk score
        family_history = float(responses.get("family_history_dementia", 0))
        family_degree = float(responses.get("family_relation_degree", 0)) / 2.0
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
        
        # Lifestyle score
        lifestyle_values = []
        if "physical_activity" in responses:
            pa = float(responses["physical_activity"])
            lifestyle_values.append(1.0 - (pa / 3.0))
        if "smoking_status" in responses:
            smoke = float(responses["smoking_status"]) / 2.0
            lifestyle_values.append(smoke)
        if "sleep_hours" in responses:
            sleep = float(responses["sleep_hours"])
            sleep_risk = abs(sleep - 7.5) / 5.0
            lifestyle_values.append(min(1.0, sleep_risk))
        if "depression" in responses:
            lifestyle_values.append(float(responses["depression"]))
        if "head_injury" in responses:
            lifestyle_values.append(float(responses["head_injury"]))
        
        lifestyle_score = np.mean(lifestyle_values) if lifestyle_values else 0.0
        features.append(lifestyle_score)
        
        return np.array(features).reshape(1, -1)
    
    def predict_risk(self, responses: Dict[str, Any]) -> Dict[str, Any]:
        """Run full Stage-1 inference pipeline"""
        if not self.is_loaded:
            raise RuntimeError("Stage-1 ML artifacts not loaded")
        
        features = self._aggregate_to_model_features(responses)
        features_imputed = self.imputer.transform(features)
        features_scaled = self.scaler.transform(features_imputed)
        
        dmatrix = xgb.DMatrix(features_scaled)
        probability = float(self.model.predict(dmatrix)[0])
        probability = max(0.0, min(1.0, probability))
        
        risk_score = probability * 100
        
        if probability < 0.3:
            risk_category = "low"
        elif probability < 0.6:
            risk_category = "moderate"
        else:
            risk_category = "elevated"
        
        # Contributing factors
        contributions = []
        for i, feature_name in enumerate(FEATURE_ORDER):
            feature_value = features_scaled[0][i]
            importance = FEATURE_IMPORTANCE.get(feature_name, 0.1)
            contribution = feature_value * importance
            if contribution > 0.01:
                contributions.append({
                    "feature": feature_name,
                    "label": FEATURE_LABELS.get(feature_name, feature_name),
                    "value": round(float(feature_value), 2),
                    "contribution": round(float(contribution), 3),
                    "importance": importance
                })
        contributions.sort(key=lambda x: x["contribution"], reverse=True)
        
        return {
            "risk_score": round(risk_score, 1),
            "probability": round(probability, 4),
            "risk_category": risk_category,
            "contributing_factors": contributions[:5]
        }


# ============================================================================
# STAGE-2 ML SERVICE
# ============================================================================

STAGE2_FEATURES = ["APOE_e4_count", "polygenic_risk_score"]

class Stage2MLService:
    """Stage-2 ML inference service for genetic risk"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.imputer = None
        self.is_loaded = False
    
    def load_artifacts(self) -> bool:
        """Load Stage-2 model artifacts"""
        try:
            model_path = MODELS_DIR / "stage2_model.json"
            if not model_path.exists():
                print(f"⚠ Stage-2 model not found at {model_path}")
                return False
            
            self.model = xgb.Booster()
            self.model.load_model(str(model_path))
            
            scaler_path = MODELS_DIR / "stage2_scaler.pkl"
            try:
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
            except Exception:
                self.scaler = joblib.load(scaler_path)
            
            imputer_path = MODELS_DIR / "stage2_imputer.pkl"
            try:
                with open(imputer_path, 'rb') as f:
                    self.imputer = pickle.load(f)
            except Exception:
                self.imputer = joblib.load(imputer_path)
            
            self.is_loaded = True
            print("✓ Stage-2 ML artifacts loaded successfully")
            return True
            
        except Exception as e:
            print(f"✗ Error loading Stage-2 ML artifacts: {e}")
            self.is_loaded = False
            return False
    
    def prepare_features(self, inputs: Dict[str, Any]) -> np.ndarray:
        """Convert genetic inputs to feature vector"""
        features = []
        for feature_name in STAGE2_FEATURES:
            if feature_name in inputs and inputs[feature_name] is not None:
                try:
                    features.append(float(inputs[feature_name]))
                except (ValueError, TypeError):
                    features.append(np.nan)
            else:
                features.append(np.nan)
        return np.array(features).reshape(1, -1)
    
    def predict_risk(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Run Stage-2 genetic risk inference"""
        if not self.is_loaded:
            raise RuntimeError("Stage-2 ML artifacts not loaded")
        
        features = self.prepare_features(inputs)
        features_imputed = self.imputer.transform(features)
        features_scaled = self.scaler.transform(features_imputed)
        
        dmatrix = xgb.DMatrix(features_scaled)
        probability = float(self.model.predict(dmatrix)[0])
        probability = max(0.0, min(1.0, probability))
        
        if probability < 0.33:
            risk_category = "low"
        elif probability < 0.66:
            risk_category = "moderate"
        else:
            risk_category = "high"
        
        apoe_count = inputs.get("APOE_e4_count", 0)
        prs = inputs.get("polygenic_risk_score")
        
        feature_analysis = [{
            "feature": "APOE_e4_count",
            "label": "APOE ε4 Allele Count",
            "value": apoe_count,
            "description": f"{apoe_count} cop{'y' if apoe_count == 1 else 'ies'} of ε4 allele",
            "risk_level": "low" if apoe_count == 0 else ("moderate" if apoe_count == 1 else "high")
        }]
        
        if prs is not None:
            feature_analysis.append({
                "feature": "polygenic_risk_score",
                "label": "Polygenic Risk Score",
                "value": round(prs, 3),
                "description": f"PRS: {prs:.3f}",
                "risk_level": "low" if prs < 0.3 else ("moderate" if prs < 0.7 else "high")
            })
        
        return {
            "probability": round(probability, 4),
            "risk_score": round(probability * 100, 1),
            "risk_category": risk_category,
            "feature_analysis": feature_analysis
        }


# ============================================================================
# STAGE-3 ML SERVICE
# ============================================================================

STAGE3_FEATURE_ORDER = ["stage2_probability", "ptau217", "ab42", "ab40", "nfl"]

class Stage3MLService:
    """Stage-3 ML inference service for biomarker risk"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.imputer = None
        self.is_loaded = False
    
    def load_artifacts(self) -> bool:
        """Load Stage-3 model artifacts"""
        try:
            model_json_path = MODELS_DIR / "stage3_model.json"
            model_joblib_path = MODELS_DIR / "stage3_model_joblib.pkl"
            
            if model_json_path.exists():
                self.model = xgb.Booster()
                self.model.load_model(str(model_json_path))
            elif model_joblib_path.exists():
                self.model = joblib.load(model_joblib_path)
            else:
                raise FileNotFoundError("No Stage-3 model file found")
            
            scaler_path = MODELS_DIR / "stage3_scaler.pkl"
            if scaler_path.exists():
                try:
                    with open(scaler_path, 'rb') as f:
                        self.scaler = pickle.load(f)
                except Exception:
                    self.scaler = joblib.load(scaler_path)
            
            imputer_path = MODELS_DIR / "stage3_imputer.pkl"
            if imputer_path.exists():
                try:
                    with open(imputer_path, 'rb') as f:
                        self.imputer = pickle.load(f)
                except Exception:
                    self.imputer = joblib.load(imputer_path)
            
            self.is_loaded = True
            print("✓ Stage-3 ML artifacts loaded successfully")
            return True
            
        except Exception as e:
            print(f"✗ Error loading Stage-3 ML artifacts: {e}")
            self.is_loaded = False
            return False
    
    def _prepare_features(self, inputs: Dict[str, Any]) -> np.ndarray:
        """Prepare feature vector from biomarker inputs"""
        features = []
        
        s2_prob = inputs.get("stage2_probability")
        features.append(float(s2_prob) if s2_prob is not None else np.nan)
        
        ptau = inputs.get("ptau217")
        features.append(float(ptau) if ptau is not None else np.nan)
        
        ab42 = inputs.get("ab42")
        features.append(float(ab42) if ab42 is not None else np.nan)
        
        ab40 = inputs.get("ab40")
        features.append(float(ab40) if ab40 is not None else np.nan)
        
        nfl = inputs.get("nfl")
        features.append(float(nfl) if nfl is not None else np.nan)
        
        return np.array(features).reshape(1, -1)
    
    def predict_risk(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Run Stage-3 biomarker risk prediction"""
        if not self.is_loaded:
            raise RuntimeError("Stage-3 ML artifacts not loaded")
        
        features = self._prepare_features(inputs)
        
        if self.imputer is not None:
            features_imputed = self.imputer.transform(features)
        else:
            features_imputed = np.nan_to_num(features, nan=0.0)
        
        if self.scaler is not None:
            features_scaled = self.scaler.transform(features_imputed)
        else:
            features_scaled = features_imputed
        
        if isinstance(self.model, xgb.Booster):
            dmatrix = xgb.DMatrix(features_scaled)
            probability = float(self.model.predict(dmatrix)[0])
        else:
            if hasattr(self.model, 'predict_proba'):
                probability = float(self.model.predict_proba(features_scaled)[0][1])
            else:
                probability = float(self.model.predict(features_scaled)[0])
        
        probability = max(0.0, min(1.0, probability))
        risk_score = probability * 100
        
        if probability < 0.33:
            risk_category = "low"
        elif probability < 0.66:
            risk_category = "moderate"
        else:
            risk_category = "high"
        
        biomarker_levels = [
            {"key": "ptau217", "label": "Plasma pTau-217", "value": inputs.get("ptau217"), "provided": inputs.get("ptau217") is not None},
            {"key": "nfl", "label": "Neurofilament Light", "value": inputs.get("nfl"), "provided": inputs.get("nfl") is not None}
        ]
        
        return {
            "risk_score": round(risk_score, 1),
            "probability": round(probability, 4),
            "risk_category": risk_category,
            "biomarker_levels": biomarker_levels
        }


# ============================================================================
# RISK CALCULATION HELPERS
# ============================================================================

def calculate_combined_risk(stage1_prob: float, stage2_prob: float,
                           stage1_weight: float = 0.6, stage2_weight: float = 0.4) -> dict:
    """Calculate combined Stage-1 + Stage-2 risk"""
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

def calculate_final_risk(stage1_prob: float, stage2_prob: float, stage3_prob: float) -> dict:
    """Calculate final combined risk from all three stages"""
    STAGE1_WEIGHT = 0.25
    STAGE2_WEIGHT = 0.35
    STAGE3_WEIGHT = 0.40
    
    final_prob = (STAGE1_WEIGHT * stage1_prob) + (STAGE2_WEIGHT * stage2_prob) + (STAGE3_WEIGHT * stage3_prob)
    final_prob = max(0.0, min(1.0, final_prob))
    
    if final_prob < 0.33:
        category = "low"
        recommendation = "Routine monitoring recommended. Continue healthy lifestyle practices."
    elif final_prob < 0.66:
        category = "moderate"
        recommendation = "Annual biomarker testing recommended. Consider cognitive assessments."
    else:
        category = "high"
        recommendation = "Neurologist consultation recommended. Advanced neuroimaging advised."
    
    return {
        "probability": round(final_prob, 4),
        "risk_score": round(final_prob * 100, 1),
        "risk_category": category,
        "recommendation": recommendation,
        "weights": {"stage1": STAGE1_WEIGHT, "stage2": STAGE2_WEIGHT, "stage3": STAGE3_WEIGHT}
    }


# ============================================================================
# STAGE-1 QUESTIONS DEFINITION
# ============================================================================

STAGE1_QUESTIONS = [
    {"section": "Demographics & Basic Information", "section_id": 1, "questions": [
        {"key": "age", "text": "What is your age?", "type": "number", "min": 1, "max": 100},
        {"key": "sex", "text": "What is your biological sex?", "type": "select", "options": [("Male", "1"), ("Female", "0")]},
        {"key": "education_years", "text": "How many years of formal education have you completed?", "type": "number", "min": 0, "max": 25},
        {"key": "employment_status", "text": "What is your current employment status?", "type": "select", 
         "options": [("Currently working", "0"), ("Retired", "1"), ("Not currently employed", "2")]}
    ]},
    {"section": "Family History", "section_id": 2, "questions": [
        {"key": "family_history_dementia", "text": "Do you have a family history of Alzheimer's disease or dementia?", "type": "select",
         "options": [("Yes", "1"), ("No", "0"), ("Not sure", "-1")]},
        {"key": "family_relation_degree", "text": "If yes, how closely related was this family member?", "type": "select",
         "options": [("First-degree relative", "2"), ("Second-degree relative", "1"), ("Not applicable", "0")]}
    ]},
    {"section": "Memory & Daily Changes", "section_id": 3, "questions": [
        {"key": "judgment_problems", "text": "Have you noticed problems making decisions or poor judgment?", "type": "select", "options": [("Yes", "1"), ("No", "0")]},
        {"key": "reduced_interest", "text": "Have you lost interest in hobbies or activities you used to enjoy?", "type": "select", "options": [("Yes", "1"), ("No", "0")]},
        {"key": "repetition_issues", "text": "Do you find yourself repeating the same questions or stories?", "type": "select", "options": [("Yes", "1"), ("No", "0")]},
        {"key": "learning_difficulty", "text": "Do you have trouble learning how to use new technology?", "type": "select", "options": [("Yes", "1"), ("No", "0")]},
        {"key": "temporal_disorientation", "text": "Do you sometimes forget the current month, year, or day?", "type": "select", "options": [("Yes", "1"), ("No", "0")]},
        {"key": "financial_difficulty", "text": "Do you have trouble handling complex financial tasks?", "type": "select", "options": [("Yes", "1"), ("No", "0")]},
        {"key": "appointment_memory", "text": "Do you frequently forget appointments or scheduled events?", "type": "select", "options": [("Yes", "1"), ("No", "0")]},
        {"key": "daily_memory_problems", "text": "Do you experience daily problems with thinking and/or memory?", "type": "select", "options": [("Yes", "1"), ("No", "0")]}
    ]},
    {"section": "Daily Living Activities", "section_id": 4, "questions": [
        {"key": "adl_assistance", "text": "Do you need help with basic daily activities?", "type": "select",
         "options": [("No, I am fully independent", "0"), ("Sometimes need assistance", "1"), ("Frequently need assistance", "2")]},
        {"key": "navigation_difficulty", "text": "Do you have difficulty finding your way in familiar places?", "type": "select", "options": [("Yes", "1"), ("No", "0")]}
    ]},
    {"section": "Health & Lifestyle Factors", "section_id": 5, "questions": [
        {"key": "diabetes", "text": "Have you been diagnosed with diabetes?", "type": "select", "options": [("Yes", "1"), ("No", "0")]},
        {"key": "hypertension", "text": "Have you been diagnosed with high blood pressure?", "type": "select", "options": [("Yes", "1"), ("No", "0")]},
        {"key": "stroke_history", "text": "Have you ever had a stroke or mini-stroke (TIA)?", "type": "select", "options": [("Yes", "1"), ("No", "0")]},
        {"key": "physical_activity", "text": "How would you describe your typical physical activity level?", "type": "select",
         "options": [("Sedentary", "0"), ("Light", "1"), ("Moderate", "2"), ("Active", "3")]},
        {"key": "smoking_status", "text": "What is your smoking history?", "type": "select",
         "options": [("Never smoked", "0"), ("Former smoker", "1"), ("Current smoker", "2")]},
        {"key": "sleep_hours", "text": "On average, how many hours of sleep do you get per night?", "type": "number", "min": 2, "max": 14},
        {"key": "depression", "text": "Have you been diagnosed with depression?", "type": "select", "options": [("Yes", "1"), ("No", "0")]},
        {"key": "head_injury", "text": "Have you ever had a significant head injury?", "type": "select", "options": [("Yes", "1"), ("No", "0")]},
        {"key": "high_cholesterol", "text": "Do you have high cholesterol?", "type": "select", "options": [("Yes", "1"), ("No", "0")]}
    ]}
]


# ============================================================================
# GLOBAL STATE & ML SERVICE INSTANCES
# ============================================================================

# Initialize database
init_database()

# Initialize ML services
stage1_ml = Stage1MLService()
stage2_ml = Stage2MLService()
stage3_ml = Stage3MLService()

# Load artifacts
print("=" * 50)
print("MirAI Gradio Application Starting...")
print("=" * 50)
stage1_ml.load_artifacts()
stage2_ml.load_artifacts()
stage3_ml.load_artifacts()
print("=" * 50)


# ============================================================================
# GRADIO UI HANDLERS
# ============================================================================

def handle_login(username: str, password: str, state: dict):
    """Handle login button click"""
    success, message, data = login_user(username, password)
    if success:
        state["token"] = data["token"]
        state["user_id"] = data["user_id"]
        state["username"] = data["username"]
        return message, state, gr.update(visible=False), gr.update(visible=True)
    return message, state, gr.update(visible=True), gr.update(visible=False)

def handle_register(username: str, password: str, state: dict):
    """Handle register button click"""
    success, message, data = register_user(username, password)
    if success:
        state["token"] = data["token"]
        state["user_id"] = data["user_id"]
        state["username"] = data["username"]
        return message, state, gr.update(visible=False), gr.update(visible=True)
    return message, state, gr.update(visible=True), gr.update(visible=False)

def handle_logout(state: dict):
    """Handle logout"""
    if state.get("token"):
        logout_user(state["token"])
    state = {}
    return state, gr.update(visible=True), gr.update(visible=False)

def start_assessment(state: dict):
    """Start a new Stage-1 assessment"""
    if not state.get("token"):
        return "Please log in first.", state
    
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO assessments (user_id, stage, status) VALUES (?, 1, 'in_progress')",
            (state["user_id"],)
        )
        assessment_id = cursor.lastrowid
        conn.commit()
    
    state["assessment_id"] = assessment_id
    state["answers"] = {}
    return f"Assessment #{assessment_id} started. Please answer the questions.", state

def submit_stage1(state: dict, *answers):
    """Process Stage-1 submission"""
    if not state.get("assessment_id"):
        return "No active assessment. Please start a new assessment.", "", state
    
    # Collect all answers
    responses = {}
    answer_idx = 0
    for section in STAGE1_QUESTIONS:
        for q in section["questions"]:
            if answer_idx < len(answers):
                val = answers[answer_idx]
                if val is not None and val != "":
                    responses[q["key"]] = val
            answer_idx += 1
    
    state["answers"] = responses
    
    # Run ML inference
    try:
        result = stage1_ml.predict_risk(responses)
    except Exception as e:
        return f"Error running analysis: {e}", "", state
    
    # Store in database
    with get_db() as conn:
        cursor = conn.cursor()
        
        # Store responses
        for key, value in responses.items():
            cursor.execute(
                "INSERT INTO question_responses (assessment_id, question_key, answer_value) VALUES (?, ?, ?)",
                (state["assessment_id"], key, str(value))
            )
        
        # Store risk score
        cursor.execute('''
            INSERT INTO risk_scores (assessment_id, stage, risk_score, probability, risk_category, contributing_factors)
            VALUES (?, 1, ?, ?, ?, ?)
        ''', (state["assessment_id"], result["risk_score"], result["probability"], 
              result["risk_category"], json.dumps(result["contributing_factors"])))
        
        # Update assessment status
        cursor.execute(
            "UPDATE assessments SET status = 'completed', completed_at = ? WHERE id = ?",
            (datetime.utcnow().isoformat(), state["assessment_id"])
        )
        
        conn.commit()
    
    state["stage1_result"] = result
    state["stage1_probability"] = result["probability"]
    
    # Format result
    factors_text = "\n".join([f"  • {f['label']}: {f['contribution']:.3f}" for f in result["contributing_factors"]])
    result_text = f"""
## Stage-1 Results

**Risk Score:** {result['risk_score']:.1f}/100
**Risk Category:** {result['risk_category'].upper()}
**Probability:** {result['probability']:.4f}

### Contributing Factors:
{factors_text}

### Recommendation:
{"Your screening indicates a lower risk profile. Continue maintaining healthy habits." if result['risk_category'] == "low" else "Your screening suggests some factors that warrant attention. Consider discussing with your healthcare provider." if result['risk_category'] == "moderate" else "Your screening indicates elevated risk factors. We recommend scheduling a consultation with a healthcare professional."}
"""
    
    return "Stage-1 complete! Continue to Stage-2 for genetic risk assessment.", result_text, state

def submit_stage2(state: dict, apoe_count: int, prs: float):
    """Process Stage-2 submission"""
    if not state.get("stage1_probability"):
        return "Please complete Stage-1 first.", "", state
    
    # Run ML inference
    inputs = {
        "APOE_e4_count": int(apoe_count),
        "polygenic_risk_score": prs if prs else None
    }
    
    try:
        result = stage2_ml.predict_risk(inputs)
    except Exception as e:
        return f"Error running genetic analysis: {e}", "", state
    
    # Calculate combined risk
    combined = calculate_combined_risk(state["stage1_probability"], result["probability"])
    
    # Store in database
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO stage2_data (assessment_id, apoe_e4_count, polygenic_risk_score, 
                                     stage2_probability, stage2_risk_category, 
                                     combined_probability, combined_risk_category)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (state["assessment_id"], int(apoe_count), prs,
              result["probability"], result["risk_category"],
              combined["probability"], combined["risk_category"]))
        conn.commit()
    
    state["stage2_result"] = result
    state["stage2_probability"] = result["probability"]
    state["combined_result"] = combined
    
    # Format result
    features_text = "\n".join([f"  • {f['label']}: {f['value']} ({f['risk_level']} risk)" for f in result["feature_analysis"]])
    result_text = f"""
## Stage-2 Genetic Risk Results

**Genetic Risk Score:** {result['risk_score']:.1f}/100
**Risk Category:** {result['risk_category'].upper()}

### Genetic Factors:
{features_text}

---

## Combined Risk (Stage-1 + Stage-2)

**Combined Risk Score:** {combined['risk_score']:.1f}/100
**Combined Category:** {combined['risk_category'].upper()}

### Next Steps:
{"Consider Stage-3 biomarker screening for a comprehensive assessment." if combined['risk_category'] in ['moderate', 'high'] else "Your combined assessment indicates lower risk. Continue healthy lifestyle practices."}
"""
    
    return "Stage-2 complete! You may proceed to Stage-3 for biomarker screening.", result_text, state

def submit_stage3(state: dict, ptau217: float, nfl: float):
    """Process Stage-3 submission"""
    if not state.get("stage2_probability"):
        return "Please complete Stage-2 first.", "", state
    
    # Run ML inference
    inputs = {
        "stage2_probability": state["stage2_probability"],
        "ptau217": ptau217 if ptau217 else None,
        "nfl": nfl if nfl else None,
        "ab42": None,
        "ab40": None
    }
    
    try:
        result = stage3_ml.predict_risk(inputs)
    except Exception as e:
        return f"Error running biomarker analysis: {e}", "", state
    
    # Calculate final risk
    final = calculate_final_risk(
        state["stage1_probability"],
        state["stage2_probability"],
        result["probability"]
    )
    
    # Store in database
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO stage3_data (assessment_id, ptau217, nfl,
                                     stage3_probability, stage3_risk_category,
                                     final_probability, final_risk_category, final_recommendation)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (state["assessment_id"], ptau217, nfl,
              result["probability"], result["risk_category"],
              final["probability"], final["risk_category"], final["recommendation"]))
        
        cursor.execute("UPDATE assessments SET stage = 3 WHERE id = ?", (state["assessment_id"],))
        conn.commit()
    
    state["stage3_result"] = result
    state["final_result"] = final
    
    # Format result
    result_text = f"""
## Stage-3 Biomarker Results

**Biomarker Risk Score:** {result['risk_score']:.1f}/100
**Risk Category:** {result['risk_category'].upper()}

### Biomarkers Analyzed:
  • pTau-217: {ptau217 if ptau217 else 'Not provided'} pg/mL
  • Neurofilament Light: {nfl if nfl else 'Not provided'} pg/mL

---

# 🎯 FINAL COMPREHENSIVE RESULTS

## All Stages Summary:
| Stage | Risk Score | Category |
|-------|------------|----------|
| Stage-1 (Clinical) | {state['stage1_result']['risk_score']:.1f} | {state['stage1_result']['risk_category'].upper()} |
| Stage-2 (Genetic) | {state['stage2_result']['risk_score']:.1f} | {state['stage2_result']['risk_category'].upper()} |
| Stage-3 (Biomarker) | {result['risk_score']:.1f} | {result['risk_category'].upper()} |

## Final Combined Assessment:
**Final Risk Score:** {final['risk_score']:.1f}/100
**Final Category:** {final['risk_category'].upper()}

### Weights Used:
  • Clinical (Stage-1): {final['weights']['stage1']*100:.0f}%
  • Genetic (Stage-2): {final['weights']['stage2']*100:.0f}%
  • Biomarker (Stage-3): {final['weights']['stage3']*100:.0f}%

### Recommendation:
{final['recommendation']}

---
⚠️ **Disclaimer:** This is a research screening tool and is NOT a diagnostic test. Results should be discussed with a healthcare professional.
"""
    
    return "Assessment complete! See your comprehensive results below.", result_text, state


# ============================================================================
# GRADIO UI DEFINITION
# ============================================================================

def create_interface():
    """Create the Gradio interface"""
    
    with gr.Blocks(title="MirAI - Alzheimer's Risk Assessment", theme=gr.themes.Soft()) as app:
        
        # State
        state = gr.State({})
        
        gr.Markdown("""
        # 🧠 MirAI - Early Alzheimer's Disease Risk Assessment
        ### Research Prototype - Multi-Stage ML Screening Platform
        
        This platform uses a 3-stage machine learning approach to assess cognitive health risk factors.
        
        ⚠️ **Disclaimer:** This is a research screening tool and is NOT a diagnostic test.
        """)
        
        # ==================== AUTH SECTION ====================
        with gr.Group(visible=True) as auth_group:
            gr.Markdown("## 🔐 Authentication")
            with gr.Row():
                with gr.Column():
                    username_input = gr.Textbox(label="Username", placeholder="Enter username")
                    password_input = gr.Textbox(label="Password", type="password", placeholder="Enter password")
                with gr.Column():
                    login_btn = gr.Button("Login", variant="primary")
                    register_btn = gr.Button("Register")
                    auth_status = gr.Textbox(label="Status", interactive=False)
        
        # ==================== MAIN APP SECTION ====================
        with gr.Group(visible=False) as main_group:
            with gr.Row():
                gr.Markdown("## Welcome!")
                logout_btn = gr.Button("Logout", size="sm")
            
            with gr.Tabs() as tabs:
                
                # -------------------- STAGE-1 TAB --------------------
                with gr.TabItem("Stage-1: Clinical Screening"):
                    gr.Markdown("""
                    ### Stage-1 Clinical Questionnaire
                    Answer the following questions about your health, lifestyle, and daily experiences.
                    """)
                    
                    start_btn = gr.Button("Start New Assessment", variant="primary")
                    start_status = gr.Textbox(label="Status", interactive=False)
                    
                    # Create input components for each question
                    stage1_inputs = []
                    for section in STAGE1_QUESTIONS:
                        gr.Markdown(f"#### {section['section']}")
                        for q in section["questions"]:
                            if q["type"] == "number":
                                inp = gr.Number(label=q["text"], minimum=q.get("min", 0), maximum=q.get("max", 100))
                            elif q["type"] == "select":
                                choices = [(opt[0], opt[1]) for opt in q["options"]]
                                inp = gr.Dropdown(label=q["text"], choices=choices, type="value")
                            stage1_inputs.append(inp)
                    
                    submit_stage1_btn = gr.Button("Submit Stage-1", variant="primary")
                    stage1_status = gr.Textbox(label="Stage-1 Status", interactive=False)
                    stage1_result = gr.Markdown(label="Stage-1 Results")
                
                # -------------------- STAGE-2 TAB --------------------
                with gr.TabItem("Stage-2: Genetic Risk"):
                    gr.Markdown("""
                    ### Stage-2 Genetic Risk Stratification
                    Enter your genetic test results for APOE genotype and Polygenic Risk Score.
                    """)
                    
                    apoe_input = gr.Slider(
                        label="APOE ε4 Allele Count",
                        info="Number of ε4 alleles (0, 1, or 2)",
                        minimum=0, maximum=2, step=1, value=0
                    )
                    prs_input = gr.Number(
                        label="Polygenic Risk Score (PRS)",
                        info="Optional: Enter your PRS value if available (typically 0-1)"
                    )
                    
                    submit_stage2_btn = gr.Button("Submit Stage-2", variant="primary")
                    stage2_status = gr.Textbox(label="Stage-2 Status", interactive=False)
                    stage2_result = gr.Markdown(label="Stage-2 Results")
                
                # -------------------- STAGE-3 TAB --------------------
                with gr.TabItem("Stage-3: Biomarker Screening"):
                    gr.Markdown("""
                    ### Stage-3 Plasma Biomarker Analysis
                    Enter your plasma biomarker test results.
                    """)
                    
                    ptau_input = gr.Number(
                        label="Plasma pTau-217 (pg/mL)",
                        info="Phosphorylated tau-217 concentration"
                    )
                    nfl_input = gr.Number(
                        label="Neurofilament Light Chain (pg/mL)",
                        info="NfL concentration"
                    )
                    
                    submit_stage3_btn = gr.Button("Submit Stage-3 & Get Final Results", variant="primary")
                    stage3_status = gr.Textbox(label="Stage-3 Status", interactive=False)
                    stage3_result = gr.Markdown(label="Final Comprehensive Results")
        
        # ==================== EVENT HANDLERS ====================
        login_btn.click(
            fn=handle_login,
            inputs=[username_input, password_input, state],
            outputs=[auth_status, state, auth_group, main_group]
        )
        
        register_btn.click(
            fn=handle_register,
            inputs=[username_input, password_input, state],
            outputs=[auth_status, state, auth_group, main_group]
        )
        
        logout_btn.click(
            fn=handle_logout,
            inputs=[state],
            outputs=[state, auth_group, main_group]
        )
        
        start_btn.click(
            fn=start_assessment,
            inputs=[state],
            outputs=[start_status, state]
        )
        
        submit_stage1_btn.click(
            fn=submit_stage1,
            inputs=[state] + stage1_inputs,
            outputs=[stage1_status, stage1_result, state]
        )
        
        submit_stage2_btn.click(
            fn=submit_stage2,
            inputs=[state, apoe_input, prs_input],
            outputs=[stage2_status, stage2_result, state]
        )
        
        submit_stage3_btn.click(
            fn=submit_stage3,
            inputs=[state, ptau_input, nfl_input],
            outputs=[stage3_status, stage3_result, state]
        )
    
    return app


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    app = create_interface()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
