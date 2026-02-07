"""
MirAI Backend - Stage-3 ML Inference Service
Loads Stage-3 biomarker model artifacts and performs risk prediction
Artifacts loaded from: backend/models/ (relative path for deployment)
"""
import json
import pickle
import joblib
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import xgboost as xgb

# Path to Stage-3 model artifacts (relative to this file)
STAGE3_MODELS_DIR = Path(__file__).parent / "models"

# Feature order expected by Stage-3 model (5 features found in artifact)
# ['Stage2_Prob', 'pT217_F', 'AB42_F', 'AB40_F', 'NfL_Q']
STAGE3_FEATURE_ORDER = [
    "stage2_probability",       # From previous stage
    "ptau217",                  # Plasma pTau-217 (pg/mL)
    "ab42",                     # Amyloid-beta 42 (Not in UI, imperting)
    "ab40",                     # Amyloid-beta 40 (Not in UI, imputing)
    "nfl"                       # Neurofilament Light (pg/mL)
]

# Human-readable feature labels
STAGE3_FEATURE_LABELS = {
    "stage2_probability": "Stage-2 Genetic Risk",
    "ptau217": "Plasma pTau-217",
    "ab42": "Amyloid-beta 42",
    "ab40": "Amyloid-beta 40",
    "nfl": "Neurofilament Light (NfL)"
}


class Stage3MLService:
    """ML inference service for Stage-3 biomarker risk assessment"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.imputer = None
        self.is_loaded = False
        self.model_path = STAGE3_MODELS_DIR
    
    def load_artifacts(self) -> bool:
        """Load Stage-3 model artifacts from disk"""
        try:
            # Try loading XGBoost model from JSON first, then joblib
            model_json_path = self.model_path / "stage3_model.json"
            model_joblib_path = self.model_path / "stage3_model_joblib.pkl"
            
            if model_json_path.exists():
                self.model = xgb.Booster()
                self.model.load_model(str(model_json_path))
                print(f"  Loaded XGBoost model from JSON: {model_json_path.name}")
            elif model_joblib_path.exists():
                self.model = joblib.load(model_joblib_path)
                print(f"  Loaded model from joblib: {model_joblib_path.name}")
            else:
                # Try to find any .pkl or .json model file
                for f in self.model_path.glob("*model*"):
                    if f.suffix == '.json':
                        self.model = xgb.Booster()
                        self.model.load_model(str(f))
                        print(f"  Loaded model from: {f.name}")
                        break
                    elif f.suffix == '.pkl':
                        self.model = joblib.load(f)
                        print(f"  Loaded model from: {f.name}")
                        break
            
            if self.model is None:
                raise FileNotFoundError("No Stage-3 model file found")
            
            # Load scaler
            scaler_path = self.model_path / "stage3_scaler.pkl"
            if scaler_path.exists():
                try:
                    with open(scaler_path, 'rb') as f:
                        self.scaler = pickle.load(f)
                except Exception:
                    self.scaler = joblib.load(scaler_path)
                print(f"  Loaded scaler: {scaler_path.name}")
            
            # Load imputer
            imputer_path = self.model_path / "stage3_imputer.pkl"
            if imputer_path.exists():
                try:
                    with open(imputer_path, 'rb') as f:
                        self.imputer = pickle.load(f)
                except Exception:
                    self.imputer = joblib.load(imputer_path)
                print(f"  Loaded imputer: {imputer_path.name}")
            
            self.is_loaded = True
            print("✓ Stage-3 ML artifacts loaded successfully")
            return True
            
        except Exception as e:
            print(f"✗ Error loading Stage-3 ML artifacts: {e}")
            import traceback
            traceback.print_exc()
            self.is_loaded = False
            return False
    
    def _prepare_features(self, inputs: Dict[str, Any]) -> np.ndarray:
        """
        Prepare feature vector from biomarker inputs.
        Handles missing values (None) which will be filled by imputer.
        """
        features = []
        
        # 1. Stage-2 Probability
        s2_prob = inputs.get("stage2_probability")
        features.append(float(s2_prob) if s2_prob is not None else np.nan)
        
        # 2. pTau-217
        ptau = inputs.get("ptau217")
        features.append(float(ptau) if ptau is not None else np.nan)
        
        # 3. Ab42 (Not in UI, so likely NaN unless passed)
        ab42 = inputs.get("ab42")
        features.append(float(ab42) if ab42 is not None else np.nan)
        
        # 4. Ab40 (Not in UI, so likely NaN unless passed)
        ab40 = inputs.get("ab40")
        features.append(float(ab40) if ab40 is not None else np.nan)
        
        # 5. NfL
        nfl = inputs.get("nfl")
        features.append(float(nfl) if nfl is not None else np.nan)
        
        return np.array(features).reshape(1, -1)
    
    def predict_risk(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run Stage-3 biomarker risk prediction.
        
        Args:
            inputs: Dict satisfying the model features. 
                   Must include: stage2_probability, ptau217, nfl
                   
        Returns:
            Dict with probability, risk_score, risk_category, biomarker_levels
        """
        if not self.is_loaded:
            raise RuntimeError("Stage-3 ML artifacts not loaded")
        
        # Prepare features
        features = self._prepare_features(inputs)
        
        # Apply imputer (handle missing values)
        if self.imputer is not None:
            features_imputed = self.imputer.transform(features)
        else:
            features_imputed = np.nan_to_num(features, nan=0.0)
        
        # Apply scaler
        if self.scaler is not None:
            features_scaled = self.scaler.transform(features_imputed)
        else:
            features_scaled = features_imputed
        
        # Run inference
        if isinstance(self.model, xgb.Booster):
            dmatrix = xgb.DMatrix(features_scaled)
            probability = float(self.model.predict(dmatrix)[0])
        else:
            # sklearn-style model
            if hasattr(self.model, 'predict_proba'):
                probability = float(self.model.predict_proba(features_scaled)[0][1])
            else:
                probability = float(self.model.predict(features_scaled)[0])
        
        # Ensure probability is in valid range
        probability = max(0.0, min(1.0, probability))
        
        # Convert to risk score (0-100)
        risk_score = probability * 100
        
        # Determine risk category
        if probability < 0.33:
            risk_category = "low"
        elif probability < 0.66:
            risk_category = "moderate"
        else:
            risk_category = "high"
        
        # Build biomarker levels for display
        biomarker_levels = []
        # We only show what the user conceptually knows about + Stage2
        
        # pTau-217
        val = inputs.get("ptau217")
        biomarker_levels.append({
            "key": "ptau217",
            "label": "Plasma pTau-217",
            "value": val if val is not None else "Not provided",
            "provided": val is not None
        })
        
        # NfL
        val = inputs.get("nfl")
        biomarker_levels.append({
            "key": "nfl",
            "label": "Neurofilament Light",
            "value": val if val is not None else "Not provided",
            "provided": val is not None
        })
        
        return {
            "risk_score": round(risk_score, 1),
            "probability": round(probability, 4),
            "risk_category": risk_category,
            "biomarker_levels": biomarker_levels
        }


# Singleton instance
_stage3_ml_service = Stage3MLService()


def get_stage3_ml_service() -> Stage3MLService:
    """Get the Stage-3 ML service instance"""
    return _stage3_ml_service
