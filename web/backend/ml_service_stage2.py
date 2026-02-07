"""
MirAI Backend - Stage-2 ML Inference Service
Loads Stage-2 XGBoost model for genetic risk assessment
"""
import json
import pickle
import joblib
import numpy as np
from pathlib import Path
from typing import Dict, Any
import xgboost as xgb

# Path to model artifacts
MODELS_DIR = Path(__file__).parent / "models"

# Stage-2 uses 2 features for genetic risk
STAGE2_FEATURES = [
    "APOE_e4_count",        # 0, 1, or 2 copies of APOE ε4 allele
    "polygenic_risk_score"  # Polygenic risk score (continuous)
]

# Feature labels for explanation
STAGE2_FEATURE_LABELS = {
    "APOE_e4_count": "APOE ε4 Allele Count",
    "polygenic_risk_score": "Polygenic Risk Score (PRS)"
}


class Stage2MLService:
    """Machine Learning inference service for Stage-2 genetic risk assessment"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.imputer = None
        self.is_loaded = False
    
    def load_artifacts(self):
        """Load Stage-2 XGBoost model, scaler, and imputer from disk"""
        try:
            # Load XGBoost model from JSON
            model_path = MODELS_DIR / "stage2_model.json"
            if not model_path.exists():
                print(f"⚠ Stage-2 model not found at {model_path}")
                return False
                
            self.model = xgb.Booster()
            self.model.load_model(str(model_path))
            
            # Load scaler (try pickle first, fallback to joblib)
            scaler_path = MODELS_DIR / "stage2_scaler.pkl"
            try:
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
            except Exception:
                self.scaler = joblib.load(scaler_path)
            
            # Load imputer (try pickle first, fallback to joblib)
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
            import traceback
            traceback.print_exc()
            self.is_loaded = False
            return False
    
    def prepare_features(self, inputs: Dict[str, Any]) -> np.ndarray:
        """
        Convert genetic inputs to feature vector
        
        Args:
            inputs: Dict with APOE_e4_count and polygenic_risk_score
            
        Returns:
            numpy array of features [APOE_e4_count, polygenic_risk_score]
        """
        features = []
        
        for feature_name in STAGE2_FEATURES:
            if feature_name in inputs and inputs[feature_name] is not None:
                try:
                    features.append(float(inputs[feature_name]))
                except (ValueError, TypeError):
                    features.append(np.nan)
            else:
                # Missing value - will be handled by imputer
                features.append(np.nan)
        
        return np.array(features).reshape(1, -1)
    
    def predict_risk(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run Stage-2 genetic risk inference:
        1. Convert inputs to feature vector
        2. Apply imputer for missing values
        3. Apply scaler for normalization
        4. Run XGBoost inference
        
        Returns dict with probability, category, and feature analysis
        """
        if not self.is_loaded:
            raise RuntimeError("Stage-2 ML artifacts not loaded")
        
        # Step 1: Prepare feature vector
        features = self.prepare_features(inputs)
        
        # Step 2: Apply imputer (handle missing values)
        features_imputed = self.imputer.transform(features)
        
        # Step 3: Apply scaler (normalize)
        features_scaled = self.scaler.transform(features_imputed)
        
        # Step 4: XGBoost inference
        dmatrix = xgb.DMatrix(features_scaled)
        probability = float(self.model.predict(dmatrix)[0])
        
        # Ensure probability is in valid range
        probability = max(0.0, min(1.0, probability))
        
        # Determine risk category
        if probability < 0.33:
            risk_category = "low"
        elif probability < 0.66:
            risk_category = "moderate"
        else:
            risk_category = "high"
        
        # Feature contribution analysis
        feature_analysis = []
        apoe_count = inputs.get("APOE_e4_count", 0)
        prs = inputs.get("polygenic_risk_score")
        
        # APOE ε4 contribution
        apoe_risk_level = "low" if apoe_count == 0 else ("moderate" if apoe_count == 1 else "high")
        feature_analysis.append({
            "feature": "APOE_e4_count",
            "label": "APOE ε4 Allele Count",
            "value": apoe_count,
            "description": f"{apoe_count} cop{'y' if apoe_count == 1 else 'ies'} of ε4 allele",
            "risk_level": apoe_risk_level
        })
        
        # PRS contribution
        if prs is not None:
            prs_risk_level = "low" if prs < 0.3 else ("moderate" if prs < 0.7 else "high")
            feature_analysis.append({
                "feature": "polygenic_risk_score",
                "label": "Polygenic Risk Score",
                "value": round(prs, 3),
                "description": f"PRS: {prs:.3f}",
                "risk_level": prs_risk_level
            })
        
        return {
            "probability": round(probability, 4),
            "risk_score": round(probability * 100, 1),
            "risk_category": risk_category,
            "feature_analysis": feature_analysis
        }


# Singleton instance
stage2_ml_service = Stage2MLService()


def get_stage2_ml_service() -> Stage2MLService:
    """Get the Stage-2 ML service instance"""
    return stage2_ml_service
