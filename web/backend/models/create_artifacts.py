"""
Script to create synthetic ML artifacts (scaler and imputer)
Run this once to generate the pickle files
"""
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Feature statistics based on typical population data
# These represent mean and std for each feature
FEATURE_STATS = {
    "age": {"mean": 65.0, "std": 12.0},
    "sex": {"mean": 0.48, "std": 0.50},
    "education_years": {"mean": 14.0, "std": 3.5},
    "employment_status": {"mean": 0.8, "std": 0.7},
    "family_history_dementia": {"mean": 0.25, "std": 0.43},
    "family_relation_degree": {"mean": 0.35, "std": 0.65},
    "judgment_problems": {"mean": 0.15, "std": 0.36},
    "reduced_interest": {"mean": 0.20, "std": 0.40},
    "repetition_issues": {"mean": 0.18, "std": 0.38},
    "learning_difficulty": {"mean": 0.22, "std": 0.41},
    "temporal_disorientation": {"mean": 0.12, "std": 0.32},
    "financial_difficulty": {"mean": 0.15, "std": 0.36},
    "appointment_memory": {"mean": 0.25, "std": 0.43},
    "daily_memory_problems": {"mean": 0.10, "std": 0.30},
    "adl_assistance": {"mean": 0.15, "std": 0.45},
    "navigation_difficulty": {"mean": 0.08, "std": 0.27},
    "diabetes": {"mean": 0.18, "std": 0.38},
    "hypertension": {"mean": 0.35, "std": 0.48},
    "stroke_history": {"mean": 0.05, "std": 0.22},
    "physical_activity": {"mean": 1.5, "std": 1.0},
    "smoking_status": {"mean": 0.6, "std": 0.75},
    "sleep_hours": {"mean": 7.0, "std": 1.2},
    "depression": {"mean": 0.15, "std": 0.36},
    "head_injury": {"mean": 0.10, "std": 0.30},
    "high_cholesterol": {"mean": 0.30, "std": 0.46}
}

FEATURE_ORDER = list(FEATURE_STATS.keys())

def create_scaler():
    """Create a StandardScaler with pre-fitted statistics"""
    scaler = StandardScaler()
    
    # Set the fitted parameters directly
    n_features = len(FEATURE_ORDER)
    scaler.mean_ = np.array([FEATURE_STATS[f]["mean"] for f in FEATURE_ORDER])
    scaler.var_ = np.array([FEATURE_STATS[f]["std"]**2 for f in FEATURE_ORDER])
    scaler.scale_ = np.array([FEATURE_STATS[f]["std"] for f in FEATURE_ORDER])
    scaler.n_features_in_ = n_features
    scaler.n_samples_seen_ = 10000  # Simulated training samples
    scaler.feature_names_in_ = np.array(FEATURE_ORDER)
    
    return scaler

def create_imputer():
    """Create a SimpleImputer with median fill strategy"""
    imputer = SimpleImputer(strategy='median')
    
    # Set the fitted parameters directly
    n_features = len(FEATURE_ORDER)
    imputer.statistics_ = np.array([FEATURE_STATS[f]["mean"] for f in FEATURE_ORDER])
    imputer.n_features_in_ = n_features
    imputer.feature_names_in_ = np.array(FEATURE_ORDER)
    
    return imputer

if __name__ == "__main__":
    import os
    
    # Ensure models directory exists
    models_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create and save scaler
    scaler = create_scaler()
    scaler_path = os.path.join(models_dir, "stage1_scaler.pkl")
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✓ Scaler saved to {scaler_path}")
    
    # Create and save imputer
    imputer = create_imputer()
    imputer_path = os.path.join(models_dir, "stage1_imputer.pkl")
    with open(imputer_path, 'wb') as f:
        pickle.dump(imputer, f)
    print(f"✓ Imputer saved to {imputer_path}")
    
    print("\nML artifacts created successfully!")
