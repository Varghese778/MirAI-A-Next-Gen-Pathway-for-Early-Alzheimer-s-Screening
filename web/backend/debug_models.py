import pickle
import joblib
import os
from pathlib import Path

models_dir = Path(r"d:\mirai\backend\models")
scaler_path = models_dir / "stage1_scaler.pkl"
imputer_path = models_dir / "stage1_imputer.pkl"
model_path = models_dir / "stage1_model.json"

print(f"Checking models in {models_dir}")

def try_load(path, name):
    if not path.exists():
        print(f"❌ {name} does not exist at {path}")
        return
    
    print(f"Checking {name} ({path.stat().st_size} bytes)...")
    
    # Try pickle
    try:
        with open(path, 'rb') as f:
            pickle.load(f)
        print(f"  ✅ {name} loaded with pickle")
        return
    except Exception as e:
        print(f"  ⚠️ {name} failed with pickle: {e}")
        
    # Try joblib
    try:
        joblib.load(path)
        print(f"  ✅ {name} loaded with joblib")
        return
    except Exception as e:
        print(f"  ⚠️ {name} failed with joblib: {e}")

try_load(scaler_path, "Scaler")
try_load(imputer_path, "Imputer")

# Check XGBoost model
try:
    import xgboost as xgb
    bst = xgb.Booster()
    bst.load_model(str(model_path))
    print(f"✅ XGBoost model loaded")
except Exception as e:
    print(f"❌ XGBoost model failed: {e}")
