# MirAI - Stage 1
**Early Alzheimer's Disease Risk Assessment Platform**

MirAI is a research-prototype web application designed to screen for early signs of Alzheimer's disease using a clinically-validated questionnaire and machine learning inference.

## Features
- **Secure Authentication**: User registration and login with bcrypt hashing and session management.
- **Stage-1 Screening**: 25-question research-based assessment covering memory, function, family history, and lifestyle.
- **ML Inference**: XGBoost-based risk prediction model trained on clinical data.
- **Interactive Dashboard**: Track assessment history and view detailed results.
- **Privacy-Focused**: Locally runnable architecture.

## Installation & Setup

1. **Prerequisites**
   - Python 3.9+
   - pip

2. **Backend Setup**
   ```bash
   cd d:\mirai\backend
   pip install -r requirements.txt
   ```

3. **ML Artifacts**
   Ensure the following files are in `d:\mirai\backend\models\`:
   - `stage1_model.json` (XGBoost model)
   - `stage1_scaler.pkl` (StandardScaler)
   - `stage1_imputer.pkl` (SimpleImputer)

## Running the Application

1. **Start the Backend Server**
   ```bash
   cd d:\mirai\backend
   uvicorn main:app --reload
   ```

2. **Access the Application**
   Open your web browser and navigate to:
   [http://localhost:8000](http://localhost:8000)

## Project Structure
- `backend/`: FastAPI application, database models, ML service.
- `frontend/`: HTML/CSS/JS user interface.
- `backend/models/`: Machine learning model artifacts.

## Usage
1. Register a new account.
2. Login to access the dashboard.
3. Click "Start Assessment" to begin the Stage-1 questionnaire.
4. Submit answers to receive your risk analysis and contributing factors.

## Disclaimer
This tool is for research purposes only and does not provide a medical diagnosis. Always consult a healthcare professional for medical advice.
