# MirAI Project Skills & Capabilities

## Core Capabilities
- **Clinical Screening**: Implements a 25-point comprehensive questionnaire based on AD8, MMSE, and modern risk factor research.
- **Risk Prediction**: Uses Gradient Boosting (XGBoost) to predict Alzheimer's risk probability based on demographic, genetic, and lifestyle factors.
- **Factor Analysis**: Identifies top contributing factors (e.g., "high blood pressure", "sleep deprivation") to explain the risk score.
- **Longitudinal Tracking**: Stores assessment history to monitor changes in risk profile over time.

## Technical Architecture
- **Backend**: FastAPI (Python) for high-performance async API.
- **Database**: SQLite with SQLAlchemy ORM for relational data persistence.
- **ML Pipeline**: Scikit-learn (Imputation/Scaling) + XGBoost (Inference).
- **Frontend**: Vanilla JS/CSS for a lightweight, dependency-free UI with glassmorphism design.
- **Security**: 
  - Password hashing (bcrypt)
  - HTTP-only session tokens
  - Input validation (Pydantic)

## Future Roadmap (Stage 2)
- **Genetic Data Integration**: Upload and analysis of genomic data (APOE-e4 variants).
- **Voice Biomarkers**: Analysis of speech patterns for cognitive decline indicators.
- **Neuroimaging**: Support for MRI/PET scan feature integration.
