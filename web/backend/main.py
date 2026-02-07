"""
MirAI Backend - FastAPI Application Entry Point
Main application with CORS, routes, and ML artifact loading
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from database import init_db
from ml_service import get_ml_service
from ml_service_stage2 import get_stage2_ml_service
from ml_service_stage3 import get_stage3_ml_service
from auth import router as auth_router
from questionnaire import router as questionnaire_router
from stage2 import router as stage2_router
from stage3 import router as stage3_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    print("=" * 50)
    print("MirAI Backend Starting...")
    print("=" * 50)
    
    # Initialize database
    print("Initializing database...")
    init_db()
    print("✓ Database initialized")
    
    # Load Stage-1 ML artifacts
    print("Loading Stage-1 ML artifacts...")
    ml_service = get_ml_service()
    success1 = ml_service.load_artifacts()
    if success1:
        print("✓ Stage-1 ML service ready")
    else:
        print("⚠ Stage-1 ML service failed to load")
    
    # Load Stage-2 ML artifacts
    print("Loading Stage-2 ML artifacts...")
    stage2_ml = get_stage2_ml_service()
    success2 = stage2_ml.load_artifacts()
    if success2:
        print("✓ Stage-2 ML service ready")
    else:
        print("⚠ Stage-2 ML service failed to load")
    
    # Load Stage-3 ML artifacts
    print("Loading Stage-3 ML artifacts...")
    stage3_ml = get_stage3_ml_service()
    success3 = stage3_ml.load_artifacts()
    if success3:
        print("✓ Stage-3 ML service ready")
    else:
        print("⚠ Stage-3 ML service failed to load")
    
    print("=" * 50)
    print("MirAI Backend Ready!")
    print("=" * 50)
    
    yield
    
    # Shutdown
    print("MirAI Backend shutting down...")


# Create FastAPI app
app = FastAPI(
    title="MirAI API",
    description="Early Alzheimer's Disease Risk Assessment Platform - Research Prototype",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth_router)
app.include_router(questionnaire_router)
app.include_router(stage2_router)
app.include_router(stage3_router)


# Serve frontend static files
from fastapi.staticfiles import StaticFiles
from pathlib import Path

# Path to frontend directory (sibling to backend)
frontend_path = Path(__file__).parent.parent / "frontend"

if frontend_path.exists():
    app.mount("/", StaticFiles(directory=str(frontend_path), html=True), name="frontend")
else:
    print(f"⚠ Frontend directory not found at {frontend_path}")



@app.get("/health")
def health_check():
    """Health check endpoint"""
    ml_service = get_ml_service()
    return {
        "status": "healthy",
        "ml_service": "ready" if ml_service.is_loaded else "not_loaded",
        "database": "connected"
    }
