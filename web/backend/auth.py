"""
MirAI Backend - Authentication Service
User registration, login, logout, and session management
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel
import bcrypt
import secrets
from datetime import datetime, timedelta

from database import get_db
from models import User, UserSession

router = APIRouter(prefix="/api/auth", tags=["authentication"])

# Session duration: 24 hours
SESSION_DURATION_HOURS = 24


class RegisterRequest(BaseModel):
    username: str
    password: str


class LoginRequest(BaseModel):
    username: str
    password: str


class AuthResponse(BaseModel):
    success: bool
    message: str
    token: str | None = None
    user_id: int | None = None
    username: str | None = None


class UserResponse(BaseModel):
    id: int
    username: str
    created_at: datetime


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


def get_current_user(token: str, db: Session) -> User | None:
    """Get user from session token"""
    session = db.query(UserSession).filter(
        UserSession.token == token,
        UserSession.expires_at > datetime.utcnow()
    ).first()
    
    if session:
        return session.user
    return None


@router.post("/register", response_model=AuthResponse)
def register(request: RegisterRequest, db: Session = Depends(get_db)):
    """Register a new user"""
    # Validate input
    if len(request.username) < 3:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username must be at least 3 characters"
        )
    if len(request.password) < 6:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must be at least 6 characters"
        )
    
    # Check if username exists
    existing = db.query(User).filter(User.username == request.username).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already exists"
        )
    
    # Create user
    user = User(
        username=request.username,
        password_hash=hash_password(request.password)
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    
    # Create session
    token = create_session_token()
    session = UserSession(
        user_id=user.id,
        token=token,
        expires_at=datetime.utcnow() + timedelta(hours=SESSION_DURATION_HOURS)
    )
    db.add(session)
    db.commit()
    
    return AuthResponse(
        success=True,
        message="Registration successful",
        token=token,
        user_id=user.id,
        username=user.username
    )


@router.post("/login", response_model=AuthResponse)
def login(request: LoginRequest, db: Session = Depends(get_db)):
    """Login user and create session"""
    # Find user
    user = db.query(User).filter(User.username == request.username).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password"
        )
    
    # Verify password
    if not verify_password(request.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password"
        )
    
    # Create new session
    token = create_session_token()
    session = UserSession(
        user_id=user.id,
        token=token,
        expires_at=datetime.utcnow() + timedelta(hours=SESSION_DURATION_HOURS)
    )
    db.add(session)
    db.commit()
    
    return AuthResponse(
        success=True,
        message="Login successful",
        token=token,
        user_id=user.id,
        username=user.username
    )


@router.post("/logout", response_model=AuthResponse)
def logout(token: str, db: Session = Depends(get_db)):
    """Logout user and invalidate session"""
    session = db.query(UserSession).filter(UserSession.token == token).first()
    if session:
        db.delete(session)
        db.commit()
    
    return AuthResponse(
        success=True,
        message="Logout successful"
    )


@router.get("/me", response_model=UserResponse)
def get_me(token: str, db: Session = Depends(get_db)):
    """Get current user info"""
    user = get_current_user(token, db)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired session"
        )
    
    return UserResponse(
        id=user.id,
        username=user.username,
        created_at=user.created_at
    )


@router.get("/validate")
def validate_session(token: str, db: Session = Depends(get_db)):
    """Validate if session token is valid"""
    user = get_current_user(token, db)
    return {
        "valid": user is not None,
        "user_id": user.id if user else None,
        "username": user.username if user else None
    }
