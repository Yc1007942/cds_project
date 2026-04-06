from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional
from database import get_db
from models import User
from datetime import datetime

router = APIRouter()

class UserResponse(BaseModel):
    id: int
    openId: str
    name: Optional[str]
    email: Optional[str]
    role: str
    
    class Config:
        from_attributes = True

class LoginRequest(BaseModel):
    openId: str
    name: Optional[str] = None
    email: Optional[str] = None
    loginMethod: Optional[str] = None

@router.post("/login")
async def login(request: LoginRequest, db: Session = Depends(get_db)):
    """Handle user login/registration"""
    user = db.query(User).filter(User.openId == request.openId).first()
    
    if not user:
        # Create new user
        user = User(
            openId=request.openId,
            name=request.name,
            email=request.email,
            loginMethod=request.loginMethod,
            role="user"
        )
        db.add(user)
    else:
        # Update existing user
        user.lastSignedIn = datetime.utcnow()
        if request.name:
            user.name = request.name
        if request.email:
            user.email = request.email
    
    db.commit()
    db.refresh(user)
    
    return {
        "success": True,
        "user": UserResponse.from_orm(user)
    }

@router.get("/me")
async def get_current_user(db: Session = Depends(get_db)):
    """Get current user info (placeholder - in production, use JWT tokens)"""
    # For now, return a mock user
    # In production, extract user from JWT token
    return {
        "id": 1,
        "openId": "test-user",
        "name": "Test User",
        "email": "test@example.com",
        "role": "user"
    }

@router.post("/logout")
async def logout():
    """Handle user logout"""
    return {"success": True}
