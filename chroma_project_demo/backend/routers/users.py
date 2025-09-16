from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional
from datetime import datetime

from database import get_db, User
from auth.jwt_handler import verify_token, get_password_hash
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

router = APIRouter()
security = HTTPBearer()

class UserProfile(BaseModel):
    username: str
    email: str
    full_name: Optional[str] = None
    is_admin: bool
    created_at: datetime

class UserUpdate(BaseModel):
    email: Optional[str] = None
    full_name: Optional[str] = None
    current_password: Optional[str] = None
    new_password: Optional[str] = None

def verify_user(credentials: HTTPAuthorizationCredentials = Depends(security), db: Session = Depends(get_db)):
    payload = verify_token(credentials.credentials)
    user = db.query(User).filter(User.username == payload["sub"]).first()
    
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="Invalid user")
    
    return user

@router.get("/profile", response_model=UserProfile)
async def get_user_profile(
    user: User = Depends(verify_user),
    db: Session = Depends(get_db)
):
    return UserProfile(
        username=user.username,
        email=user.email,
        full_name=getattr(user, 'full_name', None),
        is_admin=user.is_admin,
        created_at=user.created_at
    )

@router.put("/profile")
async def update_user_profile(
    user_update: UserUpdate,
    user: User = Depends(verify_user),
    db: Session = Depends(get_db)
):
    # Update email if provided
    if user_update.email and user_update.email != user.email:
        # Check if email is already taken
        existing_user = db.query(User).filter(User.email == user_update.email).first()
        if existing_user:
            raise HTTPException(status_code=400, detail="Email already registered")
        user.email = user_update.email
    
    # Update full name if provided
    if user_update.full_name:
        user.full_name = user_update.full_name
    
    # Update password if provided
    if user_update.new_password:
        if not user_update.current_password:
            raise HTTPException(status_code=400, detail="Current password required")
        
        # Verify current password
        from auth.jwt_handler import verify_password
        if not verify_password(user_update.current_password, user.hashed_password):
            raise HTTPException(status_code=400, detail="Current password is incorrect")
        
        # Hash new password
        user.hashed_password = get_password_hash(user_update.new_password)
    
    user.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(user)
    
    return {"message": "Profile updated successfully"}

@router.get("/faqs")
async def get_faqs_for_users(
    user: User = Depends(verify_user),
    db: Session = Depends(get_db)
):
    from database import FAQ
    faqs = db.query(FAQ).filter(FAQ.is_active == True).order_by(FAQ.category, FAQ.question).all()
    
    # Group by category
    faq_by_category = {}
    for faq in faqs:
        if faq.category not in faq_by_category:
            faq_by_category[faq.category] = []
        faq_by_category[faq.category].append({
            "id": faq.id,
            "question": faq.question,
            "answer": faq.answer
        })
    
    return faq_by_category 