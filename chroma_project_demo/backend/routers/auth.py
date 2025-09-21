from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from datetime import timedelta
from pydantic import BaseModel
from typing import Optional

from database import get_db, User
from auth.jwt_handler import verify_password, get_password_hash, create_access_token, ACCESS_TOKEN_EXPIRE_MINUTES

router = APIRouter()

class UserCreate(BaseModel):
    staff_code: str
    email: str
    password: str
    full_name: Optional[str] = None
    phone: Optional[str] = None

class UserLogin(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str
    user_id: str
    username: str
    is_admin: bool

class PasswordResetRequest(BaseModel):
    email: str


@router.post("/register", response_model=Token)
async def register(user: UserCreate, db: Session = Depends(get_db)):
    # Check if staff_code (stored as username) already exists
    db_user = db.query(User).filter(User.username == user.staff_code).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Staff code already registered")

    # Check email uniqueness
    db_user = db.query(User).filter(User.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    # Create new user
    hashed_password = get_password_hash(user.password)
    db_user = User(
        username=user.staff_code,  # store staff_code in username field
        email=user.email,
        full_name=user.full_name,
        phone=user.phone,
        hashed_password=hashed_password,
        is_admin=False
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)

    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.staff_code}, expires_delta=access_token_expires
    )

    return Token(
        access_token=access_token,
        token_type="bearer",
        user_id=db_user.id,
        username=db_user.username,
        is_admin=db_user.is_admin
    )

@router.post("/login", response_model=Token)
async def login(form_data: UserLogin, db: Session = Depends(get_db)):
    # Find user
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Account is disabled",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Update last_login timestamp
    try:
        from datetime import datetime as _dt
        user.last_login = _dt.utcnow()
        # Optionally mark account_status active on login
        try:
            if hasattr(user, 'account_status'):
                user.account_status = 'active'
        except Exception:
            pass
        db.commit()
    except Exception:
        db.rollback()

    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )

    return Token(
        access_token=access_token,
        token_type="bearer",
        user_id=user.id,
        username=user.username,
        is_admin=user.is_admin
    )

@router.post("/request-password-reset")
async def request_password_reset(payload: PasswordResetRequest, db: Session = Depends(get_db)):
    # Always respond with success message to avoid user enumeration
    try:
        user = db.query(User).filter(User.email == payload.email).first()
        if user:
            # In a real system, generate a token and send email here.
            # For now, just log to console for debugging.
            print(f"[PasswordReset] Requested by email: {payload.email} (user: {user.username})")
        else:
            print(f"[PasswordReset] Requested by email: {payload.email} (no account)")
    except Exception as e:
        # Do not reveal errors to client; keep generic response
        print(f"[PasswordReset] Error handling request for {payload.email}: {e}")
    return {"message": "If the email exists in our system, a password reset link has been sent."}