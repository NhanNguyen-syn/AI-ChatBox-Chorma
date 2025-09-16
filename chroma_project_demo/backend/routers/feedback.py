from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional

from database import get_db, Feedback, User, ChatMessage
from auth.jwt_handler import verify_token
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

router = APIRouter()
security = HTTPBearer()

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security), db: Session = Depends(get_db)) -> User:
    payload = verify_token(credentials.credentials)
    username = payload.get("sub")
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

class FeedbackCreate(BaseModel):
    chat_message_id: str
    rating: int # 1 for like, -1 for dislike
    comment: Optional[str] = None

@router.post("/", status_code=201)
def create_feedback(
    feedback: FeedbackCreate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user)
):
    """Submit feedback for a specific chat message."""
    chat_message = db.query(ChatMessage).filter(ChatMessage.id == feedback.chat_message_id).first()
    if not chat_message:
        raise HTTPException(status_code=404, detail="Chat message not found")

    existing_feedback = db.query(Feedback).filter(
        Feedback.chat_message_id == feedback.chat_message_id,
        Feedback.user_id == user.id
    ).first()

    if existing_feedback:
        existing_feedback.rating = feedback.rating
        existing_feedback.comment = feedback.comment
        db.commit()
        return {"message": "Feedback updated successfully"}
    else:
        db_feedback = Feedback(
            chat_message_id=feedback.chat_message_id,
            user_id=user.id,
            rating=feedback.rating,
            comment=feedback.comment
        )
        db.add(db_feedback)
        db.commit()
        db.refresh(db_feedback)
        return {"message": "Feedback submitted successfully", "feedback_id": db_feedback.id}
