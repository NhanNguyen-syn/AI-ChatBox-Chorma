from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from sqlalchemy import text
from pydantic import BaseModel, ConfigDict
from typing import List, Optional
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

from database import get_db, User, ChatSession, ChatMessage, Document, FAQ, SystemConfig, CrmProduct, Feedback, IgnoredQuestion
from auth.jwt_handler import verify_token
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

import os
import json
from database import ExcelFile, ExcelRow, get_chroma_client, get_chroma_collection_for_backend, engine

# OpenAI embeddings setup (reuse env flags similar to chat/files routers)
USE_OPENAI = os.getenv("USE_OPENAI", "0") == "1"
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
openai_client = None
if USE_OPENAI:
    try:
        from openai import OpenAI
        openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        print("[AdminExcel] Using OpenAI for embeddings")
    except Exception as e:
        print(f"[AdminExcel] OpenAI init failed: {e}")
        USE_OPENAI = False

# Optional local sentence-transformers fallback
try:
    from sentence_transformers import SentenceTransformer
    _st_model: SentenceTransformer | None = None
    if not USE_OPENAI:
        try:
            _st_model = SentenceTransformer(os.getenv("ST_EMBED_MODEL", "all-MiniLM-L6-v2"))
            print("[AdminExcel] Using sentence-transformers fallback")
        except Exception as _e:
            _st_model = None
except Exception:
    SentenceTransformer = None  # type: ignore
    _st_model = None  # type: ignore

# -------- Excel helpers --------
import re
import unicodedata

def _to_ascii(text: str) -> str:
    """Convert Vietnamese with diacritics to ASCII letters (san pham -> san pham).
    This keeps semantics for column names instead of dropping characters.
    """
    try:
        norm = unicodedata.normalize("NFKD", text or "")
        return "".join([c for c in norm if not unicodedata.combining(c)])
    except Exception:
        return text or ""


def _std_col_name(s: str) -> str:
    s = _to_ascii((s or "").strip())  # remove diacritics first
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^0-9a-zA-Z_]+", "", s)
    return s.lower()


def _clean_dataframe(df):
    import pandas as pd
    if df is None:
        return df
    # Drop completely empty rows/cols
    df = df.dropna(how='all')
    df = df.dropna(axis=1, how='all')
    # Standardize column names
    df.columns = [_std_col_name(str(c)) for c in df.columns]
    # Trim string cells
    for c in df.columns:
        try:
            df[c] = df[c].apply(lambda x: x.strip() if isinstance(x, str) else x)
        except Exception:
            pass
    return df


def _convert_types(df):
    import pandas as pd
    if df is None or df.empty:
        return df

    for col in df.columns:
        # Skip if column is already numeric or datetime
        if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_datetime64_any_dtype(df[col]):
            continue

        # A. Attempt to convert to numeric, handling various currency/number formats
        if pd.api.types.is_object_dtype(df[col]):
            series_as_str = df[col].astype(str).str.strip()

            # Comprehensive cleaning: remove currency, thousands separators, whitespace
            # Handles formats like "1.500.000đ", "1,500,000", "$50.99"
            cleaned_series = series_as_str.str.replace(r'[đ₫$€£\s]', '', regex=True)
            # First, remove dots (as thousands separators)
            cleaned_series = cleaned_series.str.replace('.', '', regex=False)
            # Then, replace comma with dot (for decimal)
            cleaned_series = cleaned_series.str.replace(',', '.', regex=False)

            numeric_series = pd.to_numeric(cleaned_series, errors='coerce')

            # If a significant portion is numeric after cleaning, convert the whole column
            if numeric_series.notna().sum() / len(series_as_str.dropna()) > 0.7:
                df[col] = numeric_series
                continue

            # B. Attempt to convert to datetime (if not successfully converted to numeric)
            try:
                # Coerce errors will turn unparseable dates into NaT (Not a Time)
                datetime_series = pd.to_datetime(df[col], errors='coerce', dayfirst=True)
                # If a significant portion is datetime, convert the whole column
                if datetime_series.notna().sum() / len(df[col].dropna()) > 0.7:
                    df[col] = datetime_series.dt.date.astype(str)
                    continue
            except Exception:
                pass # Ignore if datetime conversion fails

    return df


def _data_quality_report(df) -> dict:
    import pandas as pd
    if df is None:
        return {"rows": 0, "columns": [], "missing": {}}
    dtypes = {c: str(df[c].dtype) for c in df.columns}
    missing = {c: int(df[c].isna().sum()) for c in df.columns}
    return {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "dtypes": dtypes,
        "missing": missing,
    }


def _chunk_dataframe(df, chunk_size: int = 200):
    n = len(df)
    for start in range(0, n, chunk_size):
        yield start, min(start + chunk_size, n), df.iloc[start:start + chunk_size]


def _format_row_group_as_text(df_group, sheet_name: str, row_range: tuple[int, int]) -> str:
    """Converts a group of DataFrame rows into a natural language text chunk."""
    start, end = row_range
    text_lines = [f"Dữ liệu từ file Excel, sheet '{sheet_name}', các dòng từ {start + 1} đến {end}:\n"]
    for _, row in df_group.iterrows():
        row_desc = []
        for col, val in row.items():
            if val is not None and str(val).strip() != "":
                row_desc.append(f"{col.replace('_', ' ')} là '{val}'")
        if row_desc:
            text_lines.append("- " + ", ".join(row_desc) + ".")
    return "\n".join(text_lines)

# --- Helpers: SQL storage & CRM extraction ---

def _normalize_key(s: str) -> str:
    try:
        s2 = unicodedata.normalize('NFKD', s or '')
        s2 = ''.join([c for c in s2 if not unicodedata.combining(c)])
    except Exception:
        s2 = s or ''
    s2 = s2.lower().strip()
    s2 = re.sub(r"[^a-z0-9_]+", "_", s2)
    return s2.strip("_")


def _row_to_norm_text(row: dict) -> str:
    parts = []
    for k, v in (row or {}).items():
        if v is None: continue
        vs = str(v).strip()
        if not vs: continue
        ks = _normalize_key(k).replace('_', ' ')
        parts.append(f"{ks}: {vs}")
    return _to_ascii("; ".join(parts)).lower()


_PRICE_KEYS = {"gia", "gia_ban", "don_gia", "price", "lam_tron_gia"}
_SKU_KEYS = {"sku", "ma", "ma_sp", "ma_san_pham", "so_ma_hang", "so_mh_hang", "item_code", "product_code"}
_NAME_KEYS = {"ten", "ten_sp", "ten_san_pham", "san_pham", "product_name", "variant", "variant_code"}
_CAT_KEYS = {"loai", "category", "danh_muc"}
_CUR_KEYS = {"tien_te", "currency", "don_vi_tien"}
_DESC_KEYS = {"mo_ta", "ghi_chu", "description"}


def _extract_crm_from_row(row: dict) -> dict:
    """Heuristically extract CRM-style fields from a cleaned Excel row dict.
    Returns a dict with any of: sku, name, price, currency, category, description, attributes(json).
    """
    if not row: return {}
    norm_map = { _normalize_key(k): k for k in row.keys() }

    def find_value(candidates: set[str]):
        for nk, orig in norm_map.items():
            for c in candidates:
                if c in nk:
                    val = row.get(orig)
                    if val is not None and str(val).strip() != "":
                        return str(val).strip()
        return None

    price_raw = find_value(_PRICE_KEYS)
    # Normalize price: remove currency and thousands separators
    price = None
    if price_raw is not None:
        pr = re.sub(r"[đ₫$€£\s]", "", str(price_raw))
        pr = pr.replace('.', '').replace(',', '.')
        try:
            price = str(float(pr)).rstrip('0').rstrip('.') if pr else None
        except Exception:
            price = pr if pr else None

    crm = {
        "sku": find_value(_SKU_KEYS) or "",
        "name": find_value(_NAME_KEYS) or "",
        "price": price or "",
        "currency": find_value(_CUR_KEYS) or "",
        "category": find_value(_CAT_KEYS) or "",
        "description": find_value(_DESC_KEYS) or "",
    }
    # Only keep if we have at least sku or name or price
    if not (crm["sku"] or crm["name"] or crm["price"]):
        return {}
    return crm


router = APIRouter()
security = HTTPBearer()


# Super admin usernames that are immutable and shown with crown in UI
_SUPER_ADMINS = {"admin", "sg0510"}

def _is_super_admin_username(username: str | None) -> bool:
    try:
        return (username or "").strip().lower() in _SUPER_ADMINS
    except Exception:
        return False

class UserResponse(BaseModel):
    id: str
    username: str
    email: str
    full_name: str | None = None
    is_admin: bool
    is_active: bool
    account_status: Optional[str] = None
    role: Optional[str] = None
    created_at: datetime
    chat_count: int
    token_quota: int | None = None

class ChatStatsResponse(BaseModel):
    total_sessions: int
    total_messages: int
    total_tokens: int
    avg_response_time: float
    active_users_today: int
    active_users_week: int

class FAQCreate(BaseModel):
    question: str
    answer: str
    category: str

class FAQResponse(BaseModel):
    id: str
    question: str
    answer: str
    category: str
    is_active: bool
    created_at: datetime

    # Pydantic v2 config: allow parsing from ORM objects
    model_config = ConfigDict(from_attributes=True)

class SystemConfigUpdate(BaseModel):
    key: str
    value: str
    description: str

class AdminUserDetail(BaseModel):
    id: str
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    phone: Optional[str] = None
    role: Optional[str] = None
    department: Optional[str] = None
    account_status: Optional[str] = None
    is_admin: bool
    is_active: bool
    created_at: datetime
    last_login: Optional[datetime] = None
    chat_count: int

class AdminUserUpdate(BaseModel):
    email: Optional[str] = None
    full_name: Optional[str] = None
    phone: Optional[str] = None
    username: Optional[str] = None  # staff code
    role: Optional[str] = None
    department: Optional[str] = None
    account_status: Optional[str] = None  # active | inactive | suspended
    new_password: Optional[str] = None


def verify_admin(credentials: HTTPAuthorizationCredentials = Depends(security), db: Session = Depends(get_db)):
    payload = verify_token(credentials.credentials)
    user = db.query(User).filter(User.username == payload["sub"]).first()

    if not user or not user.is_active or not user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")

    return user

class PaginatedUsersResponse(BaseModel):
    users: List[UserResponse]
    total: int


class FeedbackChatResponse(BaseModel):
    feedback_id: int
    chat_message_id: str
    session_id: str
    user_question: str
    assistant_response: str
    timestamp: datetime

@router.get("/users", response_model=PaginatedUsersResponse)
async def get_all_users(
    admin: User = Depends(verify_admin),
    db: Session = Depends(get_db),
    page: int = 1,
    limit: int = 10,
    role: str = 'all',
    status: str = 'all',
):
    query = db.query(User)

    if role == 'admin':
        query = query.filter(User.is_admin == True)
    elif role == 'user':
        query = query.filter(User.is_admin == False)

    if status == 'active':
        query = query.filter(User.is_active == True)
    elif status == 'inactive':
        query = query.filter(User.is_active == False)
    elif status == 'suspended':
        try:
            query = query.filter(User.account_status == 'suspended')
        except Exception:
            pass

    all_matching_users = query.all()

    # Sort to bring super admins to the top
    all_matching_users.sort(key=lambda u: not _is_super_admin_username(u.username))
    total = len(all_matching_users)

    start = (page - 1) * limit
    end = start + limit
    paginated_users = all_matching_users[start:end]

    result = []
    for user in paginated_users:
        chat_count = db.query(ChatSession).filter(ChatSession.user_id == user.id).count()
        result.append(UserResponse(
            id=user.id,
            username=user.username,
            email=user.email,
            full_name=getattr(user, 'full_name', None),
            is_admin=bool(getattr(user, 'is_admin', False)),
            is_active=bool(getattr(user, 'is_active', True)),
            account_status=getattr(user, 'account_status', None),
            role=getattr(user, 'role', None),
            created_at=user.created_at,
            chat_count=chat_count,
            token_quota=user.token_quota
        ))

    return PaginatedUsersResponse(users=result, total=total)

@router.get("/users/{user_id}", response_model=AdminUserDetail)
async def get_user_detail(user_id: str, admin: User = Depends(verify_admin), db: Session = Depends(get_db)):
    u = db.query(User).filter(User.id == user_id).first()
    if not u:
        raise HTTPException(status_code=404, detail="User not found")
    chat_count = db.query(ChatSession).filter(ChatSession.user_id == u.id).count()
    return AdminUserDetail(
        id=u.id,
        username=u.username,
        email=u.email,
        full_name=getattr(u, 'full_name', None),
        phone=getattr(u, 'phone', None),
        role=getattr(u, 'role', None),
        department=getattr(u, 'department', None),
        account_status=getattr(u, 'account_status', None),
        is_admin=bool(getattr(u, 'is_admin', False)),
        is_active=bool(getattr(u, 'is_active', True)),
        created_at=u.created_at,
        last_login=getattr(u, 'last_login', None),
        chat_count=chat_count,
    )

@router.put("/users/{user_id}", response_model=AdminUserDetail)
async def update_user_detail(user_id: str, payload: AdminUserUpdate, admin: User = Depends(verify_admin), db: Session = Depends(get_db)):
    u = db.query(User).filter(User.id == user_id).first()
    if not u:
        raise HTTPException(status_code=404, detail="User not found")
    # Protect super admins
    is_super = _is_super_admin_username(u.username)

    # Email uniqueness
    if payload.email and payload.email != u.email:
        exists = db.query(User).filter(User.email == payload.email).first()
        if exists:
            raise HTTPException(status_code=400, detail="Email already in use")
        u.email = payload.email

    # Username (staff code)
    if payload.username and payload.username != u.username:
        if is_super:
            raise HTTPException(status_code=403, detail="Không thể đổi staff code của tài khoản admin độc quyền")
        exists2 = db.query(User).filter(User.username == payload.username).first()
        if exists2:
            raise HTTPException(status_code=400, detail="Staff code already in use")
        u.username = payload.username

    if payload.full_name is not None:
        u.full_name = payload.full_name
    if payload.phone is not None:
        u.phone = payload.phone
    if payload.department is not None:
        u.department = payload.department
    if payload.role is not None:
        if is_super:
            pass
        else:
            u.role = payload.role
            u.is_admin = True if (payload.role or '').lower() == 'admin' else u.is_admin
    if payload.account_status is not None:
        if is_super:
            pass
        else:
            u.account_status = payload.account_status
            if payload.account_status.lower() == 'active':
                u.is_active = True
            elif payload.account_status.lower() == 'inactive':
                u.is_active = False

    # Update password if provided
    if payload.new_password:
        from auth.jwt_handler import get_password_hash
        u.hashed_password = get_password_hash(payload.new_password)

    db.commit(); db.refresh(u)
    chat_count = db.query(ChatSession).filter(ChatSession.user_id == u.id).count()
    return AdminUserDetail(
        id=u.id,
        username=u.username,
        email=u.email,
        full_name=getattr(u, 'full_name', None),
        phone=getattr(u, 'phone', None),
        role=getattr(u, 'role', None),
        department=getattr(u, 'department', None),
        account_status=getattr(u, 'account_status', None),
        is_admin=bool(getattr(u, 'is_admin', False)),
        is_active=bool(getattr(u, 'is_active', True)),
        created_at=u.created_at,
        last_login=getattr(u, 'last_login', None),
        chat_count=chat_count,
    )

@router.put("/users/{user_id}", response_model=AdminUserDetail)
async def update_user_detail(user_id: str, payload: AdminUserUpdate, admin: User = Depends(verify_admin), db: Session = Depends(get_db)):
    u = db.query(User).filter(User.id == user_id).first()
    if not u:
        raise HTTPException(status_code=404, detail="User not found")
    # Protect super admins
    is_super = _is_super_admin_username(u.username)

    # Email uniqueness
    if payload.email and payload.email != u.email:
        exists = db.query(User).filter(User.email == payload.email).first()
        if exists:
            raise HTTPException(status_code=400, detail="Email already in use")
        u.email = payload.email

    # Username (staff code)
    if payload.username and payload.username != u.username:
        if is_super:
            raise HTTPException(status_code=403, detail="Không thể đổi staff code của tài khoản admin độc quyền")
        exists2 = db.query(User).filter(User.username == payload.username).first()
        if exists2:
            raise HTTPException(status_code=400, detail="Staff code already in use")
        u.username = payload.username

    if payload.full_name is not None:
        u.full_name = payload.full_name
    if payload.phone is not None:
        u.phone = payload.phone
    if payload.department is not None:
        u.department = payload.department
    if payload.role is not None:
        if is_super:
            pass  # ignore change
        else:
            u.role = payload.role
            u.is_admin = True if (payload.role or '').lower() == 'admin' else u.is_admin
    if payload.account_status is not None:
        if is_super:
            pass  # ignore change
        else:
            u.account_status = payload.account_status
            # Also mirror to is_active for active/inactive
            if payload.account_status.lower() == 'active':
                u.is_active = True
            elif payload.account_status.lower() == 'inactive':
                u.is_active = False

    db.commit(); db.refresh(u)
    chat_count = db.query(ChatSession).filter(ChatSession.user_id == u.id).count()
    return AdminUserDetail(
        id=u.id,
        username=u.username,
        email=u.email,
        full_name=getattr(u, 'full_name', None),
        phone=getattr(u, 'phone', None),
        role=getattr(u, 'role', None),
        department=getattr(u, 'department', None),
        account_status=getattr(u, 'account_status', None),
        is_admin=bool(getattr(u, 'is_admin', False)),
        is_active=bool(getattr(u, 'is_active', True)),
        created_at=u.created_at,
        last_login=getattr(u, 'last_login', None),
        chat_count=chat_count,
    )


@router.put("/users/{user_id}/toggle")
async def toggle_user_status(
    user_id: str,
    admin: User = Depends(verify_admin),
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if _is_super_admin_username(user.username):
        raise HTTPException(status_code=403, detail="Không thể thay đổi trạng thái của tài khoản admin độc quyền")

    user.is_active = not bool(getattr(user, 'is_active', True))
    db.commit()

    return {"message": f"User {'activated' if user.is_active else 'deactivated'}"}

@router.put("/users/{user_id}/admin")
async def toggle_admin_status(
    user_id: str,
    admin: User = Depends(verify_admin),
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if _is_super_admin_username(user.username):
        raise HTTPException(status_code=403, detail="Không thể thay đổi trạng thái của tài khoản admin độc quyền")

    user.is_admin = not bool(getattr(user, 'is_admin', False))
    db.commit()

    return {"message": f"User {'promoted to' if user.is_admin else 'removed from'} admin"}

class QuotaUpdateRequest(BaseModel):
    token_quota: int

@router.put("/users/{user_id}/quota")
async def update_user_quota(
    user_id: str,
    request: QuotaUpdateRequest,
    admin: User = Depends(verify_admin),
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    user.token_quota = request.token_quota
    db.commit()

    return {"message": f"User {user.username}'s token quota updated to {user.token_quota}"}

@router.get("/stats", response_model=ChatStatsResponse)
async def get_chat_stats(
    admin: User = Depends(verify_admin),
    db: Session = Depends(get_db)
):
    # Total sessions and messages
    total_sessions = db.query(ChatSession).count()
    total_messages = db.query(ChatMessage).count()

    # Total tokens used
    # Real sums/averages from chat_messages
    total_tokens = db.query(func.sum(ChatMessage.tokens_used)).scalar() or 0
    avg_response_time = float(db.query(func.avg(ChatMessage.response_time)).scalar() or 0)

    # Active users (today and this week)
    today = datetime.utcnow().date()
    week_ago = today - timedelta(days=7)

    active_users_today = db.query(func.count(func.distinct(ChatSession.user_id)))\
        .filter(func.date(ChatSession.updated_at) == today).scalar() or 0

    active_users_week = db.query(func.count(func.distinct(ChatSession.user_id)))\
        .filter(ChatSession.updated_at >= week_ago).scalar() or 0

    return ChatStatsResponse(
        total_sessions=total_sessions,
        total_messages=total_messages,
        total_tokens=total_tokens,
        avg_response_time=avg_response_time,
        active_users_today=active_users_today,
        active_users_week=active_users_week
    )

@router.get("/activity")
async def get_activity_json(
    admin: User = Depends(verify_admin),
    db: Session = Depends(get_db)
):
    daily = db.query(
        func.date(ChatMessage.timestamp).label('date'),
        func.count(ChatMessage.id).label('count')
    ).group_by(func.date(ChatMessage.timestamp)).order_by(func.date(ChatMessage.timestamp)).all()
    return [{"date": str(row.date), "count": row.count} for row in daily]


@router.get("/token-activity")
async def get_token_activity_json(
    admin: User = Depends(verify_admin),
    db: Session = Depends(get_db)
):
    daily_tokens = db.query(
        func.date(ChatMessage.timestamp).label('date'),
        func.sum(ChatMessage.tokens_used).label('tokens')
    ).group_by(func.date(ChatMessage.timestamp)).order_by(func.date(ChatMessage.timestamp)).all()
    return [{"date": str(row.date), "tokens": row.tokens or 0} for row in daily_tokens]


@router.get("/frequent-questions")
async def get_frequent_questions(
    admin: User = Depends(verify_admin),
    db: Session = Depends(get_db),
    limit: int = 10
):
    """
    Identifies and returns the most frequently asked user questions.
    Filters out simple greetings, very short messages, and ignored questions.
    """
    # Subquery to get all ignored questions
    ignored_questions_subquery = db.query(IgnoredQuestion.question).subquery()

    # Query to get questions, filter out assistant messages, ignored questions, group by content, count, and order
    frequent_questions = db.query(
        ChatMessage.message.label('question'),
        func.count(ChatMessage.message).label('count')
    ).filter(
        ChatMessage.is_user == True,
        func.length(ChatMessage.message) > 15,  # Filter out very short messages/greetings
        ChatMessage.message.notin_(ignored_questions_subquery) # Filter out ignored questions
    ).group_by(
        ChatMessage.message
    ).order_by(
        desc('count')
    ).limit(limit).all()

    return [{"question": row.question, "count": row.count} for row in frequent_questions]

class IgnoreQuestionRequest(BaseModel):
    question: str

@router.post("/frequent-questions/ignore")
async def ignore_frequent_question(
    request: IgnoreQuestionRequest,
    admin: User = Depends(verify_admin),
    db: Session = Depends(get_db)
):
    # Check if the question is already ignored
    existing = db.query(IgnoredQuestion).filter(IgnoredQuestion.question == request.question).first()
    if existing:
        return {"message": "Question already ignored"}

    # Add the new question to the ignored list
    ignored_entry = IgnoredQuestion(
        question=request.question,
        ignored_by_id=admin.id
    )
    db.add(ignored_entry)
    db.commit()

    return {"message": "Question ignored successfully"}


class IgnoredQuestionResponse(BaseModel):
    id: int
    question: str
    ignored_at: datetime
    ignored_by_username: Optional[str] = None

    # Pydantic v2 config
    model_config = ConfigDict(from_attributes=True)

@router.get("/ignored-questions", response_model=List[IgnoredQuestionResponse])
async def get_ignored_questions(
    admin: User = Depends(verify_admin),
    db: Session = Depends(get_db)
):
    rows = (
        db.query(
            IgnoredQuestion.id,
            IgnoredQuestion.question,
            IgnoredQuestion.ignored_at,
            User.username.label("ignored_by_username")
        )
        .outerjoin(User, IgnoredQuestion.ignored_by_id == User.id)
        .order_by(IgnoredQuestion.ignored_at.desc())
        .all()
    )
    return [
        {
            "id": r.id,
            "question": r.question,
            "ignored_at": r.ignored_at,
            "ignored_by_username": r.ignored_by_username,
        }
        for r in rows
    ]

@router.delete("/ignored-questions/{ignored_id}")
async def unignore_question(
    ignored_id: int,
    admin: User = Depends(verify_admin),
    db: Session = Depends(get_db)
):
    ignored_entry = db.query(IgnoredQuestion).filter(IgnoredQuestion.id == ignored_id).first()
    if not ignored_entry:
        raise HTTPException(status_code=404, detail="Ignored question entry not found")

    db.delete(ignored_entry)
    db.commit()

    return {"message": "Question restored successfully"}



@router.get("/feedback-chats", response_model=List[FeedbackChatResponse])
async def get_negative_feedback_chats(
    admin: User = Depends(verify_admin),
    db: Session = Depends(get_db)
):
    # Get all negative feedback records, joining with the message to get the session_id
    negative_feedbacks = db.query(Feedback).filter(Feedback.rating == -1).order_by(Feedback.created_at.desc()).limit(100).all()

    response_data = []
    for feedback in negative_feedbacks:
        # Get the assistant message that received the feedback
        assistant_message = db.query(ChatMessage).filter(ChatMessage.id == feedback.chat_message_id).first()
        if not assistant_message:
            continue

        # Find the preceding user message in the same session
        user_message = db.query(ChatMessage).filter(
            ChatMessage.session_id == assistant_message.session_id,
            ChatMessage.is_user == True,
            ChatMessage.timestamp < assistant_message.timestamp
        ).order_by(ChatMessage.timestamp.desc()).first()

        response_data.append(FeedbackChatResponse(
            feedback_id=feedback.id,
            chat_message_id=assistant_message.id,
            session_id=assistant_message.session_id,
            user_question=(user_message.message if user_message else "[Không tìm thấy câu hỏi]"),
            assistant_response=(assistant_message.response or ""),
            timestamp=feedback.created_at
        ))

    return response_data

@router.get("/chat-history")
async def get_all_chat_history(
    admin: User = Depends(verify_admin),
    db: Session = Depends(get_db),
    limit: int = 100
):
    messages = db.query(ChatMessage, User.username)\
        .join(ChatSession, ChatMessage.session_id == ChatSession.id)\
        .join(User, ChatSession.user_id == User.id)\
        .order_by(desc(ChatMessage.timestamp))\
        .limit(limit)\
        .all()

    result = []
    for message, username in messages:
        # Derive role and content from is_user/message/response schema
        role = "user" if getattr(message, "is_user", False) else "assistant"
        content = message.message if getattr(message, "is_user", False) else (message.response or "")
        result.append({
            "id": message.id,
            "username": username,
            "session_id": message.session_id,
            "role": role,
            "content": content,
            "timestamp": message.timestamp,
            "tokens_used": message.tokens_used,
            "response_time": message.response_time
        })

    return result

@router.delete("/chat-sessions/{session_id}")
async def delete_chat_session_admin(
    session_id: str,
    admin: User = Depends(verify_admin),
    db: Session = Depends(get_db)
):
    session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Delete messages first
    db.query(ChatMessage).filter(ChatMessage.session_id == session_id).delete()

    # Delete session
    db.delete(session)
    db.commit()

    return {"message": "Session deleted successfully"}

@router.get("/faqs", response_model=List[FAQResponse])
async def get_faqs(
    admin: User = Depends(verify_admin),
    db: Session = Depends(get_db)
):
    rows = (
        db.query(
            FAQ.id,
            FAQ.question,
            FAQ.answer,
            FAQ.category,
            FAQ.is_active,
            FAQ.created_at,
        )
        .order_by(FAQ.created_at.desc())
        .all()
    )
    return [
        {
            "id": r.id,
            "question": r.question,
            "answer": r.answer,
            "category": r.category or "General",
            "is_active": bool(r.is_active),
            "created_at": r.created_at,
        }
        for r in rows
    ]

@router.post("/faqs", response_model=FAQResponse)
async def create_faq(
    faq: FAQCreate,
    admin: User = Depends(verify_admin),
    db: Session = Depends(get_db)
):
    db_faq = FAQ(
        question=faq.question,
        answer=faq.answer,
        category=faq.category,
        created_by=admin.id
    )
    db.add(db_faq)
    db.commit()
    db.refresh(db_faq)

    return db_faq

@router.put("/faqs/{faq_id}")
async def update_faq(
    faq_id: int,
    faq: FAQCreate,
    admin: User = Depends(verify_admin),
    db: Session = Depends(get_db)
):
    db_faq = db.query(FAQ).filter(FAQ.id == faq_id).first()
    if not db_faq:
        raise HTTPException(status_code=404, detail="FAQ not found")

    db_faq.question = faq.question
    db_faq.answer = faq.answer
    db_faq.category = faq.category
    db_faq.updated_at = datetime.utcnow()

    db.commit()
    db.refresh(db_faq)

    return db_faq

@router.delete("/faqs/{faq_id}")
async def delete_faq(
    faq_id: int,
    admin: User = Depends(verify_admin),
    db: Session = Depends(get_db)
):
    db_faq = db.query(FAQ).filter(FAQ.id == faq_id).first()
    if not db_faq:
        raise HTTPException(status_code=404, detail="FAQ not found")

    db.delete(db_faq)
    db.commit()

    return {"message": "FAQ deleted successfully"}


class SuggestedFAQResponse(BaseModel):
    id: str
    question: str
    source_count: int

@router.get("/suggested-faqs", response_model=List[SuggestedFAQResponse])
async def get_suggested_faqs(
    admin: User = Depends(verify_admin),
    db: Session = Depends(get_db)
):
    # Placeholder implementation. This will be replaced with logic to find
    # common questions from user chats that are not yet in the FAQ.
    return []

@router.delete("/suggested-faqs/{faq_id}")
async def delete_suggested_faq(
    faq_id: str,
    admin: User = Depends(verify_admin),
    db: Session = Depends(get_db)
):
    # Placeholder implementation. This will be replaced with logic to delete
    # the suggested FAQ from the database or cache.
    return {"status": "ok", "message": f"Suggested FAQ {faq_id} deleted."}


@router.get("/system-configs")
async def get_system_configs(
    admin: User = Depends(verify_admin),
    db: Session = Depends(get_db)
):
    configs = db.query(SystemConfig).all()
    return configs

@router.put("/system-configs")
async def update_system_configs(
    configs: List[SystemConfigUpdate],
    admin: User = Depends(verify_admin),
    db: Session = Depends(get_db)
):
    for config in configs:
        db_config = db.query(SystemConfig).filter(SystemConfig.key == config.key).first()
        if db_config:
            db_config.value = config.value
            db_config.description = config.description
            db_config.updated_by = admin.id
            db_config.updated_at = datetime.utcnow()
        else:
            db_config = SystemConfig(
                key=config.key,
                value=config.value,
                description=config.description,
                updated_by=admin.id
            )
            db.add(db_config)

    db.commit()
    return {"message": "Configs updated successfully"}

# ---------- Excel Upload/List/Delete Endpoints ----------
from fastapi import UploadFile, File, Form
from typing import Any

class ExcelUploadResponse(BaseModel):
    id: str
    filename: str
    file_size: int
    sheet_names: list[str]
    rows_processed: int
    columns: list[str]
    data_types: dict
    missing_stats: dict
    collection_name: str
    uploaded_at: datetime


@router.post("/upload-excel", response_model=ExcelUploadResponse)
async def upload_excel(
    file: UploadFile = File(...),
    admin: User = Depends(verify_admin),
    db: Session = Depends(get_db)
):
    """Upload an Excel/CSV file, clean/normalize, chunk+embed to Chroma, and store structured rows.
    Also auto-extract CRM-like fields (SKU/Name/Price/Category) for SQL search.
    """
    import pandas as pd
    import json
    import uuid as _uuid
    import os

    # Validate extension
    allowed = {"xlsx", "xls", "csv"}
    ext = (file.filename or "").split(".")[-1].lower()
    if ext not in allowed:
        raise HTTPException(status_code=400, detail=f"Only support: {', '.join(sorted(allowed))}")

    # Save to disk with simple 50MB limit
    os.makedirs("uploads", exist_ok=True)
    tmp_path = os.path.join("uploads", f"{_uuid.uuid4()}_{file.filename}")
    size = 0
    with open(tmp_path, "wb") as f:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > 50 * 1024 * 1024:
                f.close()
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
                raise HTTPException(status_code=413, detail="File quá lớn (tối đa 50MB)")
            f.write(chunk)

    # Read file to DataFrame(s)
    sheet_dfs: dict[str, pd.DataFrame] = {}
    try:
        if ext in ("xlsx", "xls"):
            # prefer explicit engine, fallback to auto
            try:
                excel_engine = "openpyxl" if ext == "xlsx" else "xlrd"
                xls = pd.read_excel(tmp_path, sheet_name=None, engine=excel_engine, dtype=str)
            except Exception:
                xls = pd.read_excel(tmp_path, sheet_name=None, dtype=str)
            for sh, df in (xls or {}).items():
                if df is None:
                    continue
                df = _clean_dataframe(df)
                df = _convert_types(df)
                sheet_dfs[sh] = df
        else:  # csv
            try:
                df = pd.read_csv(tmp_path, dtype=str)
            except Exception:
                df = pd.read_csv(tmp_path, dtype=str, encoding="cp1258", sep=None, engine="python")
            df = _clean_dataframe(df)
            df = _convert_types(df)
            sheet_dfs["Sheet1"] = df
    except Exception as e:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        raise HTTPException(status_code=400, detail=f"Không đọc được dữ liệu: {e}")

    if not sheet_dfs:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        raise HTTPException(status_code=400, detail="Tệp rỗng hoặc không có sheet hợp lệ")

    # Build unified DataFrame by aligning columns (outer join semantics)
    import pandas as pd
    all_cols = set()
    for df in sheet_dfs.values():
        all_cols.update(list(df.columns))
    all_cols = list(all_cols)
    dfs_aligned = []
    for sh, df in sheet_dfs.items():
        d = df.copy()
        for c in all_cols:
            if c not in d.columns:
                d[c] = None
        d["__sheet__"] = sh
        dfs_aligned.append(d[all_cols + ["__sheet__"]])
    master_df = pd.concat(dfs_aligned, ignore_index=True)

    # Data quality report
    dqr = _data_quality_report(master_df)
    sheet_names = list(sheet_dfs.keys())

    # Prepare documents for embedding
    docs: list[str] = []
    metas: list[dict] = []

    # 1. Create Schema Summary Document
    schema_summary = (
        f"Tóm tắt cấu trúc file Excel '{file.filename}':\n"
        f"- Các sheet: {', '.join(sheet_names)}\n"
        f"- Tổng số dòng: {dqr['rows']}\n"
        f"- Các cột: {', '.join(dqr['columns'])}\n"
        f"- Kiểu dữ liệu (ước tính): {json.dumps(dqr['dtypes'], ensure_ascii=False, indent=2)}"
    )
    docs.append(schema_summary)
    metas.append({"type": "schema_summary", "source_file": file.filename})

    # 2. Create Documents from Rows (row-level for small files, group-chunk for large files)
    total_rows = sum(len(df) for df in sheet_dfs.values())
    max_row_level = int(os.getenv("EXCEL_ROW_LEVEL_MAX_ROWS", "600"))
    group_chunk = int(os.getenv("EXCEL_ROW_CHUNK_SIZE", "50"))

    mode = "row" if total_rows <= max_row_level else "group"
    print(f"[AdminExcel] Preparing docs: total_rows={total_rows}, mode={mode}")

    if mode == "row":
        for sheet_name, df_sheet in sheet_dfs.items():
            if df_sheet.empty: continue
            for index, row in df_sheet.iterrows():
                # Create the text content for this single row
                row_desc_parts = []
                for col, val in row.items():
                    if val is not None and str(val).strip() != "":
                        row_desc_parts.append(f"{col.replace('_', ' ')} là '{val}'")

                if not row_desc_parts:
                    continue # Skip empty rows

                row_text = f"Thông tin từ file {file.filename}, sheet '{sheet_name}', dòng {index + 2}:\n- {', '.join(row_desc_parts)}."
                docs.append(row_text)

                # Create metadata from ALL columns for this row
                row_meta = row.to_dict()
                # Clean up metadata: convert non-string/numeric types and remove NaNs
                cleaned_meta = {}
                for k, v in row_meta.items():
                    if pd.notna(v) and str(v).strip() != "":
                        cleaned_meta[k] = str(v) # Ensure all metadata values are strings for Chroma

                # Add file-level context to metadata
                cleaned_meta["type"] = "row_data"
                cleaned_meta["source_file"] = file.filename
                cleaned_meta["sheet_name"] = sheet_name
                cleaned_meta["row_number"] = str(index + 2)

                metas.append(cleaned_meta)
    else:
        # Group-chunking for large files to avoid long embedding time
        for sheet_name, df_sheet in sheet_dfs.items():
            if df_sheet.empty: continue
            start_idx = 0
            for start, end, df_group in _chunk_dataframe(df_sheet, chunk_size=group_chunk):
                # Convert chunk of rows to a summarized text
                chunk_text = _format_row_group_as_text(df_group, sheet_name, (start, end))
                docs.append(chunk_text)

                # Minimal metadata for the chunk
                metas.append({
                    "type": "row_group",
                    "source_file": file.filename,
                    "sheet_name": sheet_name,
                    "row_start": str(start + 2),  # +2 accounts for header row in Excel display
                    "row_end": str(end + 1)
                })

    if not docs:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        raise HTTPException(status_code=400, detail="Không tạo được nội dung để index")

    # Generate embeddings and store to Chroma
    try:
        embeddings: list[list[float]] = []
        backend = "openai" if (USE_OPENAI and openai_client) else ("st" if _st_model else "unknown")
        if USE_OPENAI and openai_client:
            # Batch embedding to avoid long waits and rate limits
            batch_size = int(os.getenv("OPENAI_EMBED_BATCH", "64"))
            for i in range(0, len(docs), batch_size):
                batch = docs[i:i+batch_size]
                r = openai_client.embeddings.create(model=OPENAI_EMBED_MODEL, input=batch)
                # OpenAI returns in the same order
                for item in r.data:
                    embeddings.append(item.embedding)
        elif _st_model is not None:
            embs = _st_model.encode(docs, convert_to_numpy=False)
            embeddings = [e.tolist() if hasattr(e, 'tolist') else list(e) for e in embs]
        else:
            raise Exception("No embeddings backend available; set USE_OPENAI=1 or install sentence-transformers")

        emb_dim = len(embeddings[0]) if embeddings else None
        collection_name = f"excel_data_{_uuid.uuid4().hex[:8]}"
        client = get_chroma_client()
        if not client:
            raise Exception("Chroma client is not available. Check ChromaDB connection.")

        # Directly create and use the specific collection for this Excel file.
        # This removes the complex and error-prone fallback logic.
        try:
            excel_coll = client.get_or_create_collection(name=collection_name)
            print(f"[AdminExcel] Successfully got or created collection: {collection_name}")
        except Exception as e:
            print(f"[AdminExcel] CRITICAL: Failed to get or create collection '{collection_name}': {e}")
            raise HTTPException(status_code=500, detail=f"Could not create vector collection: {e}")

        ids = [f"{i}_{_uuid.uuid4().hex[:12]}" for i in range(len(docs))]
        excel_coll.add(embeddings=embeddings, documents=docs, metadatas=metas, ids=ids)

    except Exception as e:
        print(f"[AdminExcel] Embedding/index error: {e}")
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"Embedding/index error: {e}")

    # Save metadata row (pre-generate id to reference from ExcelRow)
    ef_id = str(_uuid.uuid4())
    ef = ExcelFile(
        id=ef_id,
        filename=file.filename,
        file_path=tmp_path,
        file_size=size,
        sheet_names=json.dumps(sheet_names, ensure_ascii=False),
        rows_processed=int(dqr["rows"]),
        columns=json.dumps(dqr["columns"], ensure_ascii=False),
        data_types=json.dumps(dqr["dtypes"], ensure_ascii=False),
        missing_stats=json.dumps(dqr["missing"], ensure_ascii=False),
        collection_name=collection_name,
        uploaded_by=admin.id,
    )
    db.add(ef)
    db.commit(); db.refresh(ef)

    # Optionally persist structured rows into SQL for direct filtering
    try:
        sql_max = int(os.getenv("EXCEL_SQL_MAX_ROWS", "5000"))
    except Exception:
        sql_max = 5000

    rows_to_insert: list[ExcelRow] = []
    crm_to_insert: list[CrmProduct] = []
    total_added = 0

    for sheet_name, df_sheet in sheet_dfs.items():
        if df_sheet is None or df_sheet.empty:
            continue
        for idx, row in df_sheet.iterrows():
            row_dict = {str(k): (None if pd.isna(v) else v) for k, v in row.to_dict().items()}
            norm_text = _row_to_norm_text(row_dict)
            rows_to_insert.append(ExcelRow(
                excel_file_id=ef_id,
                source_filename=file.filename,
                sheet_name=sheet_name,
                row_index=int(idx) + 2,
                data_json=json.dumps(row_dict, ensure_ascii=False),
                normalized_text=norm_text,
            ))
            # Heuristic CRM extraction
            crm = _extract_crm_from_row(row_dict)
            if crm:
                crm_to_insert.append(CrmProduct(
                    source_filename=file.filename,
                    row_index=int(idx) + 2,
                    sku=crm.get("sku"),
                    name=crm.get("name"),
                    price=crm.get("price"),
                    currency=crm.get("currency"),
                    category=crm.get("category"),
                    description=crm.get("description"),
                    attributes=json.dumps(row_dict, ensure_ascii=False),
                ))
            total_added += 1
            if total_added >= sql_max:
                break
        if total_added >= sql_max:
            break

    # Commit structured data first
    try:
        if rows_to_insert:
            db.bulk_save_objects(rows_to_insert)
        if crm_to_insert:
            db.bulk_save_objects(crm_to_insert)
        if rows_to_insert or crm_to_insert:
            db.commit()
            print(f"[AdminExcel] Committed {len(rows_to_insert)} ExcelRow and {len(crm_to_insert)} CrmProduct objects.")
    except Exception as se:
        print(f"[AdminExcel] CRITICAL: Could not persist structured rows: {se}")
        db.rollback()
        # Do not proceed to FTS sync if commit failed
        return ExcelUploadResponse(
            id=ef.id, filename=ef.filename, file_size=ef.file_size, sheet_names=sheet_names,
            rows_processed=ef.rows_processed, columns=json.loads(ef.columns),
            data_types=json.loads(ef.data_types), missing_stats=json.loads(ef.missing_stats),
            collection_name=ef.collection_name, uploaded_at=ef.uploaded_at,
        )

    # Sync with FTS tables after successful commit
    if rows_to_insert or crm_to_insert:
        try:
            with engine.begin() as conn:
                if rows_to_insert:
                    conn.execute(text("INSERT INTO excel_rows_fts(id, normalized_text, source_filename, sheet_name) SELECT id, normalized_text, source_filename, sheet_name FROM excel_rows WHERE id NOT IN (SELECT id FROM excel_rows_fts);"))
                if crm_to_insert:
                    conn.execute(text("INSERT INTO crm_products_fts(id, sku, name, category, description, source_filename) SELECT id, sku, name, category, description, source_filename FROM crm_products WHERE id NOT IN (SELECT id FROM crm_products_fts);"))
            print("[AdminExcel] Synced new rows to FTS tables.")
        except Exception as fts_e:
            print(f"[AdminExcel] Warning: FTS sync failed: {fts_e}")


    return ExcelUploadResponse(
        id=ef.id,
        filename=ef.filename,
        file_size=ef.file_size,
        sheet_names=sheet_names,
        rows_processed=ef.rows_processed,
        columns=json.loads(ef.columns),
        data_types=json.loads(ef.data_types),
        missing_stats=json.loads(ef.missing_stats),
        collection_name=ef.collection_name,
        uploaded_at=ef.uploaded_at,
    )


class ExcelListItem(BaseModel):
    id: str
    filename: str
    file_size: int
    uploaded_at: datetime
    sheet_names: list[str]
    rows_processed: int
    collection_name: str


@router.get("/excel-files", response_model=list[ExcelListItem])
async def list_excel_files(admin: User = Depends(verify_admin), db: Session = Depends(get_db)):
    rows = db.query(ExcelFile).order_by(ExcelFile.uploaded_at.desc()).all()
    res: list[ExcelListItem] = []
    import json
    for r in rows:
        try:
            res.append(ExcelListItem(
                id=r.id,
                filename=r.filename,
                file_size=r.file_size,
                uploaded_at=r.uploaded_at,
                sheet_names=json.loads(r.sheet_names or "[]"),
                rows_processed=r.rows_processed or 0,
                collection_name=r.collection_name or "",
            ))
        except Exception:
            continue
    return res


@router.delete("/excel-files/{file_id}")
async def delete_excel_file(file_id: str, admin: User = Depends(verify_admin), db: Session = Depends(get_db)):
    row = db.query(ExcelFile).filter(ExcelFile.id == file_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Excel file not found")
    # Delete Chroma collection
    try:
        client = get_chroma_client()
        if client and row.collection_name:
            try:
                client.delete_collection(row.collection_name)
            except Exception as _e:
                print(f"[AdminExcel] Delete collection failed: {_e}")
    except Exception as _e:
        print(f"[AdminExcel] Chroma client error: {_e}")
    # Delete file
    try:
        if row.file_path and os.path.exists(row.file_path):
            os.remove(row.file_path)
    except Exception:
        pass
    db.delete(row)
    db.commit()
    return {"message": "Deleted"}


@router.get("/charts/chat-activity")
async def get_chat_activity_chart(
    admin: User = Depends(verify_admin),
    db: Session = Depends(get_db)
):
    # Get chat activity for the last 7 days
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=7)

    # Query daily chat counts
    daily_chats = db.query(
        func.date(ChatMessage.timestamp).label('date'),
        func.count(ChatMessage.id).label('count')
    ).filter(
        ChatMessage.timestamp >= start_date
    ).group_by(
        func.date(ChatMessage.timestamp)
    ).all()

    # Create chart
    dates = [str(chat.date) for chat in daily_chats]
    counts = [chat.count for chat in daily_chats]

    plt.figure(figsize=(10, 6))
    plt.plot(dates, counts, marker='o')
    plt.title('Chat Activity - Last 7 Days')
    plt.xlabel('Date')
    plt.ylabel('Number of Messages')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save to bytes
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    img_str = base64.b64encode(img_buffer.getvalue()).decode()
    plt.close()

    return {"chart_data": img_str}

# ---------------- Branding (Logo/Icon & Theme) ----------------
class BrandingResponse(BaseModel):
    brand_logo_url: Optional[str] = None
    brand_logo_height: Optional[str] = None
    brand_name: Optional[str] = None
    primary_color: Optional[str] = None
    favicon_url: Optional[str] = None

class BrandingConfigUpdate(BaseModel):
    brand_logo_url: Optional[str] = None
    brand_logo_height: Optional[str] = None
    brand_name: Optional[str] = None
    primary_color: Optional[str] = None
    favicon_url: Optional[str] = None


def _get_config(db: Session, key: str) -> Optional[str]:
    row = db.query(SystemConfig).filter(SystemConfig.key == key).first()
    return row.value if row else None


def _set_config(db: Session, key: str, value: str, admin_id: str):
    row = db.query(SystemConfig).filter(SystemConfig.key == key).first()
    now = datetime.utcnow()
    if row:
        row.value = value
        row.updated_at = now
    else:
        row = SystemConfig(key=key, value=value, description=f"Branding {key}")
        db.add(row)
    db.commit()


@router.get("/branding", response_model=BrandingResponse)
async def get_branding(admin: User = Depends(verify_admin), db: Session = Depends(get_db)):
    return BrandingResponse(
        brand_logo_url=_get_config(db, "BRAND_LOGO_URL"),
        brand_logo_height=_get_config(db, "BRAND_LOGO_HEIGHT"),
        brand_name=_get_config(db, "BRAND_NAME"),
        primary_color=_get_config(db, "BRAND_PRIMARY_COLOR"),
        favicon_url=_get_config(db, "FAVICON_URL"),
    )


@router.get("/branding/public", response_model=BrandingResponse)
async def get_branding_public(db: Session = Depends(get_db)):
    return BrandingResponse(
        brand_logo_url=_get_config(db, "BRAND_LOGO_URL"),
        brand_logo_height=_get_config(db, "BRAND_LOGO_HEIGHT"),
        brand_name=_get_config(db, "BRAND_NAME"),
        primary_color=_get_config(db, "BRAND_PRIMARY_COLOR"),
        favicon_url=_get_config(db, "FAVICON_URL"),
    )


@router.put("/branding", response_model=BrandingResponse)
async def update_branding(cfg: BrandingConfigUpdate, admin: User = Depends(verify_admin), db: Session = Depends(get_db)):
    if cfg.brand_logo_url is not None:
        _set_config(db, "BRAND_LOGO_URL", cfg.brand_logo_url, admin.id)
    if cfg.brand_logo_height is not None:
        _set_config(db, "BRAND_LOGO_HEIGHT", cfg.brand_logo_height, admin.id)
    if cfg.brand_name is not None:
        _set_config(db, "BRAND_NAME", cfg.brand_name, admin.id)
    if cfg.primary_color is not None:
        _set_config(db, "BRAND_PRIMARY_COLOR", cfg.primary_color, admin.id)
    if cfg.favicon_url is not None:
        _set_config(db, "FAVICON_URL", cfg.favicon_url, admin.id)
    return await get_branding(admin, db)


@router.post("/branding/upload-logo")
async def upload_brand_logo(
    file: UploadFile = File(...),
    admin: User = Depends(verify_admin),
    db: Session = Depends(get_db)
):
    """Upload a logo/icon image for header. Accepts png/jpg/jpeg/svg. Max 2MB.
    Stores under /static/branding and updates SystemConfig.BRAND_LOGO_URL
    """
    import uuid as _uuid
    ALLOWED = {"image/png", "image/jpeg", "image/svg+xml"}
    max_bytes = 2 * 1024 * 1024
    ctype = (file.content_type or "").lower()
    if ctype not in ALLOWED:
        raise HTTPException(status_code=400, detail=f"Loại file không hỗ trợ: {ctype}. Hỗ trợ: PNG, JPG, SVG")

    # Save to disk with size limit
    os.makedirs(os.path.join("static", "branding"), exist_ok=True)
    safe_name = f"{_uuid.uuid4().hex}_{(file.filename or 'logo').replace('..','').replace('/', '_')}"
    out_path = os.path.join("static", "branding", safe_name)

    size = 0
    try:
        with open(out_path, "wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > max_bytes:
                    f.close()
                    try:
                        os.remove(out_path)
                    except Exception:
                        pass
                    raise HTTPException(status_code=413, detail="File quá lớn (tối đa 2MB)")
                f.write(chunk)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Không lưu được file: {e}")

    url = f"/static/branding/{safe_name}"
    _set_config(db, "BRAND_LOGO_URL", url, admin.id)
    return {"url": url}


@router.post("/branding/upload-favicon")
async def upload_favicon(
    file: UploadFile = File(...),
    admin: User = Depends(verify_admin),
    db: Session = Depends(get_db)
):
    """Upload a favicon. Accepts ico/png/svg. Max 1MB.
    Stores under /static/branding and updates SystemConfig.FAVICON_URL
    """
    import uuid as _uuid
    ALLOWED = {"image/x-icon", "image/png", "image/svg+xml"}
    max_bytes = 1 * 1024 * 1024 # 1MB
    ctype = (file.content_type or "").lower()
    if ctype not in ALLOWED:
        raise HTTPException(status_code=400, detail=f"Loại file không hỗ trợ: {ctype}. Hỗ trợ: ICO, PNG, SVG")

    os.makedirs(os.path.join("static", "branding"), exist_ok=True)
    safe_name = f"{_uuid.uuid4().hex}_{(file.filename or 'favicon').replace('..','').replace('/', '_')}"
    out_path = os.path.join("static", "branding", safe_name)

    size = 0
    try:
        with open(out_path, "wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > max_bytes:
                    f.close()
                    try:
                        os.remove(out_path)
                    except Exception:
                        pass
                    raise HTTPException(status_code=413, detail="File quá lớn (tối đa 1MB)")
                f.write(chunk)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Không lưu được file: {e}")

    url = f"/static/branding/{safe_name}"
    _set_config(db, "FAVICON_URL", url, admin.id)
    return {"url": url}


    url = f"/static/branding/{safe_name}"
    _set_config(db, "BRAND_LOGO_URL", url, admin.id)
    return {"url": url}
