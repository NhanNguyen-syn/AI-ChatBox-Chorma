from sqlalchemy import create_engine, Column, String, Integer, Boolean, DateTime, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
# NOTE: Avoid importing chromadb at module import time to prevent heavy optional deps from loading
import os

from uuid import uuid4

# SQLite database
SQLALCHEMY_DATABASE_URL = "sqlite:///./chroma_chat.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Database models
class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid4()))
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    full_name = Column(String, nullable=True)
    phone = Column(String, nullable=True)

    # Role/Department/Status
    role = Column(String, default="user")            # 'admin' | 'manager' | 'user' | ...
    department = Column(String, nullable=True)        # e.g., IT, Sales, Marketing
    account_status = Column(String, default="active")  # 'active' | 'inactive' | 'suspended'
    last_login = Column(DateTime, nullable=True)

    is_admin = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)
    token_quota = Column(Integer, nullable=True, default=100000)  # Monthly token quota
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(String, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("users.id"))
    title = Column(String)
    # Rolling summary of the conversation for long context handling
    summary = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", back_populates="chat_sessions")
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")

class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(String, primary_key=True, index=True)
    session_id = Column(String, ForeignKey("chat_sessions.id"))
    message = Column(Text)
    response = Column(Text)
    is_user = Column(Boolean, default=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    # Real metrics
    tokens_used = Column(Integer, default=0)
    response_time = Column(Integer, default=0)  # milliseconds
    # For AI messages, this is a JSON string of the sources used.
    sources = Column(Text, nullable=True)

class Document(Base):
    __tablename__ = "documents"

    id = Column(String, primary_key=True, index=True)
    filename = Column(String)
    file_path = Column(String)
    file_type = Column(String)
    file_size = Column(Integer)
    file_url = Column(String, nullable=True)
    status = Column(String, default="pending")  # pending, completed, failed

    uploaded_by = Column(String, ForeignKey("users.id"))
    uploaded_at = Column(DateTime, default=datetime.utcnow)

    uploader = relationship("User")
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")



class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid4()))
    document_id = Column(String, ForeignKey("documents.id"), index=True)
    chunk_id = Column(String, index=True)
    collection = Column(String, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    document = relationship("Document", back_populates="chunks")

class FAQ(Base):
    __tablename__ = "faqs"

    id = Column(String, primary_key=True, index=True)
    question = Column(String)
    answer = Column(Text)
    created_by = Column(String, ForeignKey("users.id"))
    category = Column(String, nullable=True, default='General')
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)

class OcrText(Base):
    __tablename__ = "ocr_texts"

    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid4()))
    document_id = Column(String, ForeignKey("documents.id"), index=True)
    source_filename = Column(String, index=True)
    page = Column(Integer, nullable=True)
    chunk_index = Column(Integer, nullable=True)
    section = Column(String)                # e.g., "I. Chính sách > Phụ cấp"
    block_type = Column(String)             # e.g., 'text' | 'table'
    ocr_confidence = Column(Integer)        # average OCR confidence 0-100 (if available)
    content = Column(Text)
    # Accent-stripped lowercase copy for robust LIKE searching
    normalized_content = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

    document = relationship("Document")

class SystemConfig(Base):
    __tablename__ = "system_configs"

    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid4()))
    key = Column(String, unique=True, index=True)
    value = Column(Text)
    description = Column(String)
    updated_by = Column(String, ForeignKey("users.id"), nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class CrmProduct(Base):
    __tablename__ = "crm_products"

    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid4()))
    source_filename = Column(String, index=True)
    row_index = Column(Integer)
    sku = Column(String, index=True)
    name = Column(String, index=True)
    price = Column(String)
    currency = Column(String)
    category = Column(String)
    description = Column(Text)
    attributes = Column(Text)  # JSON string of the full row
    created_at = Column(DateTime, default=datetime.utcnow)


class ExcelFile(Base):
    __tablename__ = "excel_files"

    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid4()))
    filename = Column(String)
    file_path = Column(String)
    file_size = Column(Integer)
    sheet_names = Column(Text)  # JSON string list
    rows_processed = Column(Integer, default=0)
    columns = Column(Text)      # JSON string list
    data_types = Column(Text)   # JSON string mapping col -> dtype
    missing_stats = Column(Text)  # JSON string mapping col -> missing count/ratio
    collection_name = Column(String)
    uploaded_by = Column(String, ForeignKey("users.id"))
    uploaded_at = Column(DateTime, default=datetime.utcnow)


class ExcelRow(Base):
    __tablename__ = "excel_rows"

    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid4()))
    excel_file_id = Column(String, ForeignKey("excel_files.id"), index=True)
    source_filename = Column(String, index=True)
    sheet_name = Column(String, index=True)
    row_index = Column(Integer)
    # Full original row as JSON string (after cleaning/standardization)
    data_json = Column(Text)
    # A flattened, ASCII-normalized searchable text of the row
    normalized_text = Column(Text, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class TemporaryContext(Base):
    __tablename__ = "temporary_contexts"

    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid4()))
    session_id = Column(String, ForeignKey("chat_sessions.id"), index=True)
    user_id = Column(String, ForeignKey("users.id"), index=True)
    filename = Column(String)
    file_type = Column(String)
    file_size = Column(Integer)
    file_url = Column(String, nullable=True)  # Cloud/local URL for download/display
    summary = Column(String)
    content = Column(Text)  # Extracted text used as temporary context
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    expires_at = Column(DateTime, index=True)


class Feedback(Base):
    __tablename__ = "feedbacks"

    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid4()))
    chat_message_id = Column(String, ForeignKey("chat_messages.id"))
    user_id = Column(String, ForeignKey("users.id"))
    rating = Column(Integer)  # e.g., 1 for like, -1 for dislike
    comment = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class IgnoredQuestion(Base):
    __tablename__ = 'ignored_questions'
    id = Column(Integer, primary_key=True, index=True)
    question = Column(String, unique=True, index=True, nullable=False)
    ignored_at = Column(DateTime, default=datetime.utcnow)
    # match User.id which is a String UUID
    ignored_by_id = Column(String, ForeignKey('users.id'))

    ignored_by = relationship("User")



class QaFeedback(Base):
    __tablename__ = "qa_feedbacks"

    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid4()))
    session_id = Column(String, ForeignKey("chat_sessions.id"), index=True)
    message_id = Column(String, ForeignKey("chat_messages.id"), index=True)
    rating = Column(Integer)  # 1..5
    comment = Column(Text)
    question = Column(Text)
    answer = Column(Text)
class TokenUsage(Base):
    __tablename__ = "token_usage"

    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid4()))
    user_id = Column(String, ForeignKey("users.id"), index=True, nullable=False)
    tokens_used = Column(Integer, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

    user = relationship("User")

    created_at = Column(DateTime, default=datetime.utcnow)


# Relationships
User.chat_sessions = relationship("ChatSession", back_populates="user")
ChatSession.messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")
ChatMessage.session = relationship("ChatSession", back_populates="messages")

# Database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ChromaDB setup

def _current_collection_name() -> str:
    """Return a single, common collection name for all embeddings backends.
    Override via CHROMA_COLLECTION env; default to 'documents'.
    """
    name = os.getenv("CHROMA_COLLECTION", "documents").strip() or "documents"
    # sanitize
    return name.replace(":", "_").replace(" ", "_")


def get_chroma_collection():
    """Get or create ChromaDB collection bound to current backend"""
    try:
        client = get_chroma_client()
        if not client:
            return None
        name = _current_collection_name()
        collection = client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"}
        )
        return collection
    except Exception as e:
        print(f"Warning: Could not initialize ChromaDB: {e}")
        return None

# --- Extended Chroma helpers to support per-backend collections and multi-collection queries ---

def _collection_base_name() -> str:
    """Base collection name. Override via CHROMA_COLLECTION (default: 'documents').
    We'll suffix per embeddings backend to avoid vector-dimension mismatches.
    """
    name = os.getenv("CHROMA_COLLECTION", "documents").strip() or "documents"
    return name.replace(":", "_").replace(" ", "_")


def _collection_name_for_backend(backend: str, dim: int | None = None) -> str:
    base = _collection_base_name()
    suffix = (backend or "unknown").strip().lower().replace(":", "_").replace(" ", "_")
    return f"{base}_{suffix}_{dim}" if dim else f"{base}_{suffix}"


# --- ChromaDB Singleton Client ---
_chroma_client = None

def get_chroma_client():
    """Get a singleton chromadb client to ensure consistency across the app."""
    global _chroma_client
    if _chroma_client is None:
        try:
            import chromadb
            from chromadb.config import Settings
            print("[Database] Initializing ChromaDB Persistent Client with Telemetry OFF...")
            # Store data on disk and disable telemetry
            _chroma_client = chromadb.PersistentClient(
                path="./chroma_db",
                settings=Settings(anonymized_telemetry=False)
            )
        except Exception as e:
            print(f"CRITICAL: Could not connect to ChromaDB: {e}")
            return None
    return _chroma_client


def get_chroma_collection_for_backend(backend: str, dim: int | None = None):
    """Get or create a collection specific to the embeddings backend (and dimension)."""
    try:
        client = get_chroma_client()
        if not client:
            return None
        name = _collection_name_for_backend(backend, dim)
        coll = client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})
        return coll
    except Exception as e:
        print(f"Warning: Could not init backend collection '{backend}': {e}")
        return None


def get_all_chroma_collections():
    """Return all collections relevant to the app, including base prefix and Excel datasets.
    - Base prefix: CHROMA_COLLECTION (default 'documents')
    - Excel datasets: names starting with 'excel_data_'
    """
    try:
        client = get_chroma_client()
        if not client:
            return []
        base = _collection_base_name()
        cols = []
        for c in client.list_collections() or []:
            try:
                nm = getattr(c, "name", "")
                if nm.startswith(base) or nm.startswith("excel_data_"):
                    cols.append(c)
            except Exception:
                continue
        return cols
    except Exception as e:
        print(f"Warning: Could not list Chroma collections: {e}")
        return []

class AllowanceTable(Base):
    __tablename__ = "allowance_tables"

    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid4()))
    source_filename = Column(String, index=True)
    page = Column(Integer)
    khu_vuc = Column(String, index=True)
    phu_cap_cu = Column(Integer)
    phu_cap_moi = Column(Integer)
    muc_tang = Column(Integer)
    # For traceability
    raw_row_text = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

