from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
from dotenv import load_dotenv
import os

from routers import auth, chat, admin, files, users, feedback
from services.middleware import RateLimitMiddleware, LoggingMiddleware


from database import engine, Base
from auth.jwt_handler import verify_token

load_dotenv()

app = FastAPI(title="Chroma AI Chat System", version="1.0.0")

# Global HTTP logging and rate limiting (10 req/min per user/IP)
app.add_middleware(LoggingMiddleware)
app.add_middleware(RateLimitMiddleware, requests=int(os.getenv("RATE_LIMIT_RPM", "120")), window_seconds=60)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://localhost:3002", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create database tables
Base.metadata.create_all(bind=engine)
# Lightweight migration: add missing columns and optional FTS5 index
from sqlalchemy import inspect, text


def ensure_migrations():
    try:
        inspector = inspect(engine)
        # Chat messages metrics
        if inspector.has_table('chat_messages'):
            cols = [c['name'] for c in inspector.get_columns('chat_messages')]
            print(f"[Migration] chat_messages columns: {cols}")
            with engine.begin() as conn:  # Use begin() for auto-commit
                if 'tokens_used' not in cols:
                    print("[Migration] Adding tokens_used column...")
                    conn.execute(text("ALTER TABLE chat_messages ADD COLUMN tokens_used INTEGER DEFAULT 0"))
                if 'response_time' not in cols:
                    print("[Migration] Adding response_time column...")
                    conn.execute(text("ALTER TABLE chat_messages ADD COLUMN response_time INTEGER DEFAULT 0"))
        # Chat sessions: rolling summary column
        if inspector.has_table('chat_sessions'):
            cols = [c['name'] for c in inspector.get_columns('chat_sessions')]
            with engine.begin() as conn:
                if 'summary' not in cols:
                    print("[Migration] Adding summary to chat_sessions...")
                    conn.execute(text("ALTER TABLE chat_sessions ADD COLUMN summary TEXT"))
        # OCR texts additional columns
        if inspector.has_table('ocr_texts'):
            cols = [c['name'] for c in inspector.get_columns('ocr_texts')]
            with engine.begin() as conn:
                if 'normalized_content' not in cols:
                    print("[Migration] Adding normalized_content to ocr_texts...")
                    conn.execute(text("ALTER TABLE ocr_texts ADD COLUMN normalized_content TEXT"))
                if 'section' not in cols:
                    print("[Migration] Adding section to ocr_texts...")
                    conn.execute(text("ALTER TABLE ocr_texts ADD COLUMN section VARCHAR"))
                if 'block_type' not in cols:
                    print("[Migration] Adding block_type to ocr_texts...")
                    conn.execute(text("ALTER TABLE ocr_texts ADD COLUMN block_type VARCHAR"))
                if 'ocr_confidence' not in cols:
                    print("[Migration] Adding ocr_confidence to ocr_texts...")
                    conn.execute(text("ALTER TABLE ocr_texts ADD COLUMN ocr_confidence INTEGER"))
        # Optional backfill for normalized_content (best-effort)
        try:
            from database import SessionLocal, OcrText
            import unicodedata, re
            def _norm(s: str) -> str:
                try:
                    s = unicodedata.normalize('NFKD', s or '')
                    s = ''.join([c for c in s if not unicodedata.combining(c)])
                    s = s.lower()
                    s = re.sub(r"\s+", " ", s)
                    return s.strip()
                except Exception:
                    return (s or '').lower().strip()
            db = SessionLocal()
            try:
                rows = db.query(OcrText).filter((OcrText.normalized_content == None) | (OcrText.normalized_content == "")).limit(2000).all()
                changed = 0
                for r in rows:
                    r.normalized_content = _norm(r.content or '')
                    changed += 1
                if changed:
                    db.commit()
                    print(f"[Migration] Backfilled normalized_content for {changed} ocr_texts rows")
            finally:
                db.close()
        except Exception as be:
            print(f"[Migration] Backfill warning: {be}")
        # Optional FTS5 virtual table for faster SQL full-text (best-effort)
        try:
            with engine.begin() as conn:
                # Include 'id' so we can map back to the base table reliably
                conn.execute(text("CREATE VIRTUAL TABLE IF NOT EXISTS ocr_texts_fts USING fts5(id, content, normalized_content, source_filename, document_id);"))
                # Backfill
                conn.execute(text("INSERT INTO ocr_texts_fts(id, content, normalized_content, source_filename, document_id) SELECT id, content, normalized_content, source_filename, document_id FROM ocr_texts WHERE content IS NOT NULL AND content != '' AND id NOT IN (SELECT id FROM ocr_texts_fts)"))
                print("[Migration] Ensured FTS5 table ocr_texts_fts (with id) and backfilled")
        except Exception as fe:
            print(f"[Migration] FTS5 not available or init failed: {fe}")

        # Documents/TemporaryContext schema updates
        try:
            if inspector.has_table('documents'):
                cols = [c['name'] for c in inspector.get_columns('documents')]
                with engine.begin() as conn:
                    if 'file_url' not in cols:
                        print("[Migration] Adding file_url to documents...")
                        conn.execute(text("ALTER TABLE documents ADD COLUMN file_url TEXT"))
                    if 'status' not in cols:
                        print("[Migration] Adding status to documents...")
                        conn.execute(text("ALTER TABLE documents ADD COLUMN status VARCHAR(20) DEFAULT 'pending'"))
                    # Useful indexes
                    conn.execute(text("CREATE INDEX IF NOT EXISTS idx_documents_uploaded_at ON documents(uploaded_at)"))
                    conn.execute(text("CREATE INDEX IF NOT EXISTS idx_documents_filename ON documents(filename)"))
            if inspector.has_table('temporary_contexts'):
                cols = [c['name'] for c in inspector.get_columns('temporary_contexts')]
                with engine.begin() as conn:
                    if 'file_url' not in cols:
                        print("[Migration] Adding file_url to temporary_contexts...")
                        conn.execute(text("ALTER TABLE temporary_contexts ADD COLUMN file_url TEXT"))
                    if 'status' not in cols:
                        print("[Migration] Adding status to documents...")
                        conn.execute(text("ALTER TABLE documents ADD COLUMN status VARCHAR(20) DEFAULT 'pending'"))
                    conn.execute(text("CREATE INDEX IF NOT EXISTS idx_temp_ctx_session_created ON temporary_contexts(session_id, created_at)"))
            # Add token_quota to users table
            if inspector.has_table('users'):
                cols = [c['name'] for c in inspector.get_columns('users')]
                with engine.begin() as conn:
                    if 'token_quota' not in cols:
                        print("[Migration] Adding token_quota to users...")
                        conn.execute(text("ALTER TABLE users ADD COLUMN token_quota INTEGER DEFAULT 100000"))
                    if 'phone' not in cols:
                        print("[Migration] Adding phone to users...")
                        conn.execute(text("ALTER TABLE users ADD COLUMN phone VARCHAR"))
                    if 'role' not in cols:
                        print("[Migration] Adding role to users...")
                        conn.execute(text("ALTER TABLE users ADD COLUMN role VARCHAR DEFAULT 'user'"))
                    if 'department' not in cols:
                        print("[Migration] Adding department to users...")
                        conn.execute(text("ALTER TABLE users ADD COLUMN department VARCHAR"))
                    if 'account_status' not in cols:
                        print("[Migration] Adding account_status to users...")
                        conn.execute(text("ALTER TABLE users ADD COLUMN account_status VARCHAR DEFAULT 'active'"))
                    if 'last_login' not in cols:
                        print("[Migration] Adding last_login to users...")
                        conn.execute(text("ALTER TABLE users ADD COLUMN last_login DATETIME"))
                # Ensure index on temporary_contexts.expires_at
                with engine.begin() as conn:
                    conn.execute(text("CREATE INDEX IF NOT EXISTS idx_temp_ctx_expires ON temporary_contexts(expires_at)"))
            # Chat message pagination index
            if inspector.has_table('chat_messages'):
                with engine.begin() as conn:
                    conn.execute(text("CREATE INDEX IF NOT EXISTS idx_chat_messages_session_ts ON chat_messages(session_id, timestamp)"))
            # Feedbacks: ensure created_at exists for admin analytics
            if inspector.has_table('feedbacks'):
                cols = [c['name'] for c in inspector.get_columns('feedbacks')]
                if 'created_at' not in cols:
                    try:
                        print("[Migration] Adding created_at to feedbacks...")
                        with engine.begin() as conn:
                            conn.execute(text("ALTER TABLE feedbacks ADD COLUMN created_at DATETIME DEFAULT CURRENT_TIMESTAMP"))
                    except Exception as fe:
                        print(f"[Migration] Could not add created_at to feedbacks: {fe}")
        except Exception as e:
            print(f"[Migration] Schema/index patch failed: {e}")

        print("[Migration] Migration completed successfully")
        # FTS5 for ExcelRow
        try:
            with engine.begin() as conn:
                conn.execute(text("CREATE VIRTUAL TABLE IF NOT EXISTS excel_rows_fts USING fts5(id, normalized_text, source_filename, sheet_name);"))
                conn.execute(text("INSERT INTO excel_rows_fts(id, normalized_text, source_filename, sheet_name) SELECT id, normalized_text, source_filename, sheet_name FROM excel_rows WHERE normalized_text IS NOT NULL AND id NOT IN (SELECT id FROM excel_rows_fts);"))
                print("[Migration] Ensured FTS5 table excel_rows_fts and backfilled.")
        except Exception as fe:
            print(f"[Migration] FTS5 for excel_rows failed: {fe}")

        # FTS5 for CrmProduct
        try:
            with engine.begin() as conn:
                conn.execute(text("CREATE VIRTUAL TABLE IF NOT EXISTS crm_products_fts USING fts5(id, sku, name, category, description, source_filename);"))
                conn.execute(text("INSERT INTO crm_products_fts(id, sku, name, category, description, source_filename) SELECT id, sku, name, category, description, source_filename FROM crm_products WHERE id NOT IN (SELECT id FROM crm_products_fts);"))
                print("[Migration] Ensured FTS5 table crm_products_fts and backfilled.")
        except Exception as fe:
            print(f"[Migration] FTS5 for crm_products failed: {fe}")

    except Exception as e:
        print(f"[Migration] Warning: {e}")

ensure_migrations()


# Security
security = HTTPBearer()

# Dependency to verify JWT token
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = verify_token(credentials.credentials)
        return payload
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

# Bootstrap admin user if requested
from auth.jwt_handler import get_password_hash
from database import SessionLocal, User
from database import FAQ
import os

def bootstrap_admin_if_needed():
    if os.getenv("BOOTSTRAP_ADMIN", "0") != "1":
        return
    db = SessionLocal()
    try:
        # If there is already an admin, skip
        existing_admin = db.query(User).filter(User.is_admin == True).first()
        if existing_admin:
            print("[Bootstrap] Admin already exists, skipping.")
            return
        username = os.getenv("ADMIN_USERNAME", "admin")
        email = os.getenv("ADMIN_EMAIL", "admin@example.com")
        password = os.getenv("ADMIN_PASSWORD")
        if not password:
            print("[Bootstrap] ADMIN_PASSWORD is not set. Skip creating admin.")
            return
        # If username exists, upgrade to admin; otherwise create
        user = db.query(User).filter(User.username == username).first()
        if user:
            user.email = email
            user.hashed_password = get_password_hash(password)
            user.is_admin = True
            user.is_active = True
            action = "upgraded existing user to admin"
        else:
            user = User(
                username=username,
                email=email,
                hashed_password=get_password_hash(password),
                is_admin=True,
                is_active=True,
            )
            db.add(user)
            action = "created admin user"
        db.commit()
        print(f"[Bootstrap] {action}: '{username}' ({email}).")
    finally:
        db.close()

bootstrap_admin_if_needed()
import uuid

def seed_initial_data():
    db = SessionLocal()
    try:
        faq_question = "Dalat Hasfarm được thành lập khi nào?"
        existing_faq = db.query(FAQ).filter(FAQ.question.ilike(f'%{faq_question}%')).first()

        if not existing_faq:
            admin_user = db.query(User).filter(User.is_admin == True).order_by(User.created_at).first()
            if admin_user:
                new_faq = FAQ(
                    id=str(uuid.uuid4()),
                    question=faq_question,
                    answer="Dalat Hasfarm được thành lập vào ngày 7 tháng 6 năm 1994.",
                    category="Thông tin chung",
                    created_by=admin_user.id,
                    is_active=True,
                )
                db.add(new_faq)
                db.commit()
                print(f"[Seeding] Added initial FAQ: '{faq_question}'")
            else:
                print("[Seeding] Could not add initial FAQ, no admin user found.")
    except Exception as e:
        print(f"[Seeding] Error while seeding data: {e}")
    finally:
        db.close()

seed_initial_data()

# Include routers
app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(chat.router, prefix="/api/chat", tags=["Chat"])
app.include_router(admin.router, prefix="/api/admin", tags=["Admin"])
app.include_router(files.router, prefix="/api/files", tags=["Files"])
app.include_router(users.router, prefix="/api/users", tags=["Users"])
app.include_router(feedback.router, prefix="/api/feedback", tags=["Feedback"])

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return {"message": "Chroma AI Chat System API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)