import os
import sys
from sqlalchemy import create_engine, inspect, text

# Ensure backend path is importable when run directly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database import SessionLocal, Base, SQLALCHEMY_DATABASE_URL, User
from auth.jwt_handler import get_password_hash

def ensure_user_schema():
    """Ensure required columns exist on users table without damaging data."""
    engine = create_engine(SQLALCHEMY_DATABASE_URL)
    inspector = inspect(engine)
    # Create tables if not exist
    Base.metadata.create_all(bind=engine)
    cols = [c['name'] for c in inspector.get_columns('users')]
    with engine.begin() as conn:
        if 'full_name' not in cols:
            conn.execute(text("ALTER TABLE users ADD COLUMN full_name VARCHAR"))
            print("[reset_users] Added missing column users.full_name")
        if 'phone' not in cols:
            conn.execute(text("ALTER TABLE users ADD COLUMN phone VARCHAR"))
            print("[reset_users] Added missing column users.phone")
        if 'token_quota' not in cols:
            conn.execute(text("ALTER TABLE users ADD COLUMN token_quota INTEGER DEFAULT 100000"))
            print("[reset_users] Added missing column users.token_quota")

def reset_and_create_admin(staff_code: str, email: str, full_name: str, phone: str, password: str):
    ensure_user_schema()
    db = SessionLocal()
    try:
        # Delete all users
        deleted = db.query(User).delete()
        db.commit()
        print(f"[reset_users] Deleted {deleted} existing user(s)")

        # Create the exclusive admin
        admin = User(
            username=staff_code,
            email=email,
            full_name=full_name,
            phone=phone,
            hashed_password=get_password_hash(password),
            is_admin=True,
            is_active=True,
        )
        db.add(admin)
        db.commit()
        db.refresh(admin)
        print("[reset_users] Created admin user successfully")
        print({
            'username': admin.username,
            'email': admin.email,
            'full_name': admin.full_name,
            'phone': admin.phone,
        })
    finally:
        db.close()

if __name__ == "__main__":
    # Values can be changed or passed via env in future
    reset_and_create_admin(
        staff_code=os.getenv('ADMIN_STAFF_CODE', 'SG0510'),
        email=os.getenv('ADMIN_EMAIL', 'nhannhan05102004@gmail.com'),
        full_name=os.getenv('ADMIN_FULL_NAME', 'Nguyễn Phước Nhân'),
        phone=os.getenv('ADMIN_PHONE', '0941610627'),
        password=os.getenv('ADMIN_PASSWORD', 'admin123'),
    )

