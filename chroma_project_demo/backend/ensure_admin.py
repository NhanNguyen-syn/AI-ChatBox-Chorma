import os
import sys
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import User, SQLALCHEMY_DATABASE_URL as DATABASE_URL

def ensure_admin_privileges():
    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()

    try:
        # Find the user with the username 'admin'
        admin_user = db.query(User).filter(User.username == 'admin').first()

        if admin_user:
            # Ensure the user is an admin and is active
            admin_user.is_admin = True
            admin_user.is_active = True
            db.commit()
            print("Tài khoản 'admin' đã được cấp lại quyền quản trị viên và kích hoạt.")
        else:
            print("Không tìm thấy tài khoản có tên đăng nhập 'admin'. Vui lòng tạo tài khoản trước.")

    except Exception as e:
        print(f"Đã xảy ra lỗi: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    ensure_admin_privileges()

