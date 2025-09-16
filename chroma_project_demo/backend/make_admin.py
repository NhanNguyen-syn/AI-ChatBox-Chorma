import sys
from database import SessionLocal, User
from auth.jwt_handler import get_password_hash

"""
Usage:
  python make_admin.py <username> <email> <password>
If the user exists, they will be upgraded to admin and password will be reset.
If the user does not exist, they will be created as admin.
"""

def main():
    if len(sys.argv) < 4:
        print("Usage: python make_admin.py <username> <email> <password>")
        sys.exit(1)

    username = sys.argv[1]
    email = sys.argv[2]
    password = sys.argv[3]

    db = SessionLocal()
    try:
        u = db.query(User).filter(User.username == username).first()
        if u:
            u.email = email
            u.hashed_password = get_password_hash(password)
            u.is_admin = True
            u.is_active = True
            action = "updated existing user to admin"
        else:
            u = User(
                username=username,
                email=email,
                hashed_password=get_password_hash(password),
                is_admin=True,
                is_active=True,
            )
            db.add(u)
            action = "created new admin user"
        db.commit()
        print(f"[make_admin] {action}: '{username}' ({email})")
    finally:
        db.close()

if __name__ == "__main__":
    main()

