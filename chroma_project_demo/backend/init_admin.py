#!/usr/bin/env python3
"""
Initialize Admin User Script
Creates the first admin user if no users exist in the database
"""

import sys
import os
from pathlib import Path

# Add the backend directory to Python path
backend_dir = Path(__file__).parent
sys.path.append(str(backend_dir))

from database import engine, Base, User, SessionLocal
from auth.jwt_handler import hash_password

def create_admin_user():
    """Create default admin user if no users exist"""
    try:
        # Create database tables
        Base.metadata.create_all(bind=engine)
        
        # Create database session
        db = SessionLocal()
        
        # Check if any users exist
        existing_users = db.query(User).count()
        
        if existing_users == 0:
            # Create default admin user
            admin_user = User(
                id="admin-001",
                username="admin",
                email="admin@example.com",
                hashed_password=hash_password("admin123"),
                is_admin=True,
                is_active=True
            )
            
            db.add(admin_user)
            db.commit()
            
            print("✅ Default admin user created successfully!")
            print("Username: admin")
            print("Password: admin123")
            print("⚠️  Please change the password after first login!")
            
        else:
            print(f"✅ Database already has {existing_users} user(s)")
            print("No need to create default admin user")
        
        db.close()
        
    except Exception as e:
        print(f"❌ Error creating admin user: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("🔧 Initializing Admin User...")
    create_admin_user() 