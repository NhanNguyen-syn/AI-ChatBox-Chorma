import os
import sys
from sqlalchemy import create_engine, inspect, text

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import SQLALCHEMY_DATABASE_URL

def run_migration():
    """
    Checks for and adds the 'updated_by' column to the 'system_configs' table
    and the 'full_name' column to the 'users' table if they don't exist.
    """
    engine = create_engine(SQLALCHEMY_DATABASE_URL)
    inspector = inspect(engine)

    with engine.connect() as connection:
        try:
            # Check for 'updated_by' in 'system_configs'
            columns_system_configs = [c['name'] for c in inspector.get_columns('system_configs')]
            if 'updated_by' not in columns_system_configs:
                print("Adding 'updated_by' column to 'system_configs' table...")
                connection.execute(text('ALTER TABLE system_configs ADD COLUMN updated_by VARCHAR'))
                print("'updated_by' column added successfully.")
            else:
                print("'updated_by' column already exists in 'system_configs'.")

            # Check for 'full_name' in 'users'
            columns_users = [c['name'] for c in inspector.get_columns('users')]
            if 'full_name' not in columns_users:
                print("Adding 'full_name' column to 'users' table...")
                connection.execute(text('ALTER TABLE users ADD COLUMN full_name VARCHAR'))
                print("'full_name' column added successfully.")
            else:
                print("'full_name' column already exists in 'users'.")

            # Check for 'phone' in 'users'
            columns_users = [c['name'] for c in inspector.get_columns('users')]
            if 'phone' not in columns_users:
                print("Adding 'phone' column to 'users' table...")
                connection.execute(text('ALTER TABLE users ADD COLUMN phone VARCHAR'))
                print("'phone' column added successfully.")
            else:
                print("'phone' column already exists in 'users'.")


            # Commit the changes if any were made
            connection.commit()

        except Exception as e:
            print(f"An error occurred during migration: {e}")

if __name__ == "__main__":
    run_migration()

