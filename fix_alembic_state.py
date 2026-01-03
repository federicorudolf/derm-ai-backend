#!/usr/bin/env python3
"""
Script to fix Alembic version state in database.
This checks if tables exist and stamps the database with the correct revision.
"""
import os
from sqlalchemy import create_engine, inspect, text
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    print("ERROR: DATABASE_URL environment variable not set")
    exit(1)

engine = create_engine(DATABASE_URL)
inspector = inspect(engine)

# Check which tables exist
tables = inspector.get_table_names()
print(f"Found {len(tables)} tables in database: {', '.join(tables)}")

# Define what tables should exist for each migration
EXPECTED_TABLES = {
    'password_reset_tokens': '536aec2b0cc6',
    'moles': 'a1b2c3d4e5f6',
}

# Check if all expected tables exist
all_tables_exist = all(table in tables for table in EXPECTED_TABLES.keys())

with engine.connect() as conn:
    # Check current alembic version
    result = conn.execute(text("SELECT version_num FROM alembic_version")).fetchone()
    current_version = result[0] if result else None
    print(f"Current Alembic version: {current_version}")

    if all_tables_exist:
        # All tables exist, stamp to the merge head
        target_version = 'b50d1d9d329b'
        print(f"\nAll expected tables exist. Stamping database to {target_version}")
        conn.execute(text(f"UPDATE alembic_version SET version_num = '{target_version}'"))
        conn.commit()
        print("✓ Database stamped successfully")
    else:
        missing_tables = [table for table in EXPECTED_TABLES.keys() if table not in tables]
        print(f"\nWARNING: Missing tables: {', '.join(missing_tables)}")
        print("Not stamping database. Please run migrations normally.")

print("\nDone!")
