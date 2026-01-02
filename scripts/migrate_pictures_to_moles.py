"""
One-time data migration script to convert existing pictures to individual moles.
Each existing picture becomes its own mole.

Run this script after running the alembic migration.

Usage:
    python scripts/migrate_pictures_to_moles.py
"""

import sys
import os

# Add parent directory to path to import from project
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sqlalchemy.orm import Session
from database import SessionLocal
from models import Picture, Mole


def migrate_existing_pictures():
    """Migrate all pictures without mole_id to individual moles"""
    db: Session = SessionLocal()
    try:
        # Get all pictures that don't have a mole_id
        pictures_without_moles = db.query(Picture).filter(Picture.mole_id == None).all()

        print(f"Found {len(pictures_without_moles)} pictures to migrate")

        if len(pictures_without_moles) == 0:
            print("No pictures to migrate. Migration already completed or no data exists.")
            return

        migrated_count = 0
        for picture in pictures_without_moles:
            # Create a new mole for each picture
            mole = Mole(
                user_id=picture.user_id,
                body_part_location=picture.body_part_location,
                created_at=picture.created_at,
                updated_at=picture.updated_at
            )
            db.add(mole)
            db.flush()  # Get the mole ID

            # Link picture to new mole
            picture.mole_id = mole.id
            migrated_count += 1

            if migrated_count % 100 == 0:
                print(f"Migrated {migrated_count} pictures...")

        db.commit()
        print(f"\nMigration completed successfully!")
        print(f"Total pictures migrated: {migrated_count}")
        print(f"Total moles created: {migrated_count}")

    except Exception as e:
        db.rollback()
        print(f"\nMigration failed: {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    print("=" * 60)
    print("Starting data migration: Pictures to Moles")
    print("=" * 60)
    migrate_existing_pictures()
    print("=" * 60)
