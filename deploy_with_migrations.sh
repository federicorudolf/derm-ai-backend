#!/bin/bash
set -e  # Exit on any error

echo "========================================="
echo "Starting Deployment with Migrations"
echo "========================================="

# Check if DATABASE_URL is set
if [ -z "$DATABASE_URL" ]; then
    echo "ERROR: DATABASE_URL environment variable not set"
    exit 1
fi

# Function to get current alembic version
get_current_version() {
    python3 -c "
from sqlalchemy import create_engine, text
import os
engine = create_engine(os.getenv('DATABASE_URL'))
with engine.connect() as conn:
    result = conn.execute(text('SELECT version_num FROM alembic_version')).fetchone()
    print(result[0] if result else 'None')
" 2>/dev/null || echo "None"
}

# Get current version before migration
CURRENT_VERSION=$(get_current_version)
echo ""
echo "Current database version: $CURRENT_VERSION"
echo ""

# Check for pending migrations
echo "Checking for pending migrations..."
PENDING=$(alembic current 2>&1)
echo "$PENDING"
echo ""

# Run migrations
echo "Running migrations..."
if alembic upgrade head; then
    echo "✓ Migrations completed successfully"

    # Get new version after migration
    NEW_VERSION=$(get_current_version)
    echo "New database version: $NEW_VERSION"

    if [ "$CURRENT_VERSION" != "$NEW_VERSION" ]; then
        echo "✓ Database upgraded from $CURRENT_VERSION to $NEW_VERSION"
    else
        echo "✓ Database already at latest version"
    fi
else
    echo "✗ Migration failed!"
    echo ""
    echo "Showing migration history:"
    alembic history
    echo ""
    echo "Current alembic heads:"
    alembic heads
    exit 1
fi

echo ""
echo "========================================="
echo "Starting Application"
echo "========================================="
exec uvicorn main:app --host 0.0.0.0 --port 8080 --workers 1
