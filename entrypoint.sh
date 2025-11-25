#!/bin/sh
# Try to create a subdirectory that we can write to
if [ -d "/uploads" ]; then
    echo "📁 Creating writable subdirectory..."
    mkdir -p /uploads/images 2>/dev/null || true
    # Even if we can't chmod the parent, we might own the subdirectory
    chmod 777 /uploads/images 2>/dev/null || true
fi

# Start the application
exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1