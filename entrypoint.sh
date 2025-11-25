#!/bin/sh
# Check if /uploads exists and try to make it writable
if [ -d "/uploads" ]; then
    echo "📁 /uploads exists, checking permissions..."
    if [ ! -w "/uploads" ]; then
        echo "⚠️  /uploads not writable, attempting fix..."
        chmod 777 /uploads 2>/dev/null || echo "❌ Cannot change permissions (running as non-root)"
    else
        echo "✅ /uploads is writable"
    fi
else
    echo "❌ /uploads directory not found!"
fi

# Start the application
echo "🚀 Starting DermAI Backend..."
exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1