#!/bin/bash

echo "🔐 DermAI Backend Secure Deployment Script"
echo "=========================================="

# Check if SECRET_KEY is set
if [ -z "$SECRET_KEY" ]; then
    echo "⚠️  WARNING: SECRET_KEY not set!"
    echo "Please generate a secure secret key:"
    echo "export SECRET_KEY=\$(openssl rand -hex 32)"
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Build and run with Docker Compose
echo "🐳 Building Docker container..."
docker-compose build

echo "🚀 Starting DermAI Backend..."
docker-compose up -d

echo "✅ Deployment complete!"
echo ""
echo "🌐 API will be available at: http://localhost:8000"
echo "📊 Health check: http://localhost:8000/health"
echo "📖 API docs: http://localhost:8000/docs"
echo ""
echo "🔧 To view logs: docker-compose logs -f"
echo "🛑 To stop: docker-compose down"