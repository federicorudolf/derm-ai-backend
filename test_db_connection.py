#!/usr/bin/env python3
"""
Database connection test script
Run this to test your database connection before starting the main application
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_database_connection():
    """Test database connection with detailed error reporting"""
    try:
        from database import test_connection, engine, DATABASE_URL
        
        print("🔍 Testing database connection...")
        print(f"📡 Database URL: {DATABASE_URL.replace(DATABASE_URL.split('@')[0].split('//')[1], '***') if '@' in DATABASE_URL else DATABASE_URL}")
        
        # Test basic connection
        if test_connection():
            print("✅ Database connection successful!")
            
            # Test a simple query
            try:
                from sqlalchemy import text
                with engine.connect() as conn:
                    result = conn.execute(text("SELECT version()"))
                    version = result.fetchone()[0]
                    print(f"📊 PostgreSQL version: {version}")
                    
                print("✅ Database query test successful!")
                return True
            except Exception as e:
                print(f"❌ Database query test failed: {e}")
                return False
        else:
            print("❌ Database connection failed!")
            return False
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure you have installed all required dependencies:")
        print("   pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def main():
    """Main function"""
    print("🚀 DermAI Database Connection Test")
    print("=" * 40)
    
    # Check if .env file exists
    if not os.path.exists('.env'):
        print("⚠️  No .env file found!")
        print("💡 Copy .env.example to .env and configure your database settings:")
        print("   cp .env.example .env")
        print("   # Then edit .env with your database credentials")
        return False
    
    success = test_database_connection()
    
    print("=" * 40)
    if success:
        print("🎉 All tests passed! Your database is ready.")
        print("🚀 You can now start your FastAPI application.")
    else:
        print("💥 Database connection failed!")
        print("🔧 Please check your database configuration and try again.")
        print("\n📋 Common solutions:")
        print("   1. Verify your DATABASE_URL in .env file")
        print("   2. Ensure PostgreSQL is running")
        print("   3. Check firewall/network settings")
        print("   4. Verify database credentials")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
