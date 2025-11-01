@echo off
echo Starting AI Business Automation API Server...
echo ============================================
echo.
echo Server will be available at: http://localhost:8000
echo API Documentation: http://localhost:8000/docs
echo.
echo Press Ctrl+C to stop the server
echo.

uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
