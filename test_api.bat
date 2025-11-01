@echo off
echo Testing AI Business Automation API Endpoints
echo ============================================
echo.

echo 1. Testing root endpoint...
curl -s http://localhost:8000/
echo.
echo.

echo 2. Testing health check...
curl -s http://localhost:8000/health
echo.
echo.

echo 3. Testing RAG health...
curl -s http://localhost:8000/api/rag/health
echo.
echo.

echo 4. Testing Support health...
curl -s http://localhost:8000/api/support/health
echo.
echo.

echo 5. Testing Inventory health...
curl -s http://localhost:8000/api/inventory/health
echo.
echo.

echo 6. Testing RAG query...
curl -s -X POST "http://localhost:8000/api/rag/query" -H "Content-Type: application/json" -d "{\"query\": \"What is attention mechanism?\", \"top_k\": 2}"
echo.
echo.

echo 7. Testing Support triage...
curl -s -X POST "http://localhost:8000/api/support/triage" -H "Content-Type: application/json" -d "{\"ticket_id\": \"T-001\", \"subject\": \"Cannot login\", \"description\": \"I forgot my password\", \"customer_email\": \"test@example.com\", \"customer_name\": \"John Doe\"}"
echo.
echo.

echo 8. Testing Inventory analysis with default data...
curl -s -X POST "http://localhost:8000/api/inventory/analyze" -H "Content-Type: application/json" -d "{}"
echo.
echo.

echo ============================================
echo All tests complete!
pause
