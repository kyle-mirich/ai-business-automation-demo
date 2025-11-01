Write-Host "Testing AI Business Automation API Endpoints" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "1. Testing root endpoint..." -ForegroundColor Yellow
curl.exe -s http://localhost:8000/
Write-Host "`n"

Write-Host "2. Testing health check..." -ForegroundColor Yellow
curl.exe -s http://localhost:8000/health
Write-Host "`n"

Write-Host "3. Testing RAG health..." -ForegroundColor Yellow
curl.exe -s http://localhost:8000/api/rag/health
Write-Host "`n"

Write-Host "4. Testing Support health..." -ForegroundColor Yellow
curl.exe -s http://localhost:8000/api/support/health
Write-Host "`n"

Write-Host "5. Testing Inventory health..." -ForegroundColor Yellow
curl.exe -s http://localhost:8000/api/inventory/health
Write-Host "`n"

Write-Host "6. Testing RAG query..." -ForegroundColor Yellow
curl.exe -s -X POST "http://localhost:8000/api/rag/query" -H "Content-Type: application/json" -d '{\"query\": \"What is attention mechanism?\", \"top_k\": 2}'
Write-Host "`n"

Write-Host "7. Testing Support triage..." -ForegroundColor Yellow
curl.exe -s -X POST "http://localhost:8000/api/support/triage" -H "Content-Type: application/json" -d '{\"ticket_id\": \"T-001\", \"subject\": \"Cannot login\", \"description\": \"I forgot my password\", \"customer_email\": \"test@example.com\", \"customer_name\": \"John Doe\"}'
Write-Host "`n"

Write-Host "8. Testing Inventory analysis with default data..." -ForegroundColor Yellow
curl.exe -s -X POST "http://localhost:8000/api/inventory/analyze" -H "Content-Type: application/json" -d '{}'
Write-Host "`n"

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "All tests complete!" -ForegroundColor Green
