# API Fixes Applied

## Summary
Fixed all API endpoints to work with existing data files and ChromaDB vector store.

## Changes Made

### 1. RAG Chatbot Agent (`agents/rag_chatbot_agent.py`)
- ✅ Added `query()` method for non-streaming API requests
- ✅ Added `query_stream()` method for streaming responses
- ✅ Auto-loads existing ChromaDB from `data/papers/rag_chroma/chroma.sqlite3`

### 2. RAG Router (`api/routers/rag_chatbot.py`)
- ✅ Auto-loads documents on agent initialization
- ✅ Properly handles load failures with detailed error messages
- ✅ Uses existing ChromaDB vector store

### 3. Support Agent (`agents/support_agents.py`)
- ✅ Added `self.model` attribute for API compatibility
- ✅ Added `run()` method wrapper for `process_ticket()`

### 4. Support Router (`api/routers/support.py`)
- ✅ Fixed ticket format to include proper `message` and `customer` structure
- ✅ Fixed cost calculation to use new `calculate_gemini_cost()` signature

### 5. Inventory Agent (`agents/inventory_agent.py`)
- ✅ Made `data_path` parameter optional
- ✅ Added `self.model` attribute
- ✅ Added comprehensive `run()` method that:
  - Accepts DataFrame directly
  - Runs full analysis pipeline
  - Formats results for API response

### 6. Inventory Router (`api/routers/inventory.py`)
- ✅ Updated default file path to use `data/inventory.csv`
- ✅ Fixed cost calculation to use new function signature

## Data Files Used

The API now uses these default data files:
- **RAG**: `data/papers/rag_chroma/chroma.sqlite3` (vector database)
- **RAG PDFs**: `data/papers/*.pdf` (12 AI research papers)
- **Inventory**: `data/inventory.csv`
- **Sales**: `data/sales_2025_q3.csv`
- **Support**: `data/tickets.json`

## Testing

### Start the API Server
```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Or use the batch file:
```bash
start_api.bat
```

### Run Tests
PowerShell:
```powershell
.\test_api.ps1
```

Batch:
```batch
.\test_api.bat
```

### Available Endpoints

#### Health Checks
- `GET /` - Root endpoint
- `GET /health` - Main health check
- `GET /api/rag/health` - RAG service health
- `GET /api/support/health` - Support service health
- `GET /api/inventory/health` - Inventory service health

#### RAG Chatbot
- `POST /api/rag/query` - Query with full response
- `POST /api/rag/query/stream` - Streaming response

#### Support Triage
- `POST /api/support/triage` - Process and classify support ticket

#### Inventory Analysis
- `POST /api/inventory/analyze` - Analyze inventory and get recommendations
- `POST /api/inventory/analyze/upload` - Upload CSV for analysis

## Next.js Integration

The API is ready to connect to your Next.js frontend. All endpoints return properly formatted JSON responses with:
- Consistent error handling
- Token usage tracking
- Cost calculations
- Proper CORS configuration

### Example Fetch
```typescript
const response = await fetch('http://localhost:8000/api/rag/query', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    query: 'What is attention mechanism?',
    top_k: 3
  })
});

const data = await response.json();
```

## Documentation
- Interactive API docs: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
