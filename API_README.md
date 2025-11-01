# AI Business Automation API

FastAPI backend for AI-powered business automation agents.

## Features

- **RAG Chatbot** - Query documents with AI-powered retrieval
- **Support Triage** - Automated ticket classification and routing
- **Inventory Optimization** - AI-driven inventory recommendations

## Quick Start

### Local Development

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set environment variables**
   ```bash
   cp .env.example .env
   # Add your GOOGLE_API_KEY to .env
   ```

3. **Run the server**
   ```bash
   uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
   ```

4. **Access API documentation**
   - Swagger UI: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

## API Endpoints

### Health Check
```
GET /health
```

### RAG Chatbot
```
POST /api/rag/query
POST /api/rag/query/stream  # Streaming responses
GET /api/rag/health
```

### Support Triage
```
POST /api/support/triage
GET /api/support/health
```

### Inventory Optimization
```
POST /api/inventory/analyze
POST /api/inventory/analyze/upload
GET /api/inventory/health
```

## Deployment to Render

1. **Connect your GitHub repository to Render**

2. **Create a new Web Service**
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`

3. **Set environment variables**
   - `GOOGLE_API_KEY`: Your Google Gemini API key
   - `PYTHON_VERSION`: 3.11.0

4. **Deploy**
   - Render will automatically deploy from your `beta` branch

## Example Usage

### RAG Chatbot
```bash
curl -X POST "http://localhost:8000/api/rag/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is attention mechanism?",
    "top_k": 3
  }'
```

### Support Triage
```bash
curl -X POST "http://localhost:8000/api/support/triage" \
  -H "Content-Type: application/json" \
  -d '{
    "ticket_id": "T-12345",
    "subject": "Cannot login",
    "description": "I forgot my password",
    "customer_email": "user@example.com"
  }'
```

### Inventory Analysis
```bash
curl -X POST "http://localhost:8000/api/inventory/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "file_path": "./data/inventory_sample.csv"
  }'
```

## CORS Configuration

By default, CORS is configured to allow all origins. Update `api/main.py` to restrict origins in production:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-frontend-domain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Next.js Integration

The API is designed to work seamlessly with Next.js. Example fetch:

```typescript
const response = await fetch('http://localhost:8000/api/rag/query', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    query: 'What is attention mechanism?',
    top_k: 3,
  }),
});

const data = await response.json();
```

## License

MIT
