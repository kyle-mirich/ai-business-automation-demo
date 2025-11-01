# Financial Module - Quick Start Guide

## ✅ What's Been Fixed

The financial module import error has been resolved. The issue was an incorrect import statement in `api/routers/financial.py`.

## 🚀 How to Start Everything

### 1. Start the Backend API

```bash
# From project root
cd "C:\Users\kylem\OneDrive\Documents\Code\Projects\AI Business Automation Demo Platform"

# Make sure your virtual environment is activated
.venv\Scripts\activate

# Start the FastAPI server
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

You should see output like:
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

### 2. Test the Financial API (Optional)

```bash
# In a new terminal
python test_financial_api.py
```

This will test:
- Health check endpoint
- Data loading endpoint
- Stats endpoint

### 3. Start the Frontend

```bash
# In a new terminal, navigate to frontend
cd frontend

# Start Next.js dev server
npm run dev
```

You should see:
```
   ▲ Next.js 15.x.x
   - Local:        http://localhost:3000
   - Ready in XXXms
```

### 4. Access the Financial Report Page

Open your browser to: **http://localhost:3000/financial**

## 📋 Financial Module Features

### Backend Routes (All Working ✅)
- `POST /api/financial/load` - Load sales data
- `GET /api/financial/stats` - Get summary statistics
- `POST /api/financial/chat` - Chat with AI analyst
- `POST /api/financial/generate-report` - Generate comprehensive report
- `GET /api/financial/visualizations` - Get Plotly charts
- `GET /api/financial/health` - Health check

### Frontend Features (3 Tabs)

#### Tab 1: Chat with AI
- Ask questions about Q3 2025 sales data
- See tool usage (what the AI is doing behind the scenes)
- View citation dataframes (source data)
- Cost tracking per query
- 8 suggested prompts

#### Tab 2: AI-Generated Report
- Generate comprehensive financial analysis
- Progress bar with status updates
- View intermediate steps (AI's work)
- Cost tracking

#### Tab 3: Data Dashboards
- Interactive Plotly visualizations
- Charts auto-generated from sales data
- Responsive and interactive

## 🐛 Troubleshooting

### "Financial module not loading"
**Fixed!** The import issue has been resolved.

### "Module 'api.routers.financial' has no attribute..."
Make sure you've restarted the API server after the fix:
```bash
# Stop the server (Ctrl+C)
# Start it again
uvicorn api.main:app --reload
```

### "Connection refused" or "Network error"
Make sure the API server is running on port 8000:
```bash
# Check if port 8000 is in use
netstat -ano | findstr :8000

# Start the API server
uvicorn api.main:app --reload
```

### Frontend can't connect to API
Check `frontend/.env.local` or `frontend/.env`:
```env
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```

### "GOOGLE_API_KEY not configured"
Make sure you have a `.env` file in the project root:
```env
GOOGLE_API_KEY=your_key_here
```

## 📊 Testing the Financial Page

1. **Load Data**: The page auto-loads Q3 2025 sales data on mount
2. **Try Chat**: Click on a suggested prompt like "What were the top 5 products by revenue?"
3. **View Tools**: Expand "Tools Used" to see what the AI did
4. **Check Citations**: Expand "View Source Data" to see the data that supports the answer
5. **Generate Report**: Switch to "AI-Generated Report" tab and click "Generate Report"
6. **View Dashboards**: Switch to "Data Dashboards" tab to see interactive charts

## 📁 Key Files

### Backend
- `api/routers/financial.py` - Financial API router (✅ Fixed)
- `agents/financial_agent_langchain.py` - LangChain financial agent
- `data/sales_2025_q3.csv` - Q3 2025 sales data

### Frontend
- `frontend/app/(dashboard)/financial/page.tsx` - Main financial page
- `frontend/lib/api.ts` - API client methods
- `frontend/lib/types.ts` - TypeScript types

## 🎯 Next Steps

The Financial Report page is now fully functional! You can:
1. Test all 3 tabs
2. Ask questions and see AI reasoning
3. Generate reports with progress tracking
4. View interactive visualizations

Want to enhance it further? Consider adding:
- Drill-down on citation tables (click to see transactions)
- Pandas code execution viewer
- Export reports as PDF
- Save chat history

## 💡 Pro Tips

- **Cost Tracking**: Watch the session cost ticker to see API usage
- **Suggested Prompts**: Use the quick-action buttons for common queries
- **Tool Visibility**: Always expand "Tools Used" to understand AI reasoning
- **Citation Tables**: Click to expand and verify the data sources

---

**Status**: ✅ **WORKING** - All financial module issues resolved!
