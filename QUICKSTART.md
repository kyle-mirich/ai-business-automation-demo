# Quick Start Guide

## You're Ready to Run! 🚀

All dependencies are installed and configured. Your API key is set up.

## Run the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## What You Can Do

1. **View the Homepage**
   - See the overview of all 3 demos
   - Demo 1 (Financial Report) is ready to use
   - Demos 2 & 3 are coming soon

2. **Try the Financial Report Generator**
   - Click "Launch Demo" on the Financial Report card
   - Click "🚀 Process sales_2025_q3.csv"
   - Watch the AI agent work in real-time:
     - Load Q3 2025 sales data (1,413 transactions)
     - Analyze with Gemini API (see the actual prompt!)
     - Generate Q4 forecast with Prophet
     - Create 5 interactive charts
   - Explore the results and insights

## Behind the Scenes

The app shows you:
- ✅ Actual prompts sent to Gemini
- ✅ Streaming AI responses
- ✅ Prophet forecasting process
- ✅ API costs and execution time

## Troubleshooting

If you encounter any issues:

1. **Make sure you're in the right directory:**
   ```bash
   cd "c:\Users\kylem\OneDrive\Documents\Code\Projects\AI Business Automation Demo Platform"
   ```

2. **API key issues:**
   - Check that `.env` file has your actual Gemini API key
   - Get one free at: https://makersuite.google.com/app/apikey

3. **Dependencies missing:**
   ```bash
   python test_setup.py
   ```

## What's Next

This is Feature 1 complete! Next steps:
- Build Demo 2: Support Triage (multi-agent with LangGraph)
- Build Demo 3: Inventory Optimizer
- Deploy to Streamlit Cloud

## Project Files

- `app.py` - Homepage
- `pages/1_📊_Financial_Report.py` - Financial demo
- `agents/financial_agent.py` - AI agent logic
- `data/sales_2025_q3.csv` - Pre-loaded sales data
- `utils/` - Helper functions

---

**Enjoy exploring the demo!** 🎉
