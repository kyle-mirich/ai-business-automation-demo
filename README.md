# AI Business Automation Demo Platform

A portfolio demo showcasing AI agents automating real business workflows using LangChain, Gemini AI, and Prophet.

## Overview

This Streamlit application demonstrates three interactive AI-powered business automation demos:

1. **📊 Financial Report Generator** (✅ Complete) - Analyzes Q3 sales data with Gemini AI and forecasts Q4 revenue using Prophet
2. **🎫 Support Triage** (⏳ Coming Soon) - Multi-agent system using LangGraph to process and route support tickets
3. **📦 Inventory Optimizer** (⏳ Coming Soon) - AI-powered inventory analysis and reorder recommendations

## Features

- ✨ **Real AI Calls** - Actual Gemini API responses (not mocked)
- 📊 **Pre-loaded Data** - Zero errors during demos
- 🔍 **Behind the Scenes** - See prompts, responses, and agent decisions in real-time
- 🚀 **Single Deployment** - Pure Python Streamlit app, no microservices

## Tech Stack

- **Framework:** Streamlit
- **AI/Agents:** LangChain, LangGraph
- **LLM:** Google Generative AI (Gemini)
- **Forecasting:** Prophet (Facebook)
- **Data Processing:** Pandas, NumPy
- **Visualization:** Plotly

## Installation

### Prerequisites

- Python 3.9 or higher
- Google Gemini API key ([Get one free](https://makersuite.google.com/app/apikey))

### Setup Steps

1. **Clone or navigate to the project directory**

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API key**

   Edit the `.env` file and add your Google Gemini API key:
   ```
   GOOGLE_API_KEY=your_actual_api_key_here
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open in browser**

   The app will automatically open at `http://localhost:8501`

## Usage

### Demo 1: Financial Report Generator

1. Navigate to "📊 Financial Report" in the sidebar
2. Click "🚀 Process sales_2025_q3.csv"
3. Watch the AI agent work in real-time:
   - Load and validate Q3 2025 sales data (1,413 transactions)
   - Analyze trends with Gemini API (see the actual prompt!)
   - Generate 90-day Q4 forecast with Prophet
   - Create 5 interactive visualizations
4. Review insights, metrics, and charts
5. Click "🔄 Run Again" for a fresh analysis

### Behind the Scenes

Each demo includes a "🔍 Behind the Scenes" panel showing:
- Data loading status
- Actual prompts sent to Gemini
- Streaming LLM responses
- Prophet model training
- API costs and execution time

## Project Structure

```
AI Business Automation Demo Platform/
├── app.py                          # Homepage with navigation
├── pages/
│   └── 1_📊_Financial_Report.py   # Demo 1: Financial analysis
├── agents/
│   └── financial_agent.py          # Agent for financial analysis
├── data/
│   └── sales_2025_q3.csv          # Pre-generated Q3 sales data
├── utils/
│   ├── data_loader.py             # Helper functions for loading data
│   └── cost_calculator.py         # Calculate API costs
├── .streamlit/
│   └── config.toml                # Streamlit theme configuration
├── requirements.txt               # Python dependencies
├── .env                           # Environment variables
└── README.md                      # This file
```

## Data

### Sales Data (Q3 2025)

The demo uses pre-loaded sales data with:
- **Rows:** 1,413 transactions
- **Date Range:** July 1 - September 30, 2025
- **Products:** 10 products across 3 categories
- **Revenue:** ~$100K with intentional growth trend
- **Customer Segments:** Consumer, Business, Education

## API Costs

The Financial Report demo typically costs:
- **Gemini API:** ~$0.02 - $0.05 per run
- **Prophet:** Free (local computation)

Total cost per demo run: **< $0.05**

## Deployment

### Streamlit Cloud

1. Push code to GitHub
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
3. Connect your GitHub repository
4. Set main file: `app.py`
5. Add secret: `GOOGLE_API_KEY` in Streamlit Cloud settings
6. Deploy!

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## Development Roadmap

- [x] Project setup and configuration
- [x] Generate realistic Q3 sales data
- [x] Build FinancialAgent with Gemini and Prophet
- [x] Create Financial Report Streamlit page
- [x] Build homepage with navigation
- [ ] Build Support Triage demo (LangGraph multi-agent)
- [ ] Build Inventory Optimizer demo
- [ ] Add tests
- [ ] Deploy to Streamlit Cloud

## License

MIT License - feel free to use this project as a portfolio template!

## Contact

Built by Kyle - Demonstrating AI-powered business automation

## Troubleshooting

### "API key not configured" error

Make sure you've:
1. Created a `.env` file in the project root
2. Added your Gemini API key: `GOOGLE_API_KEY=your_key_here`
3. Restarted the Streamlit app

### Prophet installation issues

On Windows, you may need to install Prophet dependencies:
```bash
pip install pystan
pip install prophet
```

### Import errors

Make sure you're running the app from the project root:
```bash
cd "AI Business Automation Demo Platform"
streamlit run app.py
```

## Acknowledgments

- Google Generative AI (Gemini)
- Facebook Prophet
- Streamlit
- LangChain
