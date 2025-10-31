# 🤖 AI Business Automation Demo Platform

> A portfolio project demonstrating how AI agents can automate real-world business workflows using LangChain, Google Gemini AI, and Prophet forecasting.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Demo Applications](#demo-applications)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Usage Guide](#usage-guide)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This Streamlit-based platform showcases **three complete AI-powered demos** that solve real business problems. Each demo uses production-ready patterns including RAG (Retrieval Augmented Generation), multi-agent orchestration, and ML-powered forecasting.

### Why This Project?

This isn't a toy demo with mocked responses - it's a **fully functional platform** that demonstrates:
- ✅ Real API calls to Google Gemini AI
- ✅ Multi-agent systems with LangGraph
- ✅ RAG implementation with verifiable citations
- ✅ Time-series forecasting with Prophet ML
- ✅ Production patterns (error handling, cost tracking, data validation)
- ✅ Clean architecture (separation of concerns, reusable components)

Perfect for demonstrating AI/ML engineering skills to recruiters and technical interviewers.

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🧠 **Real AI Agents** | Actual LLM API calls - not mocked or simulated |
| 🔍 **Full Transparency** | See every prompt, response, and agent decision in real-time |
| 📊 **Pre-loaded Data** | 1,413 transactions, 50 inventory items, 5 support tickets - zero setup errors |
| 💰 **Cost Tracking** | Real-time token usage and API cost calculations |
| 🎨 **Professional UI** | Clean Streamlit interface with custom styling |
| 🔒 **Secure by Default** | Environment-based API key management, safe code execution |
| 📈 **Production Ready** | Error handling, data validation, session management |

## 🚀 Demo Applications

### 1. 📊 Financial Report Generator

An AI agent that analyzes sales data, generates insights, and forecasts future revenue.

**Capabilities:**
- Natural language queries over CSV data (1,413 Q3 2025 transactions)
- **RAG with citations** - every answer links to specific CSV rows
- Interactive drill-down - click aggregated data to see source transactions
- Pandas code generation for complex calculations
- Prophet ML forecasting (90-day Q4 predictions)
- 5 interactive Plotly visualizations

**Technical Highlights:**
- Custom tool system with 4 specialized tools (QuerySalesData, CalculateStatistics, FindSpecificData, ForecastRevenue)
- Safe code execution sandbox for Pandas operations
- Session state management for chat history
- Fuzzy date matching (handles "sept", "Q3", date ranges)

**Agent Architecture:** LangChain with custom tools + Prophet integration

---

### 2. 🎫 Support Ticket Triage

A multi-agent system that automatically triages, prioritizes, and routes support tickets.

**Capabilities:**
- End-to-end ticket processing (classification → prioritization → routing → response)
- 4 coordinated AI agents working in sequence
- Real-time agent reasoning display
- Token usage and cost tracking per agent
- Batch processing of 5 diverse ticket scenarios

**Technical Highlights:**
- LangGraph state machine with typed state management
- Parallel agent execution where possible
- JSON response parsing with fallback handling
- Category/priority normalization
- Comprehensive prompt engineering for each agent role

**Agent Architecture:** LangGraph with 4-node workflow (Classifier → Prioritizer → Router → Responder)

---

### 3. 📦 Inventory Optimizer

AI-powered inventory analysis with demand forecasting and reorder recommendations.

**Capabilities:**
- Per-SKU demand forecasting using Prophet ML
- Risk scoring algorithm (combines low stock, fast movers, lead times)
- Gemini-powered reorder recommendations with business context
- Stock health diagnostics (identifies low stock, overstock, fast movers)
- Interactive forecast visualizations with confidence intervals

**Technical Highlights:**
- Synthetic historical data generation for Prophet training (MD5-seeded for reproducibility)
- Fallback forecasting when Prophet fails
- Coverage days and projected gap calculations
- LangChain chain with structured JSON output
- Color-coded inventory status (red/yellow/green)

**Agent Architecture:** LangChain chain + Prophet forecasting + Gemini reasoning

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit UI Layer                       │
│  (Home.py + pages/1_Financial_Report.py, etc.)             │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│                   Agent Layer                               │
│                                                             │
│  ┌──────────────────┐  ┌──────────────────┐  ┌───────────┐│
│  │ Financial Agent  │  │ Support Agents   │  │ Inventory ││
│  │ (LangChain +     │  │ (LangGraph       │  │ Agent     ││
│  │  Custom Tools)   │  │  Multi-Agent)    │  │ (LangChain││
│  └──────────────────┘  └──────────────────┘  └───────────┘│
│         │                      │                    │       │
└─────────┼──────────────────────┼────────────────────┼───────┘
          │                      │                    │
          ▼                      ▼                    ▼
┌─────────────────────────────────────────────────────────────┐
│              Google Gemini API (LLM Layer)                  │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    Data Layer                               │
│  • sales_2025_q3.csv (1,413 transactions)                  │
│  • tickets.json (5 support scenarios)                       │
│  • inventory.csv (50 SKUs)                                  │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                ML/Forecasting Layer                         │
│             Facebook Prophet (local)                        │
└─────────────────────────────────────────────────────────────┘
```

### Design Principles

1. **Separation of Concerns**: UI (pages) vs Logic (agents) vs Utils
2. **Reusability**: Shared utilities for data loading, cost calculation
3. **Transparency**: Every agent exposes its reasoning process
4. **Fail-Safe**: Graceful error handling, fallback strategies
5. **Cost-Conscious**: Token tracking and cost estimation throughout

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- Google Gemini API key ([Get one free](https://makersuite.google.com/app/apikey))

### Installation

```bash
# 1. Clone or download this repository
git clone <your-repo-url>
cd "AI Business Automation Demo Platform"

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure your API key (choose ONE method):

## Option A: Using .env file (recommended for local development)
cp .env.example .env
# Edit .env and add: GOOGLE_API_KEY=your_actual_key_here

## Option B: Using Streamlit secrets (recommended for deployment)
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Edit .streamlit/secrets.toml and add your key

# 4. Get your free API key
# Visit: https://makersuite.google.com/app/apikey

# 5. Run the application
streamlit run Home.py
```

The app will open automatically at `http://localhost:8501`

### Verify Setup

```bash
# Optional: Run setup verification
python test_setup.py
```

This checks Python version, dependencies, data files, and API key configuration.

## 📖 Usage Guide

### Demo 1: Financial Report Generator

1. Navigate to **📊 Financial Report** in the sidebar
2. Choose interaction mode:
   - **Chat Mode**: Ask natural language questions about the data
   - **Quick Report**: Generate automated analysis
3. Try example queries:
   - "What were total sales in September?"
   - "Show me top 5 products by revenue"
   - "Which customer segment had highest growth?"
4. Click on citation data to drill down to source transactions
5. View 90-day revenue forecast in the Visualizations tab

**Sample Questions:**
- "What's the sales trend over Q3?"
- "Which products should we focus on in Q4?"
- "Compare Consumer vs Business segment performance"

---

### Demo 2: Support Ticket Triage

1. Navigate to **🎫 Support Triage** in the sidebar
2. Review the 5 pre-loaded support tickets
3. Click **🚀 Process All Tickets**
4. Watch the multi-agent system work:
   - Classifier categorizes each ticket
   - Prioritizer assigns urgency levels
   - Router assigns to correct department
   - Responder drafts initial reply
5. Review the summary dashboard showing:
   - Category breakdown
   - Priority distribution
   - Department routing
   - Token usage and costs

**Behind the Scenes**: Expand each ticket to see all 4 agent decisions

---

### Demo 3: Inventory Optimizer

1. Navigate to **📦 Inventory Optimizer** in the sidebar
2. Click **🚀 Analyze Inventory**
3. The agent will:
   - Load 50 SKU inventory data
   - Generate demand forecasts for each item
   - Calculate risk scores
   - Provide reorder recommendations
4. Review outputs:
   - Priority reorder recommendations (red/yellow/green coding)
   - Stock health diagnostics
   - Interactive forecast charts for top 3 critical items
   - Priority mix visualization

**Key Metrics:**
- Coverage Days: How long current stock will last
- Projected Gap: Predicted stockout timing
- Risk Score: Composite urgency metric

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **Framework** | Streamlit (Web UI) |
| **AI/LLM** | Google Generative AI (Gemini 2.5/2.0 Flash) |
| **Agent Orchestration** | LangChain, LangGraph |
| **ML Forecasting** | Prophet (Facebook) |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Plotly |
| **Environment** | Python 3.9+, python-dotenv |

### Why These Choices?

- **Streamlit**: Rapid development, native Python, great for demos
- **Gemini**: Fast, cost-effective, competitive with GPT-4 for many tasks
- **LangChain/LangGraph**: Industry-standard agent frameworks
- **Prophet**: Battle-tested time-series forecasting (used by Facebook, Uber)

## 📁 Project Structure

```
AI Business Automation Demo Platform/
├── Home.py                             # Main entry point and homepage
├── pages/                              # Streamlit multi-page app
│   ├── 1_📊_Financial_Report.py       # Financial analysis demo (516 lines)
│   ├── 2_🎫_Support_Triage.py         # Multi-agent triage demo (218 lines)
│   └── 3_📦_Inventory_Optimizer.py    # Inventory optimization (459 lines)
├── agents/                             # Business logic layer
│   ├── financial_agent_langchain.py   # Financial agent with custom tools (888 lines)
│   ├── support_agents.py              # LangGraph workflow (444 lines)
│   └── inventory_agent.py             # Inventory forecasting (463 lines)
├── utils/                              # Shared utilities
│   ├── data_loader.py                 # Data loading and validation
│   └── cost_calculator.py             # Token/cost estimation
├── data/                               # Pre-loaded demo data
│   ├── sales_2025_q3.csv              # 1,413 Q3 transactions
│   ├── tickets.json                   # 5 support ticket examples
│   └── inventory.csv                  # 50 inventory SKUs
├── .streamlit/
│   └── config.toml                    # UI theme configuration
├── requirements.txt                   # Python dependencies
├── .env.example                       # Environment variable template
├── .gitignore                         # Git ignore rules
├── LICENSE                            # MIT License
├── test_setup.py                      # Setup verification script
└── README.md                          # This file
```

**Total Lines of Code:** ~3,460 lines of production Python

## 💰 API Costs

All demos use Google Gemini API with the following pricing (Standard Paid Tier):

**Google Gemini Pricing (per 1 million tokens):**
- Input (text/image/video): $0.10
- Output (including thinking tokens): $0.40
- Context caching: $0.01 (text/image/video)

**Estimated costs per demo run:**

| Demo | Input Tokens | Output Tokens | Estimated Cost | Notes |
|------|--------------|---------------|----------------|-------|
| Financial Report | ~2,000-5,000 | ~1,000-3,000 | $0.0006-$0.0017 | Prophet forecasting is free (runs locally) |
| Support Triage | ~4,000-8,000 | ~2,000-4,000 | $0.0012-$0.0024 | 4 agents × 5 tickets with JSON outputs |
| Inventory Optimizer | ~6,000-10,000 | ~3,000-5,000 | $0.0018-$0.0030 | 50 SKU forecasts + recommendations |

**Total estimated cost per complete demo session: ~$0.004-$0.006** (less than one cent!)

**Free Tier:** Google Gemini offers a generous free tier for testing and development.

## 🌐 Deployment

### Streamlit Cloud (Recommended)

1. **Push code to GitHub**
   ```bash
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

2. **Deploy to Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Connect your GitHub repository
   - Configure:
     - **Main file**: `Home.py`
     - **Python version**: 3.9+

3. **Add secrets in Streamlit Cloud**
   - In your app dashboard, go to "Settings" → "Secrets"
   - Add the following (TOML format):
   ```toml
   GOOGLE_API_KEY = "your_actual_api_key_here"
   ```
   - Click "Save"

4. **Deploy!**
   - Your app will automatically deploy
   - Any future git pushes will trigger automatic redeployment

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run Home.py

# Run on specific port
streamlit run Home.py --server.port 8080
```

### Docker (Optional)

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "Home.py"]
```

## 🧪 Testing

```bash
# Run setup verification
python test_setup.py

# Check all dependencies
pip check

# Verify data files
ls -lh data/
```

## 🤝 Contributing

This is a portfolio project, but suggestions are welcome!

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 🐛 Troubleshooting

### "API key not configured" error

The app supports **two methods** for API key configuration:

**Method 1: .env file (Local Development)**
1. Copy `.env.example` to `.env`
2. Add your Gemini API key: `GOOGLE_API_KEY=your_actual_key_here`
3. Restart the Streamlit app

**Method 2: Streamlit secrets (Deployment)**
1. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`
2. Add your key in TOML format: `GOOGLE_API_KEY = "your_key_here"`
3. Restart the Streamlit app

**For Streamlit Cloud:**
- Add secrets via the app dashboard: Settings → Secrets
- Use TOML format (no quotes around key names)

**Get a free API key:**
Visit [Google AI Studio](https://makersuite.google.com/app/apikey)

### Prophet installation issues on Windows

```bash
# Install Prophet dependencies first
pip install pystan
pip install prophet

# If still failing, try conda
conda install -c conda-forge prophet
```

### Import errors

Make sure you're running from the project root:
```bash
cd "AI Business Automation Demo Platform"
streamlit run Home.py
```

### Streamlit caching issues

```bash
# Clear Streamlit cache
streamlit cache clear
```

## 📚 Learn More

- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [LangGraph Guide](https://langchain-ai.github.io/langgraph/)
- [Google Gemini API Docs](https://ai.google.dev/docs)
- [Prophet Forecasting](https://facebook.github.io/prophet/)
- [Streamlit Documentation](https://docs.streamlit.io/)

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Kyle M**

Built to demonstrate practical AI/ML engineering skills for business automation.

## 🙏 Acknowledgments

- **Google Generative AI** - Gemini LLM models
- **LangChain/LangGraph** - Agent orchestration frameworks
- **Facebook Prophet** - Time-series forecasting library
- **Streamlit** - Rapid web app development
- **Plotly** - Interactive visualizations

---

**⭐ If this project helped you, consider giving it a star!**
