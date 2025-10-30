"""
AI Business Automation Demo Platform
Homepage with navigation to 3 interactive demos
"""

import streamlit as st

# Page configuration
st.set_page_config(
    page_title="AI Business Automation Demo",
    page_icon="🤖",
    layout="wide"
)

# Header
st.title("🤖 AI Business Automation Demo")
st.markdown("""
Watch AI agents automate real business workflows in real-time.
See actual prompts, LLM responses, and results - no setup required.
""")

st.divider()

# Demo cards
st.header("Interactive Demos")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### 🤖 AI Financial Analyst with Chat

    **LangChain-powered** agent that answers questions about Q3 2025 sales
    data using **RAG with citations** and **Prophet forecasting**.

    **Features:**
    - 💬 Interactive chatbot (ask anything!)
    - 🔧 Visible tool usage & reasoning
    - 📚 Citations to source data (row numbers)
    - 🔮 ML-based Q4 forecasting
    - 📊 Automated report generation

    **Tech Stack:**
    `LangChain` `Gemini` `Prophet` `RAG` `Plotly`
    """)

    st.page_link("pages/1_📊_Financial_Report.py", label="🚀 Launch Demo", use_container_width=True)

with col2:
    st.markdown("""
    ### 🎫 Support Triage (Coming Soon)

    Multi-agent system built with **LangGraph** processes support tickets
    through 4 coordinated agents.

    **Features:**
    - Classifier Agent
    - Prioritizer Agent
    - Router Agent
    - Response Generator

    **Tech Stack:**
    `LangGraph` `Gemini` `Multi-Agent`
    """)

    st.button("🔒 Coming Soon", disabled=True, use_container_width=True)

with col3:
    st.markdown("""
    ### 📦 Inventory Optimizer (Coming Soon)

    Analyzes inventory levels, forecasts demand with **Prophet**,
    and generates reorder recommendations.

    **Features:**
    - Inventory analysis
    - Demand forecasting
    - AI recommendations
    - Priority scoring

    **Tech Stack:**
    `Prophet` `Gemini` `Pandas`
    """)

    st.button("🔒 Coming Soon", disabled=True, use_container_width=True)

st.divider()

# Features section
st.header("Why This Demo?")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    **✨ Real AI Calls**

    Not mocked - actual Gemini API
    responses every time
    """)

with col2:
    st.markdown("""
    **📊 Pre-loaded Data**

    Zero errors during demos,
    works perfectly every time
    """)

with col3:
    st.markdown("""
    **🔍 Behind the Scenes**

    See prompts, responses,
    and agent decisions live
    """)

with col4:
    st.markdown("""
    **🚀 Single Deployment**

    Pure Python Streamlit app,
    no microservices needed
    """)

st.divider()

# Tech stack
st.header("Built With")

st.markdown("""
**Framework:** Streamlit |
**AI/Agents:** LangChain, LangGraph |
**LLM:** Google Generative AI (Gemini) |
**Forecasting:** Prophet |
**Data:** Pandas, NumPy |
**Viz:** Plotly
""")

st.divider()

# Footer
st.markdown("""
<div style='text-align: center; color: #64748b; font-size: 0.9em;'>
    Built by Kyle • Demonstrating AI-powered business automation •
    <a href='https://github.com' style='color: #2563eb;'>View on GitHub</a>
</div>
""", unsafe_allow_html=True)

# Sidebar info
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This is a portfolio demo showcasing AI agents automating real business workflows.

    **Current Status:**
    - ✅ Demo 1: Financial Report (Complete)
    - ⏳ Demo 2: Support Triage (Coming)
    - ⏳ Demo 3: Inventory Optimizer (Coming)

    **Setup Required:**
    1. Get a Google Gemini API key
    2. Add to `.env` file
    3. Install dependencies
    4. Run `streamlit run app.py`

    **No file uploads needed** - all data is pre-loaded!
    """)

    st.divider()

    st.markdown("""
    **Quick Links:**
    - [Google AI Studio](https://makersuite.google.com/app/apikey)
    - [Streamlit Docs](https://docs.streamlit.io)
    - [Prophet Docs](https://facebook.github.io/prophet/)
    """)
