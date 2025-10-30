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

st.markdown("#### Agent Sandboxes")

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
    ### 🎫 Support Ticket Triage

    LangGraph-powered workflow that classifies, prioritizes,
    routes, and drafts responses for live support tickets.

    **Features:**
    - Classifier, Prioritizer, Router, Responder agents
    - Step-by-step agent reasoning
    - Token usage & cost tracking
    - Gemini-generated replies

    **Tech Stack:**
    `LangGraph` `LangChain` `Gemini`
    """)

    st.page_link("pages/2_🎫_Support_Triage.py", label="🚀 Launch Support Triage", use_container_width=True)

with col3:
    st.markdown("""
    ### 📦 Inventory Optimizer

    End-to-end inventory planning with **Prophet** demand forecasting
    and **Gemini** reorder recommendations via LangChain.

    **Features:**
    - Stock health diagnostics
    - Per-SKU demand forecasts
    - AI reorder plans with reasoning
    - Cost transparency

    **Tech Stack:**
    `LangChain` `Gemini` `Prophet` `Plotly`
    """)

    st.page_link("pages/3_📦_Inventory_Optimizer.py", label="🚀 Launch Inventory Optimizer", use_container_width=True)

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
    This is a portfolio demo by Kyle Mirich showcasing AI agents automating real business workflows.
    """)

    st.divider()

    st.markdown("""
    **Quick Links:**
    - [Kyle's Portfolio](https://kyle-mirich.vercel.app/)
    - [Kyle's Github](https://github.com/kyle-mirich)
    """)

st.divider()

