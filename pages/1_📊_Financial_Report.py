"""
Financial Report Generator with Interactive AI Chatbot - Demo 1
Features:
- LangChain-powered AI agent with visible tool usage
- Interactive chatbot with RAG (Retrieval Augmented Generation)
- Real-time analysis with citations to source data
- Prophet forecasting integration
- Transparent AI reasoning
"""

import streamlit as st
import os
from pathlib import Path
import time
from dotenv import load_dotenv
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.financial_agent_langchain import FinancialAgentLangChain

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="AI Financial Analyst",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS for chat interface
st.markdown("""
<style>
.chat-message {
    padding: 1rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    display: flex;
    flex-direction: column;
}
.user-message {
    background-color: #e3f2fd;
    border-left: 4px solid #2563eb;
}
.ai-message {
    background-color: #f5f5f5;
    border-left: 4px solid #10b981;
}
.tool-usage {
    background-color: #fff3cd;
    border-left: 4px solid #ff9800;
    padding: 0.5rem;
    margin: 0.5rem 0;
    border-radius: 0.25rem;
    font-family: monospace;
    font-size: 0.9em;
}
.citation {
    background-color: #e8f5e9;
    padding: 0.25rem 0.5rem;
    border-radius: 0.25rem;
    font-size: 0.85em;
    color: #2e7d32;
    margin-top: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("🤖 AI Financial Analyst with Interactive Chat")
st.markdown("""
Ask questions about your Q3 2025 sales data and watch the AI agent work in real-time!
See which tools it uses, how it retrieves data, and get answers with citations to source data.
""")

st.divider()

# Check for API key
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key or api_key == "your_api_key_here":
    st.error("⚠️ Google Gemini API key not configured!")
    st.info("""
    **To use this demo:**
    1. Get a free API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
    2. Add it to the `.env` file: `GOOGLE_API_KEY=your_key_here`
    3. Restart the Streamlit app
    """)
    st.stop()

# Data file path
data_path = Path(__file__).parent.parent / "data" / "sales_2025_q3.csv"

# Initialize session state
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []
if 'report_generated' not in st.session_state:
    st.session_state.report_generated = False
if 'figures' not in st.session_state:
    st.session_state.figures = None
if 'auto_load_attempted' not in st.session_state:
    st.session_state.auto_load_attempted = False
if 'data_load_error' not in st.session_state:
    st.session_state.data_load_error = None


def attempt_sales_data_load() -> None:
    """Instantiate the financial agent and load sales data."""
    st.session_state.auto_load_attempted = True
    st.session_state.data_load_error = None
    st.session_state.agent = FinancialAgentLangChain(api_key=api_key, data_path=str(data_path))

    result = st.session_state.agent.load_data()
    if result.get('success'):
        st.session_state.data_loaded = True
        st.session_state.chat_messages = []
        st.session_state.report_generated = False
        st.session_state.figures = None
    else:
        st.session_state.agent = None
        st.session_state.data_loaded = False
        st.session_state.data_load_error = result.get('message', 'Unknown error while loading data.')


if not st.session_state.data_loaded and not st.session_state.auto_load_attempted:
    with st.spinner("Loading sales data..."):
        attempt_sales_data_load()

# Sidebar - System Status
with st.sidebar:
    st.header("🔧 System Status")

    if not st.session_state.data_loaded:
        st.warning("⏸️ Data not loaded")
        if st.button("📂 Load Sales Data", use_container_width=True):
            with st.spinner("Loading data..."):
                st.session_state.agent = FinancialAgentLangChain(api_key=api_key, data_path=str(data_path))
                result = st.session_state.agent.load_data()

                if result['success']:
                    st.session_state.data_loaded = True
                    st.success("✅ Data loaded!")
                    st.rerun()
                else:
                    st.error(f"Error: {result['message']}")
    else:
        st.success("✅ Data loaded")

        # Show data stats
        st.metric("Transactions", f"{st.session_state.agent.summary_stats['row_count']:,}")
        st.metric("Total Revenue", f"${st.session_state.agent.summary_stats['total_revenue']:,.2f}")
        st.metric("Products", st.session_state.agent.summary_stats['unique_products'])

        if st.button("🔄 Reload Data", use_container_width=True):
            st.session_state.data_loaded = False
            st.session_state.agent = None
            st.session_state.chat_messages = []
            st.rerun()

    st.divider()

    st.header("💡 Example Questions")
    st.markdown("""
    Try asking:
    - "What were the top 5 products?"
    - "Show me revenue by month"
    - "Which customer segment bought the most?"
    - "Forecast Q4 revenue"
    - "Find all Wireless Headphones sales"
    - "Calculate average transaction value"
    - "What products are declining?"
    """)

    st.divider()

    st.header("🛠️ AI Tools Available")
    st.markdown("""
    The agent has access to:
    1. **QuerySalesData** - Search with citations
    2. **CalculateStatistics** - Compute metrics
    3. **FindSpecificData** - Get exact rows
    4. **ForecastRevenue** - Prophet ML model
    """)

# Main content area
if not st.session_state.data_loaded:
    # Welcome screen
    st.info("👈 Click 'Load Sales Data' in the sidebar to get started!")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        ### 🎯 What's Different?
        - **LangChain-powered** agent
        - **Tool-based reasoning** (visible!)
        - **RAG with citations** to source data
        - **Interactive chat** interface
        """)

    with col2:
        st.markdown("""
        ### 🔍 Transparent AI
        - See which tools the AI uses
        - View actual data queries
        - Citations with row numbers
        - Real-time reasoning display
        """)

    with col3:
        st.markdown("""
        ### 🚀 Capabilities
        - Natural language queries
        - Complex multi-step analysis
        - ML-based forecasting
        - Automated report generation
        """)

else:
    # Main interface with tabs
    tab1, tab2, tab3 = st.tabs(["💬 Interactive Chat", "📊 Automated Report", "📈 Visualizations"])

    # TAB 1: Interactive Chat
    with tab1:
        st.markdown("### Ask the AI Analyst Anything")
        st.caption("The AI will use tools to retrieve data and cite its sources")

        # Display only the latest Q&A (if exists)
        if len(st.session_state.chat_messages) >= 2:
            # Get last user message and last assistant response
            last_messages = st.session_state.chat_messages[-2:]

            for msg in last_messages:
                if msg['role'] == 'user':
                    st.markdown(f"""
                    <div class="chat-message user-message">
                        <strong>👤 You:</strong><br>
                        {msg['content']}
                    </div>
                    """, unsafe_allow_html=True)

                elif msg['role'] == 'assistant':
                    # Show tool usage if available
                    if msg.get('tools_used'):
                        with st.expander(f"🔧 Tools Used ({len(msg['tools_used'])})", expanded=False):
                            for tool in msg['tools_used']:
                                st.markdown(f"""
                                <div class="tool-usage">
                                    <strong>🛠️ {tool['tool']}</strong><br>
                                    <em>Input:</em> {tool['input']}<br>
                                    <em>Output:</em> <pre>{tool['output'][:200]}...</pre>
                                </div>
                                """, unsafe_allow_html=True)

                    # Show AI response
                    st.markdown(f"""
                    <div class="chat-message ai-message">
                        <strong>🤖 AI Analyst:</strong><br>
                        {msg['content']}
                    </div>
                    """, unsafe_allow_html=True)

                    # Show citation dataframe if available
                    if msg.get('citation_dataframe') is not None and not msg['citation_dataframe'].empty:
                        with st.expander(f"📊 View Source Data (aggregated)", expanded=False):
                            st.caption("📚 This table shows the aggregated data that supports the AI's answer")
                            st.caption("👇 Click on any row below to see the underlying transactions")

                            # Display the citation dataframe with selection
                            event = st.dataframe(
                                msg['citation_dataframe'],
                                use_container_width=True,
                                height=400,
                                on_select="rerun",
                                selection_mode="single-row"
                            )

                            # Check if a row was selected
                            if event.selection and len(event.selection.get('rows', [])) > 0:
                                selected_row_idx = event.selection['rows'][0]
                                selected_index = msg['citation_dataframe'].index[selected_row_idx]

                                st.divider()
                                st.markdown(f"### 🔍 Drill-Down: **{selected_index}**")
                                st.caption("Showing all individual transactions for this item")

                                # Get the underlying transactions based on what was selected
                                df = st.session_state.agent.df

                                # Determine the filter based on the index name
                                if selected_index in df['product'].unique():
                                    # Product drill-down
                                    filtered_df = df[df['product'] == selected_index].copy()
                                elif selected_index in df['category'].unique():
                                    # Category drill-down
                                    filtered_df = df[df['category'] == selected_index].copy()
                                elif selected_index in df['customer_segment'].unique():
                                    # Customer segment drill-down
                                    filtered_df = df[df['customer_segment'] == selected_index].copy()
                                elif selected_index in ['July', 'August', 'September']:
                                    # Month drill-down
                                    filtered_df = df[df['date'].dt.strftime('%B') == selected_index].copy()
                                else:
                                    filtered_df = df.copy()

                                # Calculate metrics BEFORE formatting
                                total_transactions = len(filtered_df)
                                total_rev = filtered_df['revenue'].sum()
                                total_qty = filtered_df['quantity'].sum()

                                # Now format the detailed dataframe for display
                                display_detail = filtered_df[['date', 'product', 'quantity', 'revenue', 'cost', 'category', 'customer_segment']].copy()
                                display_detail['date'] = display_detail['date'].dt.strftime('%Y-%m-%d')
                                display_detail['revenue'] = display_detail['revenue'].apply(lambda x: f"${x:,.2f}")
                                display_detail['cost'] = display_detail['cost'].apply(lambda x: f"${x:,.2f}")

                                # Show metrics
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Total Transactions", total_transactions)
                                with col2:
                                    st.metric("Total Revenue", f"${total_rev:,.2f}")
                                with col3:
                                    st.metric("Total Quantity", f"{total_qty:,}")

                                # Display the transactions
                                st.dataframe(
                                    display_detail,
                                    use_container_width=True,
                                    height=300
                                )

                            st.caption("✅ This is the actual data source used to generate the answer above")

                    # Show pandas code execution if available
                    if msg.get('pandas_code'):
                        with st.expander("🐼 Generated Pandas Code", expanded=False):
                            st.code(msg['pandas_code'], language='python')

                            if st.button(f"▶️ Execute Code", key=f"exec_{len(st.session_state.chat_messages)}"):
                                with st.spinner("Executing..."):
                                    result = st.session_state.agent.execute_pandas_code(msg['pandas_code'])

                                    if result['success']:
                                        st.success("✅ Execution successful")

                                        if result['result_type'] in ['dataframe', 'series']:
                                            st.dataframe(result['result'], use_container_width=True)
                                        else:
                                            st.write(result['result'])
                                    else:
                                        st.error(f"Error: {result.get('error', 'Unknown error')}")
        else:
            # No messages yet - show welcome
            st.info("👋 Ask a question about your Q3 2025 sales data to get started!")

        # Chat input
        st.divider()

        col1, col2 = st.columns([5, 1])

        with col1:
            user_input = st.text_input(
                "Your question:",
                key="chat_input",
                placeholder="e.g., What were our top products in September?",
                label_visibility="collapsed"
            )

        with col2:
            send_button = st.button("📤 Send", use_container_width=True, type="primary")

        # Quick action buttons
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("📊 Top Products", use_container_width=True):
                user_input = "What were the top 5 products by revenue? Include specific numbers and citations."
                send_button = True

        with col2:
            if st.button("📈 Monthly Trend", use_container_width=True):
                user_input = "Show me the monthly revenue trend from July to September."
                send_button = True

        with col3:
            if st.button("💼 Best Segment", use_container_width=True):
                user_input = "Which customer segment generated the most revenue?"
                send_button = True

        col4, col5, col6 = st.columns(3)

        with col4:
            if st.button("🔮 Forecast Q4", use_container_width=True):
                user_input = "Forecast Q4 revenue using Prophet and tell me the expected growth rate."
                send_button = True

        with col5:
            if st.button("🎧 Find Headphones Sales", use_container_width=True):
                user_input = "Find all sales for 'Wireless Headphones'."
                send_button = True

        with col6:
            if st.button("📉 Declining Products", use_container_width=True):
                user_input = "What products are declining in revenue?"
                send_button = True

        # Process user input
        if send_button and user_input:
            # Add user message
            st.session_state.chat_messages.append({
                'role': 'user',
                'content': user_input
            })

            # Show thinking indicator
            with st.spinner("🤔 AI is thinking and using tools..."):
                result = st.session_state.agent.chat(user_input)

                if result['success']:
                    # Add AI response with tool usage, citation dataframe, and pandas code
                    st.session_state.chat_messages.append({
                        'role': 'assistant',
                        'content': result['response'],
                        'tools_used': result['intermediate_steps'],
                        'citation_dataframe': result.get('citation_dataframe'),
                        'pandas_code': result.get('pandas_code')
                    })
                else:
                    st.error(f"Error: {result['response']}")

            st.rerun()

        # Show history button if there are previous messages
        if len(st.session_state.chat_messages) > 2:
            with st.expander(f"📜 View Previous Queries ({len(st.session_state.chat_messages)//2 - 1} older)", expanded=False):
                # Show all except the last 2
                for msg in st.session_state.chat_messages[:-2]:
                    if msg['role'] == 'user':
                        st.caption(f"**Q:** {msg['content']}")
                    elif msg['role'] == 'assistant':
                        st.caption(f"**A:** {msg['content'][:100]}...")
                        st.divider()

    # TAB 2: Automated Report
    with tab2:
        st.markdown("### 📋 Automated Comprehensive Analysis")
        st.caption("Let the AI generate a full report using all available tools")

        if not st.session_state.report_generated:
            if st.button("🚀 Generate Comprehensive Report", type="primary", use_container_width=True):
                with st.spinner("Generating comprehensive report... This may take 30-60 seconds."):

                    # Show progress
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    status_text.text("🔄 Initializing agent...")
                    progress_bar.progress(20)

                    status_text.text("📊 Analyzing sales data...")
                    progress_bar.progress(40)

                    # Generate report
                    result = st.session_state.agent.generate_comprehensive_report()

                    status_text.text("🔍 Gathering insights...")
                    progress_bar.progress(70)

                    # Generate visualizations
                    st.session_state.figures = st.session_state.agent.generate_visualizations()

                    status_text.text("✨ Finalizing report...")
                    progress_bar.progress(100)

                    time.sleep(0.5)
                    progress_bar.empty()
                    status_text.empty()

                    if result['success']:
                        # Store report
                        st.session_state.report_generated = True
                        st.session_state.report_content = result
                        st.rerun()
                    else:
                        st.error(f"Error generating report: {result['response']}")

        else:
            # Display report
            report = st.session_state.report_content

            # Show tool usage
            with st.expander("🔧 Tools Used by AI Agent", expanded=True):
                for i, step in enumerate(report['intermediate_steps'], 1):
                    st.markdown(f"""
                    **Step {i}: {step['tool']}**

                    *Input:* `{step['input']}`

                    *Output:*
                    ```
                    {step['output'][:300]}...
                    ```
                    """)
                    st.divider()

            # Show report content
            st.markdown("### 📄 Report")
            st.markdown(report['response'])

            # Action buttons
            if st.button("🔄 Generate New Report", use_container_width=True):
                st.session_state.report_generated = False
                st.rerun()

    # TAB 3: Visualizations
    with tab3:
        st.markdown("### 📊 Data Visualizations")

        if st.session_state.figures is None:
            if st.button("📈 Generate Charts", type="primary", use_container_width=True):
                with st.spinner("Creating visualizations..."):
                    st.session_state.figures = st.session_state.agent.generate_visualizations()
                    st.rerun()
        else:
            # Display charts
            for i, fig in enumerate(st.session_state.figures):
                st.plotly_chart(fig, use_container_width=True)
                if i < len(st.session_state.figures) - 1:
                    st.divider()

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #64748b; font-size: 0.9em;'>
    Powered by LangChain + Google Gemini + Prophet •
    Real-time tool usage • Citations to source data •
    <a href='https://github.com' style='color: #2563eb;'>View Source</a>
</div>
""", unsafe_allow_html=True)
