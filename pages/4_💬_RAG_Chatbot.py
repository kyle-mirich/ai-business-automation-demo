"""
RAG Chatbot with Streaming - AI Research Papers

Features:
- Chat with AI research papers using RAG
- Real-time streaming responses
- Source citations with page numbers
- Conversation memory
- Beautiful chat UI with st.chat_message
"""

import streamlit as st
import sys
from pathlib import Path
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.rag_chatbot_agent import RAGChatbotAgent
from utils.secrets_manager import get_api_key, display_api_key_error
from utils.cost_calculator import estimate_tokens, calculate_gemini_cost

# Page config
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="💬",
    layout="wide"
)

# Custom CSS for chat
st.markdown("""
<style>
.source-card {
    background-color: #f8f9fa;
    border-left: 4px solid #2563eb;
    padding: 12px;
    margin: 8px 0;
    border-radius: 4px;
}
.source-header {
    font-weight: 600;
    color: #1e40af;
    margin-bottom: 4px;
}
.source-content {
    font-size: 0.9em;
    color: #334155;
    line-height: 1.5;
}
</style>
""", unsafe_allow_html=True)

# Title
st.title("💬 RAG Chatbot: AI Research Papers")
st.markdown("""
Chat with AI research papers using **Retrieval Augmented Generation (RAG)**.
Ask questions and get answers with **source citations** and **streaming responses**.
""")

st.divider()

# Check for API key
api_key = get_api_key("GOOGLE_API_KEY")
if not api_key:
    display_api_key_error()
    st.stop()

# Documents path
documents_path = Path(__file__).parent.parent / "data" / "papers"

# Session state initialization
if "rag_agent" not in st.session_state:
    st.session_state.rag_agent = None
if "rag_loaded" not in st.session_state:
    st.session_state.rag_loaded = False
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "rag_total_cost" not in st.session_state:
    st.session_state.rag_total_cost = 0.0
if "rag_query_count" not in st.session_state:
    st.session_state.rag_query_count = 0

# Suggested questions for foundational papers
SUGGESTED_QUESTIONS = [
    "What is the self-attention mechanism and how does it work?",
    "Explain the transformer architecture from 'Attention is All You Need'",
    "What are the key differences between BERT and GPT?",
    "How does multi-head attention improve model performance?",
    "What is positional encoding and why is it necessary?",
    "How does BERT's masked language modeling differ from GPT's approach?",
    "What is the Vision Transformer and how does it apply transformers to images?",
    "Explain the role of feed-forward networks in transformers",
]

# Auto-load the pre-built vector store on first visit
if not st.session_state.rag_loaded and st.session_state.rag_agent is None:
    with st.spinner("Loading foundational AI papers database..."):
        # Initialize agent
        st.session_state.rag_agent = RAGChatbotAgent(
            api_key=api_key,
            documents_path=str(documents_path),
            model="gemini-2.0-flash-exp"
        )

        # Load documents from pre-built vector store
        result = st.session_state.rag_agent.load_documents()

        if result['success']:
            st.session_state.rag_loaded = True
        else:
            st.error(f"Failed to load papers: {result['message']}")
            st.stop()

# Sidebar
with st.sidebar:
    st.header("📚 Database Info")

    st.success("✅ Vector database loaded")

    # Show stats
    if st.session_state.rag_agent:
        stats = st.session_state.rag_agent.get_stats()
        st.metric("Document Chunks", f"{stats['total_chunks']:,}")
        st.metric("Papers Loaded", "6 foundational AI papers")
        st.metric("Conversation Turns", stats['conversation_length'] // 2 if stats['conversation_length'] > 0 else 0)

    st.divider()

    # Paper info
    st.subheader("📄 Papers in Database")
    st.markdown("""
    - Attention Is All You Need
    - BERT (Pre-training of Deep)
    - GPT-3 (Language Models)
    - Vision Transformer (ViT)
    - RoBERTa
    - And more foundational papers
    """)

    st.divider()

    # Clear chat history
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.rag_agent.clear_history()
        st.session_state.chat_messages = []
        st.session_state.rag_total_cost = 0.0
        st.session_state.rag_query_count = 0
        st.success("Chat history cleared!")
        time.sleep(1)
        st.rerun()

    st.divider()

    # Cost tracking
    st.header("💰 Session Costs")
    st.metric("Queries", st.session_state.rag_query_count)
    st.metric("Total Cost", f"${st.session_state.rag_total_cost:.6f}")
    if st.session_state.rag_query_count > 0:
        avg_cost = st.session_state.rag_total_cost / st.session_state.rag_query_count
        st.metric("Avg per Query", f"${avg_cost:.6f}")

    st.divider()

    # Suggested questions (dynamic, shows 3-4 random questions)
    if st.session_state.rag_loaded:
        st.header("💡 Suggested Questions")
        st.markdown("*Click any question to ask it*")

        # Show 4 random suggested questions
        import random
        suggested = random.sample(SUGGESTED_QUESTIONS, min(4, len(SUGGESTED_QUESTIONS)))

        for i, question in enumerate(suggested, 1):
            if st.button(f"{i}. {question}", use_container_width=True, key=f"suggest_{i}"):
                # Simulate click by adding to chat
                st.session_state.chat_messages.append({
                    "role": "user",
                    "content": question
                })
                st.rerun()
    else:
        st.header("💡 What to Ask")
        st.markdown("""
        Once papers are loaded, you can ask questions like:
        - Explain the transformer architecture
        - What is self-attention?
        - How does BERT differ from GPT?
        - What is positional encoding?
        """)

# Main chat area header
st.markdown("""
### Chat with Foundational AI Papers 🤖

Ask questions about transformers, attention mechanisms, BERT, GPT, and more.
The chatbot retrieves answers from research papers and provides citations.
""")

# Display chat messages
for message in st.session_state.chat_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Show sources if available
        if message.get("sources"):
            with st.expander(f"📚 Sources ({len(message['sources'])} documents)", expanded=False):
                for i, source in enumerate(message["sources"], 1):
                    st.markdown(f"""
                    <div class="source-card">
                        <div class="source-header">Source {i}: {source['source']} (Page {source['page']})</div>
                        <div class="source-content">{source['content']}</div>
                    </div>
                    """, unsafe_allow_html=True)

        # Show cost if available
        if message.get("cost"):
            cost = message["cost"]
            st.caption(
                f"💰 Cost: ${cost['total_cost']:.6f} | "
                f"Input: {cost['input_tokens']:,} tokens | "
                f"Output: {cost['output_tokens']:,} tokens"
            )

# Chat input
if prompt := st.chat_input("Ask a question about the research papers..."):
    # Add user message
    st.session_state.chat_messages.append({
        "role": "user",
        "content": prompt
    })

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response with streaming
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        sources_placeholder = st.empty()
        cost_placeholder = st.empty()

        full_response = ""
        sources = []

        # Stream the response
        for chunk in st.session_state.rag_agent.chat_stream(prompt):
            if chunk["type"] == "sources":
                sources = chunk["content"]

            elif chunk["type"] == "token":
                full_response += chunk["content"]
                response_placeholder.markdown(full_response + "▌")

            elif chunk["type"] == "complete":
                full_response = chunk["content"]
                sources = chunk.get("sources", sources)
                response_placeholder.markdown(full_response)

            elif chunk["type"] == "error":
                response_placeholder.error(chunk["content"])
                break

        # Calculate cost
        input_tokens = estimate_tokens(prompt)
        output_tokens = estimate_tokens(full_response)
        cost_info = calculate_gemini_cost(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model="gemini-2.0-flash-exp"
        )

        # Update session totals
        st.session_state.rag_total_cost += cost_info['total_cost']
        st.session_state.rag_query_count += 1

        # Display sources
        if sources:
            with sources_placeholder.expander(f"📚 Sources ({len(sources)} documents)", expanded=False):
                for i, source in enumerate(sources, 1):
                    st.markdown(f"""
                    <div class="source-card">
                        <div class="source-header">Source {i}: {source['source']} (Page {source['page']})</div>
                        <div class="source-content">{source['content']}</div>
                    </div>
                    """, unsafe_allow_html=True)

        # Display cost
        cost_placeholder.caption(
            f"💰 Cost: ${cost_info['total_cost']:.6f} | "
            f"Input: {cost_info['input_tokens']:,} tokens | "
            f"Output: {cost_info['output_tokens']:,} tokens"
        )

        # Save assistant message
        st.session_state.chat_messages.append({
            "role": "assistant",
            "content": full_response,
            "sources": sources,
            "cost": cost_info
        })

# Footer
st.divider()
st.caption("💡 **Tip:** The chatbot remembers conversation context. Ask follow-up questions!")
