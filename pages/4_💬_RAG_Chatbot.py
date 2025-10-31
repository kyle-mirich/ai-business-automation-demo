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
import html
from typing import List, Dict, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.rag_chatbot_agent import RAGChatbotAgent
from utils.secrets_manager import get_api_key, display_api_key_error
from utils.cost_calculator import estimate_tokens, calculate_gemini_cost

# Cache the RAG agent initialization
@st.cache_resource
def load_rag_agent(api_key: str, documents_path: str) -> RAGChatbotAgent:
    """Load RAG agent with vector store (cached)."""
    agent = RAGChatbotAgent(
        api_key=api_key,
        documents_path=documents_path,
        model="gemini-2.5-flash"
    )
    # Load the vector store
    agent.last_load_result = agent.load_documents()
    return agent

# Page config
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="💬",
    layout="wide"
)

# Custom CSS for chat and sources
st.markdown("""
<style>
.source-card {
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    border-left: 5px solid #2563eb;
    padding: 16px;
    margin: 12px 0;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    transition: all 0.3s ease;
}
.source-card:hover {
    box-shadow: 0 4px 8px rgba(37, 99, 235, 0.15);
    transform: translateX(2px);
}
.source-header {
    font-weight: 600;
    color: #1e40af;
    margin-bottom: 8px;
    font-size: 1em;
    display: flex;
    align-items: center;
    gap: 8px;
}
.source-content {
    font-size: 0.92em;
    color: #475569;
    line-height: 1.6;
    font-style: italic;
    background-color: white;
    padding: 10px;
    border-radius: 4px;
    margin-top: 8px;
}
.source-links {
    font-size: 0.9em;
    color: #1e3a8a;
    margin-bottom: 10px;
}
.rag-highlight {
    background: rgba(37,99,235,0.08);
    border-left: 4px solid #2563eb;
    padding: 14px 18px;
    border-radius: 8px;
    margin-top: 12px;
    line-height: 1.6;
    color: #1f2937;
}
.rag-highlight.rag-highlight-active {
    border-left-color: #f97316;
    background: rgba(249,115,22,0.1);
}
.rag-highlight mark {
    background-color: #fde68a;
    color: #1f2937;
    padding: 0 3px;
    border-radius: 3px;
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
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

# Auto-load the pre-built vector store on first visit
if not st.session_state.rag_loaded:
    with st.spinner("Loading foundational AI papers database..."):
        try:
            # Load cached agent with pre-built vector store
            st.session_state.rag_agent = load_rag_agent(api_key, str(documents_path))
            st.session_state.rag_loaded = True

            # Get chunk count for display
            load_result = getattr(st.session_state.rag_agent, "last_load_result", {}) or {}
            chunk_count = load_result.get("chunk_count", 0)

            if chunk_count > 0:
                st.toast(f"✅ Loaded {chunk_count:,} document chunks from vector store", icon="📦")
            else:
                st.warning("⚠️ Vector store loaded but appears empty. You may need to rebuild it using the sidebar button.")
        except Exception as e:
            st.error(f"Failed to load vector database: {str(e)}")
            st.info("💡 The vector store may need to be rebuilt. Use the 'Rebuild Vector Store' button in the sidebar.")
            st.stop()

# Sidebar
with st.sidebar:
    st.header("📚 Database Info")

    st.success("✅ Vector database loaded")

    # Show stats
    if st.session_state.rag_agent:
        stats = st.session_state.rag_agent.get_stats()
        st.metric("Document Chunks", f"{stats['total_chunks']:,}")
        st.metric("Papers Loaded", "12 foundational AI papers")
        st.metric("Conversation Turns", stats['conversation_length'] // 2 if stats['conversation_length'] > 0 else 0)
        if stats["total_chunks"] == 0:
            st.warning(
                "Vector store currently empty. Rebuild the index to unlock grounded answers.",
                icon="⚠️",
            )

        # Show ChromaDB path for debugging
        chroma_path = Path(st.session_state.rag_agent.persist_directory)
        with st.expander("🔧 Debug Info"):
            st.caption(f"**ChromaDB Path:**")
            st.code(str(chroma_path))
            st.caption(f"**Documents Path:**")
            st.code(str(documents_path))

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
        st.session_state.pending_prompt = None
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


# Main chat area header
st.markdown("""
### Chat with Foundational AI Papers 🤖

Ask questions about transformers, attention mechanisms, BERT, GPT, and more.
The chatbot retrieves answers from research papers and provides citations.
""")


# Function to render source details
def render_source_details(
    sources: List[Dict[str, Any]],
    key_prefix: str,
    default_expanded: bool = False,
) -> None:
    """Show retrieved chunks with full text so users can see raw context."""
    if not sources:
        return

    # Outer expander containing all sources
    with st.expander(f"📚 **View {len(sources)} Source Document{'s' if len(sources) != 1 else ''}**", expanded=default_expanded):
        for idx, source in enumerate(sources, 1):
            doc_name = source.get('source', 'Document')
            page = source.get('page', 'N/A')
            source_url = source.get('source_url')

            chunk_text = source.get("chunk_full") or source.get("content") or ""
            highlight = source.get("highlighted_excerpt") or source.get("chunk_excerpt") or ""
            score = source.get("score")
            file_path = source.get("source_path")
            page_num = source.get('page', 1)

            # Build metadata string
            meta_bits = []
            if score is not None:
                meta_bits.append(f"Relevance: {score:.3f}")
            meta_str = " • ".join(meta_bits) if meta_bits else ""

            # Build GitHub link button HTML
            github_link_html = ""
            if source_url:
                github_link_html = f"""
                <div style="margin: 10px 0;">
                    <a href="{source_url}" target="_blank" style="
                        display: inline-block;
                        padding: 8px 16px;
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white;
                        text-decoration: none;
                        border-radius: 6px;
                        font-weight: 500;
                        transition: all 0.3s ease;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    ">
                        📄 View {html.escape(doc_name)} on GitHub (page {page_num})
                    </a>
                    <p style="font-size: 0.85em; color: #64748b; margin-top: 5px;">
                        💡 Opens in GitHub's PDF viewer - manually navigate to page {page_num}
                    </p>
                </div>
                """

            # Build highlighted excerpt HTML
            excerpt_html = ""
            if highlight:
                excerpt_html = f"""
                <div style="margin: 10px 0;">
                    <strong>Relevant excerpt:</strong>
                    <div style="margin-top: 5px;">{highlight}</div>
                </div>
                """

            # Create collapsible HTML with details/summary
            open_attr = 'open' if idx == 1 else ''
            html_content = f"""
            <details {open_attr} style="
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                padding: 12px;
                margin: 12px 0;
                background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
                box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            ">
                <summary style="
                    cursor: pointer;
                    font-weight: 600;
                    color: #1e40af;
                    font-size: 1.05em;
                    padding: 8px;
                    user-select: none;
                    list-style: none;
                ">
                    📄 Source {idx}: {html.escape(doc_name)} (page {page})
                    {f'<span style="color: #64748b; font-weight: normal; font-size: 0.9em;"> • {meta_str}</span>' if meta_str else ''}
                </summary>
                <div style="margin-top: 12px; padding: 8px;">
                    {github_link_html}
                    {excerpt_html}
                    <div style="margin: 10px 0;">
                        <strong>Full context:</strong>
                        <pre style="
                            background: #f1f5f9;
                            padding: 12px;
                            border-radius: 6px;
                            overflow-x: auto;
                            font-size: 0.9em;
                            line-height: 1.5;
                            margin-top: 8px;
                            border-left: 3px solid #3b82f6;
                        ">{html.escape(chunk_text)}</pre>
                    </div>
                    {f'<p style="font-size: 0.85em; color: #64748b; margin-top: 8px;">{html.escape(file_path)}</p>' if file_path else ''}
                </div>
            </details>
            """

            st.markdown(html_content, unsafe_allow_html=True)


# Show suggested prompts only if there are no messages yet
if not st.session_state.chat_messages:
    st.markdown("<h5 style='text-align: center; color: #64748b; margin-top: 2rem;'>Try one of these suggested prompts to get started:</h5>", unsafe_allow_html=True)

    # Define prompts with their button labels
    prompts = {
        "Self-Attention": "What is the self-attention mechanism and how does it work?",
        "Transformer Architecture": "Explain the transformer architecture from 'Attention is All You Need'",
        "BERT vs GPT": "What are the key differences between BERT and GPT?",
        "Multi-Head Attention": "How does multi-head attention improve model performance?",
        "Positional Encoding": "What is positional encoding and why is it necessary?",
        "BERT Masking": "How does BERT's masked language modeling differ from GPT's approach?",
        "Vision Transformer": "What is the Vision Transformer and how does it apply transformers to images?",
        "Feed-Forward": "Explain the role of feed-forward networks in transformers"
    }

    # First row
    cols = st.columns(4)
    for i, (label, prompt_text) in enumerate(list(prompts.items())[:4]):
        with cols[i]:
            if st.button(label, use_container_width=True, key=f"prompt_{label}"):
                st.session_state.pending_prompt = prompt_text
                st.rerun()

    # Second row
    cols2 = st.columns(4)
    for i, (label, prompt_text) in enumerate(list(prompts.items())[4:8]):
        with cols2[i]:
            if st.button(label, use_container_width=True, key=f"prompt_{label}"):
                st.session_state.pending_prompt = prompt_text
                st.rerun()

# Display chat messages
for idx, message in enumerate(st.session_state.chat_messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Show sources if available
        if message.get("sources"):
            render_source_details(message["sources"], key_prefix=f"history-{idx}", default_expanded=False)

        # Show cost if available
        if message.get("cost"):
            cost = message["cost"]
            st.caption(
                f"💰 Cost: ${cost['total_cost']:.6f} | "
                f"Input: {cost['input_tokens']:,} tokens | "
                f"Output: {cost['output_tokens']:,} tokens"
            )

# Chat input (fixed to bottom)
user_input = st.chat_input("Ask a question about AI research papers...")

# Check if there's a pending prompt from suggested buttons
if "pending_prompt" in st.session_state and st.session_state.pending_prompt:
    user_input = st.session_state.pending_prompt
    st.session_state.pending_prompt = None

# Process user input
if user_input:
    prompt = user_input
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
            model="gemini-2.5-flash"
        )

        # Update session totals
        st.session_state.rag_total_cost += cost_info['total_cost']
        st.session_state.rag_query_count += 1

        # Display sources with improved formatting
        if sources:
            with sources_placeholder.container():
                key_prefix = f"live-{len(st.session_state.chat_messages)}"
                render_source_details(sources, key_prefix=key_prefix, default_expanded=True)

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
