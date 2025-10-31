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
if "used_prompts" not in st.session_state:
    st.session_state.used_prompts = set()
if "rag_auto_rebuilt" not in st.session_state:
    st.session_state.rag_auto_rebuilt = False

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
if not st.session_state.rag_loaded:
    with st.spinner("Loading foundational AI papers database..."):
        try:
            # Load cached agent (vector store is cached)
            st.session_state.rag_agent = load_rag_agent(api_key, str(documents_path))
            st.session_state.rag_loaded = True

            # Debug: Show where ChromaDB is loading from
            chroma_path = Path(st.session_state.rag_agent.persist_directory)
            st.toast(f"✅ Loaded ChromaDB from: {chroma_path}", icon="📦")
        except Exception as e:
            st.error(f"Failed to load papers: {str(e)}")
            st.stop()

if (
    st.session_state.rag_agent
    and not st.session_state.rag_auto_rebuilt
):
    load_result = getattr(st.session_state.rag_agent, "last_load_result", {}) or {}
    if load_result.get("chunk_count", 0) == 0:
        with st.spinner("Vector store empty — rebuilding embeddings from PDFs..."):
            rebuild_result = st.session_state.rag_agent.load_documents(force_reload=True)
        st.session_state.rag_agent.last_load_result = rebuild_result
        if rebuild_result.get("success") and rebuild_result.get("chunk_count", 0) > 0:
            st.toast(
                f"Rebuilt vector store with {rebuild_result.get('chunk_count', 0):,} chunks.",
                icon="🧠",
            )
            # Reset conversation so new retrieval powers the chat
            st.session_state.chat_messages = []
            st.session_state.rag_total_cost = 0.0
            st.session_state.rag_query_count = 0
        else:
            st.warning(
                "The vector store is currently empty. Use the **Rebuild Vector Store** button "
                "in the sidebar once your PDFs are available."
            )
        st.session_state.rag_auto_rebuilt = True

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
        st.session_state.used_prompts = set()  # Reset used prompts
        st.success("Chat history and suggested prompts cleared!")
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

    if st.session_state.rag_agent:
        if st.button("♻️ Rebuild Vector Store", use_container_width=True):
            with st.spinner("Re-indexing papers with fresh chunks..."):
                result = st.session_state.rag_agent.load_documents(force_reload=True)
            if result.get("success"):
                st.session_state.rag_agent.last_load_result = result
                st.success(
                    f"Rebuilt vector store with {result.get('chunk_count', 0):,} chunks "
                    f"from {result.get('pdf_count', 0)} papers."
                )
                st.session_state.chat_messages = []
                st.session_state.rag_total_cost = 0.0
                st.session_state.rag_query_count = 0
                st.session_state.used_prompts = set()
                st.session_state.rag_auto_rebuilt = True
                st.info("Chat history cleared so responses use the refreshed embeddings.")
                time.sleep(1)
                st.rerun()
            else:
                st.error(result.get("message", "Unable to rebuild vector store."))

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

    for idx, source in enumerate(sources, 1):
        doc_name = source.get('source', 'Document')
        page = source.get('page', 'N/A')
        source_url = source.get('source_url')

        title = f"Source {idx}: `{doc_name}` (page {page})"
        chunk_text = source.get("chunk_full") or source.get("content") or ""
        highlight = source.get("highlighted_excerpt") or source.get("chunk_excerpt") or ""

        with st.expander(title, expanded=default_expanded and idx == 1):
            # Show clickable link to navigate to GitHub PDF
            if source_url:
                page_num = source.get('page', 1)

                # Use st.link_button to open GitHub PDF in new tab
                st.link_button(
                    f"📄 View {doc_name} on GitHub (page {page_num})",
                    source_url,
                    type="secondary"
                )
                st.caption(f"💡 Opens in GitHub's PDF viewer - manually navigate to page {page_num}")
                st.markdown("---")

            if highlight:
                st.markdown(highlight, unsafe_allow_html=True)
            st.code(chunk_text, language="text")
            score = source.get("score")
            meta_bits = []
            if score is not None:
                meta_bits.append(f"Relevance score {score:.3f}")
            file_path = source.get("source_path")
            if file_path:
                meta_bits.append(file_path)
            if meta_bits:
                st.caption(" • ".join(meta_bits))


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

# Chat input area (similar to Financial Report)
st.divider()

col1, col2 = st.columns([5, 1])

with col1:
    user_input = st.text_input(
        "Your question:",
        key="chat_input",
        placeholder="e.g., What is the self-attention mechanism?",
        label_visibility="collapsed"
    )

with col2:
    send_button = st.button("📤 Send", use_container_width=True, type="primary")

st.markdown("<h5 style='text-align: center; color: #64748b;'>Or try one of these suggested prompts:</h5>", unsafe_allow_html=True)

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

# Filter out used prompts
available_prompts = {k: v for k, v in prompts.items() if k not in st.session_state.used_prompts}

# Quick action buttons - First row
if available_prompts:
    available_keys = list(available_prompts.keys())
    cols = st.columns(4)

    for i, (label, prompt_text) in enumerate(available_prompts.items()):
        if i < 4:  # First row
            with cols[i]:
                if st.button(label, use_container_width=True, key=f"prompt_{label}"):
                    user_input = prompt_text
                    send_button = True
                    st.session_state.used_prompts.add(label)

    # Second row
    if len(available_prompts) > 4:
        cols2 = st.columns(4)
        for i, (label, prompt_text) in enumerate(list(available_prompts.items())[4:]):
            if i < 4:  # Second row
                with cols2[i]:
                    if st.button(label, use_container_width=True, key=f"prompt_{label}"):
                        user_input = prompt_text
                        send_button = True
                        st.session_state.used_prompts.add(label)
else:
    st.info("💡 All suggested prompts have been used! Feel free to ask your own questions.")

# Process user input
if send_button and user_input:
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

# Footer
st.divider()
st.caption("💡 **Tip:** The chatbot remembers conversation context. Ask follow-up questions!")
