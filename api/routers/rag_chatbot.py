"""
RAG Chatbot API Router
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import os
from typing import AsyncGenerator

from api.models import RAGQueryRequest, RAGQueryResponse, SourceDocument
from agents.rag_chatbot_agent import RAGChatbotAgent

router = APIRouter()

# Initialize RAG agent (singleton)
_rag_agent = None


def get_rag_agent() -> RAGChatbotAgent:
    """Get or create RAG agent instance"""
    global _rag_agent
    if _rag_agent is None:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="GOOGLE_API_KEY not configured")

        documents_path = os.path.join(os.path.dirname(__file__), "../../data/papers")
        _rag_agent = RAGChatbotAgent(
            api_key=api_key,
            documents_path=documents_path,
            model="gemini-2.5-flash"
        )
        # Try to auto-load existing ChromaDB vector store
        # Don't fail here - let the query method handle loading if needed
        try:
            _rag_agent.load_documents(force_reload=False)
        except Exception as e:
            print(f"Warning: Could not preload documents: {str(e)}")
            # Agent will try to load on first query
    return _rag_agent


@router.post("/query", response_model=RAGQueryResponse)
async def query_rag(request: RAGQueryRequest):
    """
    Query the RAG chatbot with a question

    Returns answer with source citations
    """
    try:
        agent = get_rag_agent()

        # Convert conversation history to the format expected by the agent
        history = []
        for msg in request.conversation_history:
            if msg.get("role") == "user":
                history.append({"type": "human", "content": msg.get("content", "")})
            elif msg.get("role") == "assistant":
                history.append({"type": "ai", "content": msg.get("content", "")})

        # Get response from agent
        result = agent.query(
            query=request.query,
            conversation_history=history,
            top_k=request.top_k
        )

        # Format sources
        sources = []
        for doc in result.get("sources", []):
            sources.append(SourceDocument(
                content=doc.get("content", ""),
                metadata=doc.get("metadata", {}),
                page=doc.get("metadata", {}).get("page"),
                source=doc.get("metadata", {}).get("source")
            ))

        return RAGQueryResponse(
            answer=result.get("answer", ""),
            sources=sources,
            tokens_used=result.get("tokens_used")
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@router.post("/query/stream")
async def query_rag_stream(request: RAGQueryRequest):
    """
    Query the RAG chatbot with streaming response

    Returns Server-Sent Events (SSE) stream
    """
    try:
        agent = get_rag_agent()

        # Convert conversation history
        history = []
        for msg in request.conversation_history:
            if msg.get("role") == "user":
                history.append({"type": "human", "content": msg.get("content", "")})
            elif msg.get("role") == "assistant":
                history.append({"type": "ai", "content": msg.get("content", "")})

        async def event_generator() -> AsyncGenerator[str, None]:
            """Generate SSE events"""
            try:
                for chunk in agent.query_stream(
                    query=request.query,
                    conversation_history=history,
                    top_k=request.top_k
                ):
                    if chunk:
                        yield f"data: {chunk}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                yield f"data: {{\"error\": \"{str(e)}\"}}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@router.get("/health")
async def health():
    """Check if RAG agent is initialized"""
    try:
        agent = get_rag_agent()
        return {
            "status": "healthy",
            "collection_name": agent.collection_name,
            "document_count": agent.vectorstore._collection.count() if hasattr(agent.vectorstore, '_collection') else None
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }
