"""
Test script to verify RAG agent returns sources
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from agents.rag_chatbot_agent import RAGChatbotAgent
from utils.secrets_manager import get_api_key

def test_rag_agent():
    print("=" * 60)
    print("Testing RAG Agent - Source Retrieval")
    print("=" * 60)

    # Get API key
    api_key = get_api_key("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: GOOGLE_API_KEY not found")
        return

    # Setup paths
    documents_path = Path(__file__).parent / "data" / "papers"
    print(f"\nDocuments path: {documents_path}")
    print(f"ChromaDB path: {documents_path / 'rag_chroma'}")

    # Initialize agent
    print("\nLoading RAG agent...")
    agent = RAGChatbotAgent(
        api_key=api_key,
        documents_path=str(documents_path),
        model="gemini-2.0-flash-exp"
    )

    # Load documents
    print("Loading vector store...")
    result = agent.load_documents()
    print(f"Load result: {result}")

    # Get stats
    stats = agent.get_stats()
    print(f"\nStats:")
    print(f"   - Total chunks: {stats['total_chunks']}")
    print(f"   - Vector store loaded: {stats['vector_store_loaded']}")
    print(f"   - Retriever ready: {stats['retriever_ready']}")

    # Test query
    test_question = "What is the self-attention mechanism?"
    print(f"\nTest question: {test_question}")
    print("\n" + "=" * 60)
    print("Response:")
    print("=" * 60)

    sources_found = []
    full_response = ""

    for chunk in agent.chat_stream(test_question):
        if chunk["type"] == "sources":
            sources_found = chunk["content"]
            print(f"\nSOURCES FOUND: {len(sources_found)} documents")
            for i, source in enumerate(sources_found, 1):
                print(f"\n   Source {i}:")
                print(f"   - File: {source['source']}")
                print(f"   - Page: {source['page']}")
                print(f"   - Content preview: {source['content'][:100]}...")

        elif chunk["type"] == "token":
            full_response += chunk["content"]
            print(chunk["content"], end="", flush=True)

        elif chunk["type"] == "complete":
            print("\n\n" + "=" * 60)
            print(f"Response complete!")
            print(f"Total sources: {len(sources_found)}")

        elif chunk["type"] == "error":
            print(f"\nError: {chunk['content']}")

    # Final check
    print("\n" + "=" * 60)
    if sources_found:
        print(f"SUCCESS: {len(sources_found)} sources retrieved!")
    else:
        print("FAILURE: No sources found! Vector store needs rebuilding.")
    print("=" * 60)

if __name__ == "__main__":
    test_rag_agent()
