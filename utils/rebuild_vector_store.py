"""
Rebuild Vector Store Utility

This standalone script rebuilds the ChromaDB vector store from PDF files.
Use this when you want to update the embeddings after adding/modifying papers.

Usage:
    python utils/rebuild_vector_store.py

After running:
    1. Verify the new vector store works
    2. Commit the changes: git add data/papers/rag_chroma/
    3. Push to deploy: git push
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.rag_chatbot_agent import RAGChatbotAgent
from utils.secrets_manager import get_api_key


def rebuild_vector_store():
    """Rebuild the vector store from PDF files."""
    print("=" * 60)
    print("ChromaDB Vector Store Rebuilder")
    print("=" * 60)
    print()

    # Get API key
    api_key = get_api_key("GOOGLE_API_KEY")
    if not api_key:
        print("❌ Error: GOOGLE_API_KEY not found in environment or secrets.toml")
        print()
        print("Please set your API key:")
        print("  1. Create a secrets.toml file (copy from secrets.toml.example)")
        print("  2. Add: GOOGLE_API_KEY = \"your-key-here\"")
        print("  3. Or set environment variable: export GOOGLE_API_KEY=your-key")
        return False

    # Documents path
    documents_path = Path(__file__).parent.parent / "data" / "papers"

    if not documents_path.exists():
        print(f"❌ Error: Documents folder not found: {documents_path}")
        return False

    pdf_files = list(documents_path.glob("*.pdf"))
    if not pdf_files:
        print(f"❌ Error: No PDF files found in {documents_path}")
        print()
        print("Please add PDF files to the data/papers/ directory first.")
        return False

    print(f"📁 Documents folder: {documents_path}")
    print(f"📄 Found {len(pdf_files)} PDF files:")
    for pdf in pdf_files:
        size_mb = pdf.stat().st_size / (1024 * 1024)
        print(f"   - {pdf.name} ({size_mb:.2f} MB)")
    print()

    # Create agent
    print("🔧 Initializing RAG agent...")
    agent = RAGChatbotAgent(
        api_key=api_key,
        documents_path=str(documents_path),
        model="gemini-2.5-flash"
    )
    print("✅ Agent initialized")
    print()

    # Rebuild vector store
    print("🧠 Rebuilding vector store (this may take a few minutes)...")
    print("   - Loading PDFs")
    print("   - Splitting into chunks")
    embedding_msg = "HuggingFace model" if agent.embedding_type == "huggingface" else "Google Gemini API"
    print(f"   - Generating embeddings (using {embedding_msg})")
    print("   - Storing in ChromaDB")
    print()

    result = agent.load_documents(force_reload=True)

    print()
    print("=" * 60)
    if result.get("success"):
        print("✅ SUCCESS!")
        print("=" * 60)
        print()
        print(f"📊 Statistics:")
        print(f"   - Papers processed: {result.get('pdf_count', 0)}")
        print(f"   - Document chunks: {result.get('chunk_count', 0):,}")
        print(f"   - ChromaDB location: {agent.persist_directory}")
        print()
        print("📝 Next steps:")
        print("   1. Test the vector store locally with the Streamlit app")
        print("   2. Commit the changes:")
        print(f"      git add {agent.persist_directory}")
        print("      git commit -m \"Update vector store with new embeddings\"")
        print("   3. Push to deploy:")
        print("      git push")
        print()
        return True
    else:
        print("❌ FAILED!")
        print("=" * 60)
        print()
        print(f"Error: {result.get('message', 'Unknown error')}")
        print()
        return False


if __name__ == "__main__":
    success = rebuild_vector_store()
    sys.exit(0 if success else 1)
