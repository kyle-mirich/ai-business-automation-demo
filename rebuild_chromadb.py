"""
Rebuild ChromaDB vector store from scratch with compatible schema
"""
import os
import sys
import shutil
from pathlib import Path
from dotenv import load_dotenv
from agents.rag_chatbot_agent import RAGChatbotAgent

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Load environment
load_dotenv()

def rebuild_chromadb():
    """Rebuild the ChromaDB vector store"""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: GOOGLE_API_KEY not found in environment")
        return False

    documents_path = Path("data/papers")
    chroma_path = documents_path / "rag_chroma"

    print("Removing old ChromaDB...")
    if chroma_path.exists():
        shutil.rmtree(chroma_path)
        print(f"   Deleted: {chroma_path}")

    print("\nInitializing RAG agent...")
    agent = RAGChatbotAgent(
        api_key=api_key,
        documents_path=str(documents_path),
        model="gemini-2.5-flash"
    )

    print("\nLoading documents and building vector store...")
    print("   This may take a few minutes...")
    result = agent.load_documents(force_reload=True)

    if result.get("success"):
        print(f"\nSuccess!")
        print(f"   Loaded {result.get('pdf_count', 0)} PDF files")
        print(f"   Created {result.get('chunk_count', 0)} document chunks")
        print(f"   Vector store saved to: {chroma_path}")
        print(f"\n   Papers indexed:")
        for paper in result.get("papers", []):
            print(f"      - {paper}")
        return True
    else:
        print(f"\nFailed to rebuild ChromaDB")
        print(f"   Error: {result.get('message', 'Unknown error')}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("ChromaDB Rebuild Script")
    print("=" * 60)
    print()

    success = rebuild_chromadb()

    print("\n" + "=" * 60)
    if success:
        print("✅ ChromaDB rebuild complete!")
        print("   You can now restart your API server.")
    else:
        print("❌ ChromaDB rebuild failed!")
        print("   Check the error messages above.")
    print("=" * 60)
