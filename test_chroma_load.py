"""
Test script to verify ChromaDB loading
"""
from pathlib import Path
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Setup paths
base_path = Path(__file__).parent
documents_path = base_path / "data" / "papers"
persist_directory = documents_path / "rag_chroma"

print(f"Documents path: {documents_path}")
print(f"Persist directory: {persist_directory}")
print(f"Persist directory exists: {persist_directory.exists()}")

chroma_file = persist_directory / "chroma.sqlite3"
print(f"ChromaDB file: {chroma_file}")
print(f"ChromaDB file exists: {chroma_file.exists()}")

if chroma_file.exists():
    print(f"ChromaDB file size: {chroma_file.stat().st_size} bytes")

# Initialize embeddings (same as in agent)
print("\nInitializing embeddings...")
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)
print("[OK] Embeddings initialized")

# Try to load the vector store
print("\nLoading vector store...")
try:
    vectorstore = Chroma(
        collection_name="ai_papers",
        embedding_function=embeddings,
        persist_directory=str(persist_directory)
    )
    print("[OK] Vector store loaded")

    # Try to get collection
    print("\nGetting collection...")
    collection = vectorstore._collection
    print(f"Collection: {collection}")

    # Get count
    print("\nGetting count...")
    doc_count = collection.count()
    print(f"[OK] Document count: {doc_count}")

    # Try a test query
    if doc_count > 0:
        print("\nTrying a test query...")
        results = vectorstore.similarity_search("attention mechanism", k=3)
        print(f"[OK] Query returned {len(results)} results")
        if results:
            print(f"First result preview: {results[0].page_content[:100]}...")
    else:
        print("\n[WARNING] Collection is empty! This is the issue.")

        # Check if we can see the collection metadata
        print("\nInspecting collection metadata...")
        try:
            metadata = collection.get()
            print(f"Collection metadata: {metadata}")
        except Exception as e:
            print(f"Error getting metadata: {e}")

except Exception as e:
    print(f"[ERROR] Error loading vector store: {e}")
    import traceback
    traceback.print_exc()
