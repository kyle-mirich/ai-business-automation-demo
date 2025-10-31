"""
Script to scrape AI research papers from arXiv and build a ChromaDB vector store.

This script downloads papers on key AI topics and creates a vector database
for use with the RAG chatbot.
"""

import os
import sys
from pathlib import Path
from typing import Optional
import arxiv
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.cost_calculator import estimate_tokens

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Specific arXiv paper IDs for foundational papers
# Using direct IDs ensures we get the exact papers we want
FOUNDATIONAL_PAPERS = [
    "1706.03762",  # Attention is All You Need
    "1810.04805",  # BERT: Pre-training of Deep Bidirectional Transformers
    "1912.10683",  # RoBERTa: A Robustly Optimized BERT Pretraining Approach
    "1910.10683",  # XLNet: Generalized Autoregressive Pretraining
    "1506.02640",  # LSTM: A Search Space Odyssey
    "1409.1556",  # Very Deep Convolutional Networks (VGGNet)
]

# Output directory
DATA_DIR = Path(__file__).parent.parent / "data" / "papers"
METADATA_FILE = DATA_DIR / "papers_metadata.json"


def download_papers(
    paper_ids: Optional[list[str]] = None,
    output_dir: Optional[Path] = None
) -> list[dict]:
    """
    Download specific papers from arXiv by paper ID.

    Args:
        paper_ids: List of arXiv paper IDs (e.g., ["1706.03762", "1810.04805"])
        output_dir: Directory to save papers (default: data/papers/)

    Returns:
        List of downloaded paper metadata
    """
    if paper_ids is None:
        paper_ids = FOUNDATIONAL_PAPERS

    if output_dir is None:
        output_dir = DATA_DIR

    output_dir.mkdir(parents=True, exist_ok=True)

    downloaded_papers = []
    client = arxiv.Client()

    for paper_id in paper_ids:
        logger.info(f"Downloading paper: {paper_id}")

        try:
            # Fetch the paper by ID
            papers = client.results(
                arxiv.Search(
                    id_list=[paper_id],
                    max_results=1
                )
            )

            for paper in papers:
                # Generate filename from arXiv ID
                pdf_filename = f"{paper_id}.pdf"
                pdf_path = output_dir / pdf_filename

                if pdf_path.exists():
                    logger.info(f"Paper already exists: {paper.title}")
                    downloaded_papers.append({
                        "id": paper_id,
                        "title": paper.title,
                        "exists": True,
                        "pdf_path": str(pdf_path)
                    })
                    continue

                try:
                    logger.info(f"Downloading: {paper.title}")
                    paper.download_pdf(dirpath=str(output_dir), filename=pdf_filename)

                    paper_metadata = {
                        "id": paper_id,
                        "title": paper.title,
                        "authors": [author.name for author in paper.authors],
                        "published": str(paper.published),
                        "summary": paper.summary,
                        "pdf_path": str(pdf_path),
                        "arxiv_url": f"https://arxiv.org/abs/{paper_id}"
                    }

                    downloaded_papers.append(paper_metadata)
                    logger.info(f"✓ Downloaded: {paper.title}")

                except Exception as e:
                    logger.error(f"Failed to download {paper_id}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error fetching paper '{paper_id}': {e}")
            continue

    logger.info(f"Downloaded {len([p for p in downloaded_papers if 'exists' not in p])} new papers")
    return downloaded_papers


def build_vector_store(
    papers_dir: Optional[Path] = None,
    use_local: bool = True
) -> dict:
    """
    Build ChromaDB vector store from downloaded papers using HuggingFace embeddings.

    Best HuggingFace embedding models for 8GB VRAM:
    - BAAI/bge-small-en-v1.5 (384 dims, ~33MB) - Fast, efficient, competitive quality
    - sentence-transformers/all-MiniLM-L6-v2 (384 dims, ~22MB) - Very fast, lightweight
    - sentence-transformers/all-mpnet-base-v2 (768 dims, ~438MB) - Higher quality

    Args:
        papers_dir: Directory containing PDF files
        use_local: Whether to use local ChromaDB (vs in-memory)

    Returns:
        Dictionary with vector store info and stats
    """
    if papers_dir is None:
        papers_dir = DATA_DIR

    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_chroma import Chroma
    from langchain_community.embeddings import HuggingFaceEmbeddings

    logger.info("Building ChromaDB vector store...")

    # Find all PDFs
    pdf_files = list(papers_dir.glob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files")

    if not pdf_files:
        logger.warning("No PDF files found in papers directory")
        return {"success": False, "message": "No PDF files found"}

    # Load and split documents
    documents = []
    logger.info("Loading and splitting documents...")

    for pdf_path in pdf_files:
        try:
            loader = PyPDFLoader(str(pdf_path))
            docs = loader.load()

            # Add metadata for source tracking
            for i, doc in enumerate(docs):
                doc.metadata["source"] = pdf_path.name
                doc.metadata["source_path"] = str(pdf_path)
                doc.metadata["page"] = doc.metadata.get("page", i)

            documents.extend(docs)
            logger.info(f"Loaded {len(docs)} pages from {pdf_path.name}")

        except Exception as e:
            logger.error(f"Failed to load {pdf_path.name}: {e}")
            continue

    if not documents:
        logger.error("No documents loaded from PDFs")
        return {"success": False, "message": "No documents loaded"}

    # Split documents with metadata preservation
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""],
        add_start_index=True  # Add character index to metadata
    )
    splits = splitter.split_documents(documents)
    logger.info(f"Created {len(splits)} document chunks with metadata")

    # Add chunk IDs and additional metadata
    for i, split in enumerate(splits):
        split.metadata["chunk_id"] = i
        split.metadata["total_chunks"] = len(splits)

    # Initialize HuggingFace embeddings (best for 8GB VRAM)
    logger.info("Initializing HuggingFace embeddings (BAAI/bge-small-en-v1.5)...")
    logger.info("This model runs efficiently on 8GB VRAM and provides competitive quality")

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cpu"},  # Use CPU for stability, change to "cuda" if you have GPU
        encode_kwargs={"normalize_embeddings": True}
    )

    # Create vector store
    db_path = papers_dir / "rag_chroma"
    logger.info(f"Creating ChromaDB at {db_path}...")
    logger.info(f"Embedding {len(splits)} chunks (this may take a few minutes on CPU)...")

    try:
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=str(db_path),
            collection_metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"✓ Vector store created with {len(splits)} chunks")

    except Exception as e:
        logger.error(f"Failed to create vector store: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "message": f"Failed to create vector store: {str(e)}"
        }

    return {
        "success": True,
        "message": "Vector store created successfully",
        "chunks": len(splits),
        "documents": len(documents),
        "db_path": str(db_path),
        "pdf_files": len(pdf_files),
        "embedding_model": "BAAI/bge-small-en-v1.5"
    }


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Download foundational arXiv papers and build RAG vector store"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip downloading papers, only build vector store"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for papers (default: data/papers/)"
    )

    args = parser.parse_args()

    try:
        # Download papers
        if not args.skip_download:
            logger.info("=" * 60)
            logger.info("STEP 1: Downloading foundational papers from arXiv")
            logger.info("=" * 60)
            papers = download_papers(output_dir=args.output_dir)
            logger.info(f"\nDownloaded {len(papers)} papers")

        # Build vector store
        logger.info("\n" + "=" * 60)
        logger.info("STEP 2: Building ChromaDB vector store")
        logger.info("=" * 60)
        result = build_vector_store(papers_dir=args.output_dir)

        if result["success"]:
            logger.info("\n✓ SUCCESS!")
            logger.info(f"  - Vector store created at: {result['db_path']}")
            logger.info(f"  - Total chunks: {result['chunks']}")
            logger.info(f"  - Total documents: {result['documents']}")
            logger.info(f"  - PDF files: {result['pdf_files']}")
            logger.info("\nYou can now use the RAG chatbot with these papers!")
        else:
            logger.error(f"\n✗ FAILED: {result['message']}")
            sys.exit(1)

    except Exception as e:
        logger.error(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
