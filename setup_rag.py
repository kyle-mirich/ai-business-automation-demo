#!/usr/bin/env python
"""
Quick setup script to scrape arXiv papers and build the RAG vector store.

Run this once to download papers and create the ChromaDB database:
    python setup_rag.py

Or with custom options:
    python setup_rag.py --max-papers 10 --skip-download
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Run the arXiv scraper with sensible defaults."""
    script_path = Path(__file__).parent / "scripts" / "scrape_arxiv_papers.py"

    if not script_path.exists():
        print(f"Error: Script not found at {script_path}")
        sys.exit(1)

    print("\n" + "=" * 70)
    print(" ArXiv Paper Scraper for RAG Chatbot")
    print("=" * 70)
    print("\nThis script will:")
    print("  1. Download AI research papers from arXiv")
    print("  2. Extract text from PDFs")
    print("  3. Create a ChromaDB vector store for RAG")
    print("\nNote: First run may take 2-5 minutes depending on paper sizes")
    print("=" * 70 + "\n")

    try:
        # Run the scraper
        result = subprocess.run(
            [sys.executable, str(script_path)] + sys.argv[1:],
            check=False
        )

        if result.returncode == 0:
            print("\n" + "=" * 70)
            print(" Setup Complete! ✓")
            print("=" * 70)
            print("\nYou can now:")
            print("  1. Start the app: streamlit run Home.py")
            print("  2. Navigate to the '💬 RAG Chatbot' page")
            print("  3. The papers should already be loaded!")
            print("=" * 70 + "\n")
        else:
            print("\n" + "=" * 70)
            print(" Setup Failed ✗")
            print("=" * 70)
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\nSetup cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError running setup: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
