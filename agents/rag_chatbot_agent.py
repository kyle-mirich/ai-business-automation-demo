"""
Simple RAG Chatbot Agent with LangChain

Features:
- ChromaDB vector store for local PDF documents
- Streaming responses
- Conversation memory
- Source citations
"""

import html
import json
import re
import shutil
from typing import List, Dict, Any, Optional, Generator, Tuple
from pathlib import Path

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader


class RAGChatbotAgent:
    """
    Simple RAG chatbot with vector search and streaming
    """

    def __init__(
        self,
        api_key: str,
        documents_path: str,
        model: str = "gemini-2.0-flash-exp",
        collection_name: str = "ai_papers"
    ):
        """
        Initialize the RAG chatbot

        Args:
            api_key: Google Gemini API key
            documents_path: Path to documents folder (PDFs)
            model: Gemini model to use
            collection_name: ChromaDB collection name
        """
        self.api_key = api_key
        self.model_name = model
        self.collection_name = collection_name
        self.documents_path = Path(documents_path)

        # Initialize LLM with streaming
        self.llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=self.api_key,
            temperature=0.7,
            streaming=True,
            convert_system_message_to_human=True
        )

        # Initialize embeddings with HuggingFace (BAAI/bge-small-en-v1.5)
        # This model is lightweight and efficient, perfect for 8GB VRAM
        # No API calls or quotas, fully local
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

        # Vector store
        self.vectorstore = None
        self.retriever = None

        # Conversation memory (simple list instead of ConversationBufferMemory)
        self.chat_history = []

        # Tracking
        self.last_sources = []
        self.last_load_result: Optional[Dict[str, Any]] = None

    def load_documents(self, force_reload: bool = False) -> Dict[str, Any]:
        """
        Load PDFs and create/load vector store

        Args:
            force_reload: Force recreation of vector store

        Returns:
            Status dict with success, message, and stats
        """
        try:
            # Setup persist directory (inside papers folder)
            persist_directory = self.documents_path / "chroma_db"
            if force_reload and persist_directory.exists():
                shutil.rmtree(persist_directory)
            persist_directory.mkdir(parents=True, exist_ok=True)
            if force_reload:
                self.vectorstore = None
                self.retriever = None

            # Check if vector store already exists
            chroma_file = persist_directory / "chroma.sqlite3"
            if not force_reload and chroma_file.exists():
                # Load existing vector store
                self.vectorstore = Chroma(
                    collection_name=self.collection_name,
                    embedding_function=self.embeddings,
                    persist_directory=str(persist_directory)
                )

                try:
                    collection = self.vectorstore._collection
                    doc_count = collection.count()
                except Exception:
                    doc_count = 0

                self.retriever = self.vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 4}
                )

                result = {
                    "success": True,
                    "message": f"Loaded existing vector database with {doc_count} document chunks",
                    "chunk_count": doc_count,
                    "is_new": False
                }
                self.last_load_result = result
                return result

            # Check if PDFs exist
            if not self.documents_path.exists():
                return {
                    "success": False,
                    "message": f"Documents folder not found: {self.documents_path}"
                }

            pdf_files = list(self.documents_path.glob("*.pdf"))
            if not pdf_files:
                return {
                    "success": False,
                    "message": f"No PDF files found in {self.documents_path}. Please add AI research papers."
                }

            # Load all PDFs
            loader = DirectoryLoader(
                str(self.documents_path),
                glob="*.pdf",
                loader_cls=PyPDFLoader
            )
            documents = loader.load()

            if not documents:
                return {
                    "success": False,
                    "message": "Failed to load PDF documents"
                }

            # Normalize metadata
            resolved_base = self.documents_path.resolve()
            for doc in documents:
                source_path = Path(doc.metadata.get("source", resolved_base))
                doc.metadata["source"] = str(source_path.resolve())

            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1400,
                chunk_overlap=280,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            chunks = text_splitter.split_documents(documents)

            for chunk in chunks:
                meta = chunk.metadata
                source_path = Path(meta.get("source", resolved_base)).resolve()
                meta["source"] = str(source_path)
                if "page" in meta:
                    try:
                        page_idx = int(meta["page"])
                        meta["page"] = page_idx + 1
                    except Exception:
                        pass
                if "chunk_index" not in meta:
                    meta["chunk_index"] = meta.get("seq_num")

            # Create vector store
            self.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                collection_name=self.collection_name,
                persist_directory=str(persist_directory)
            )
            self.vectorstore.persist()

            try:
                collection = self.vectorstore._collection
                doc_count = collection.count()
            except Exception:
                doc_count = len(chunks)

            # Setup retriever
            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4}
            )

            result = {
                "success": True,
                "message": f"Successfully loaded {len(pdf_files)} papers and created {doc_count} chunks",
                "pdf_count": len(pdf_files),
                "chunk_count": doc_count,
                "is_new": True,
                "papers": [pdf.name for pdf in pdf_files]
            }
            self.last_load_result = result
            return result

        except Exception as e:
            result = {
                "success": False,
                "message": f"Error loading documents: {str(e)}"
            }
            self.last_load_result = result
            return result

    def chat_stream(self, message: str) -> Generator[Dict[str, Any], None, None]:
        """
        Stream response tokens with RAG

        Args:
            message: User message

        Yields:
            Dicts with type and content
        """
        if not self.retriever:
            yield {
                "type": "error",
                "content": "Vector database not loaded. Please load documents first."
            }
            return

        try:
            docs, scores = self._retrieve_with_scores(message)

            # Extract and yield sources enriched with highlights
            sources = [
                self._build_source_payload(doc, scores[idx], message, rank=idx + 1)
                for idx, doc in enumerate(docs)
            ]

            self.last_sources = sources

            yield {
                "type": "sources",
                "content": sources
            }

            # Build context from retrieved docs
            context = "\n\n".join([
                f"[Source: {Path(doc.metadata.get('source', 'Unknown')).name}, Page {doc.metadata.get('page', 'N/A')}]\n{doc.page_content}"
                for doc in docs
            ])

            # Format recent chat history (last 3 exchanges)
            recent_history = ""
            if self.chat_history:
                for msg in self.chat_history[-6:]:  # Last 3 Q&A pairs
                    role = "User" if msg["role"] == "user" else "Assistant"
                    recent_history += f"{role}: {msg['content']}\n"

            # Create prompt
            system_context = """You are an expert AI research assistant specializing in Large Language Models (LLMs).

You have access to cutting-edge research papers on LLMs, transformers, prompt engineering, fine-tuning, and related topics.

Guidelines:
- Always cite specific papers and page numbers when using information from the context
- Provide accurate, technical explanations of LLM concepts
- Compare different approaches and models mentioned in the papers
- Highlight key innovations and their implications
- If the papers don't contain enough information to answer a question, say so clearly
- Use specific examples, equations, and results from the papers when relevant"""

            prompt = f"""{system_context}

Context from research papers:
{context}

{f"Recent conversation:{chr(10)}{recent_history}" if recent_history else ""}

User question: {message}

Please provide a helpful answer based on the research papers. Always cite your sources."""

            # Stream the response
            full_response = ""
            for chunk in self.llm.stream(prompt):
                if hasattr(chunk, 'content'):
                    token = chunk.content
                    full_response += token
                    yield {
                        "type": "token",
                        "content": token
                    }

            # Save to chat history
            self.chat_history.append({"role": "user", "content": message})
            self.chat_history.append({"role": "assistant", "content": full_response})

            yield {
                "type": "complete",
                "content": full_response,
                "sources": sources
            }

        except Exception as e:
            yield {
                "type": "error",
                "content": f"Error: {str(e)}"
            }

    def clear_history(self) -> None:
        """Clear conversation history"""
        self.chat_history = []

    def get_stats(self) -> Dict[str, Any]:
        """Get chatbot statistics"""
        stats = {
            "vector_store_loaded": self.vectorstore is not None,
            "retriever_ready": self.retriever is not None,
            "total_chunks": 0,
            "conversation_length": len(self.chat_history)
        }

        if self.vectorstore:
            try:
                collection = self.vectorstore._collection
                stats["total_chunks"] = collection.count()
            except:
                pass

        return stats

    def _retrieve_with_scores(self, query: str, k: int = 4) -> Tuple[List, List[Optional[float]]]:
        """
        Retrieve documents alongside similarity scores when available.
        """
        docs: List = []
        scores: List[Optional[float]] = []

        if self.vectorstore and hasattr(self.vectorstore, "similarity_search_with_relevance_scores"):
            try:
                results = self.vectorstore.similarity_search_with_relevance_scores(query, k=k)
                for item in results:
                    if isinstance(item, tuple) and len(item) == 2:
                        doc, score = item
                    else:
                        doc, score = item, None
                    docs.append(doc)
                    if score is None:
                        scores.append(None)
                    else:
                        scores.append(float(score))
            except Exception:
                docs = []
                scores = []

        if not docs:
            raw_docs = self.retriever.invoke(query)
            docs = list(raw_docs)
            scores = [None] * len(docs)

        return docs, scores

    def _build_source_payload(
        self,
        doc,
        score: Optional[float],
        query: str,
        rank: int,
    ) -> Dict[str, Any]:
        """
        Construct enriched payload for a retrieved document chunk.
        """
        page_content = doc.page_content or ""
        metadata = doc.metadata or {}
        source_path = metadata.get("source") or "Unknown"
        page = metadata.get("page", "N/A")

        path_obj = Path(source_path)
        filename = path_obj.name or str(source_path)
        try:
            relative_href = path_obj.relative_to(self.documents_path)
        except (ValueError, RuntimeError):
            try:
                relative_href = path_obj.relative_to(self.documents_path.resolve())
            except (ValueError, RuntimeError):
                relative_href = path_obj

        source_url = None
        if isinstance(relative_href, Path):
            relative_str = relative_href.as_posix()
        else:
            relative_str = str(relative_href)

        if relative_str:
            page_fragment = ""
            try:
                page_int = int(page)
                page_fragment = f"#page={page_int}"
            except Exception:
                page_fragment = ""

            if not relative_str.startswith(("http://", "https://", "/")):
                relative_str = f"data/papers/{relative_str}"
            source_url = f"./{relative_str}{page_fragment}"

        return {
            "rank": rank,
            "source": filename,
            "page": page,
            "score": score,
            "chunk_excerpt": self._build_plain_excerpt(page_content),
            "highlighted_excerpt": self._build_highlight_excerpt(page_content, query),
            "chunk_full": page_content,
            "metadata": metadata,
            "source_url": source_url if source_url else None,
            "source_path": str(path_obj),
        }

    @staticmethod
    def _build_plain_excerpt(text: str, limit: int = 420) -> str:
        """Return a trimmed plain-text excerpt for quick preview."""
        text = text.strip()
        if len(text) <= limit:
            return text
        return text[:limit] + "..."

    def _build_highlight_excerpt(self, text: str, query: str, window: int = 760) -> str:
        """Generate HTML-safe excerpt with highlighted query terms."""
        clean_text = text.strip()
        if not clean_text:
            return ""

        keywords = self._extract_keywords(query)
        for keyword in keywords:
            pattern = re.compile(re.escape(keyword), re.IGNORECASE)
            match = pattern.search(clean_text)
            if match:
                start = max(0, match.start() - window // 2)
                end = min(len(clean_text), match.end() + window // 2)
                excerpt = clean_text[start:end]
                highlighted = self._render_highlight(excerpt, pattern)
                prefix = "..." if start > 0 else ""
                suffix = "..." if end < len(clean_text) else ""
                return prefix + highlighted + suffix

        excerpt = clean_text[:window]
        if len(clean_text) > window:
            excerpt += "..."
        return html.escape(excerpt)

    def _extract_keywords(self, query: str) -> List[str]:
        """Extract meaningful keywords from the user query."""
        raw_tokens = re.findall(r"\w+", query.lower())
        unique_tokens = sorted({token for token in raw_tokens if len(token) > 3}, key=len, reverse=True)
        if not unique_tokens:
            return [query.strip()] if query.strip() else []
        return unique_tokens

    @staticmethod
    def _render_highlight(text: str, pattern: re.Pattern) -> str:
        """Escape text and wrap matched terms in <mark> tags."""
        result_parts: List[str] = []
        last_idx = 0
        for match in pattern.finditer(text):
            result_parts.append(html.escape(text[last_idx:match.start()]))
            result_parts.append(f"<mark>{html.escape(match.group(0))}</mark>")
            last_idx = match.end()
        result_parts.append(html.escape(text[last_idx:]))
        return "".join(result_parts)
