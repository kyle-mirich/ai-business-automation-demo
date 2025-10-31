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
from urllib.parse import quote

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage


class RAGChatbotAgent:
    """
    Simple RAG chatbot with vector search and streaming
    """

    def __init__(
        self,
        api_key: str,
        documents_path: str,
        model: str = "gemini-2.5-flash",
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
        # This model is lightweight and efficient
        # No API calls or quotas, fully local
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        self.embedding_type = "huggingface"

        # Vector store
        self.vectorstore = None
        self.retriever = None

        # Conversation memory (simple list instead of ConversationBufferMemory)
        self.chat_history = []

        # Tracking
        self.last_sources = []
        self.last_load_result: Optional[Dict[str, Any]] = None

    @property
    def persist_directory(self) -> Path:
        """Directory where the Chroma vector store is persisted."""
        return self.documents_path / "rag_chroma"

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
            persist_directory = self.documents_path / "rag_chroma"
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
                    search_kwargs={"k": 10}
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
                search_kwargs={"k": 10}
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

    def _expand_query(self, query: str) -> str:
        """
        Expand user query to be more detailed and specific for better retrieval.

        Args:
            query: Original user query

        Returns:
            Expanded query with more context and details
        """
        expansion_prompt = f"""You are a query expansion expert for AI/ML research papers.
Your job is to take a user's question and expand it into a more detailed, specific search query
that will retrieve better results from a vector database of AI research papers.

Original question: {query}

Expand this question by:
1. Adding relevant technical terms and synonyms
2. Including related concepts that might appear in relevant papers
3. Making it more specific and searchable
4. Keeping it concise (2-3 sentences max)

Return ONLY the expanded query, nothing else."""

        try:
            response = self.llm.invoke(expansion_prompt)
            expanded = response.content.strip()
            return expanded if expanded else query
        except Exception:
            # If expansion fails, return original query
            return query

    def _search_papers(self, query: str) -> Tuple[List, List[Optional[float]], str]:
        """
        Tool function to search research papers.

        Args:
            query: Search query

        Returns:
            Tuple of (documents, scores, search query used)
        """
        # Expand query for better retrieval
        expanded_query = self._expand_query(query)

        # Retrieve documents
        docs, scores = self._retrieve_with_scores(expanded_query)

        return docs, scores, expanded_query

    def chat_stream(self, message: str) -> Generator[Dict[str, Any], None, None]:
        """
        Stream response tokens with RAG using tool-based retrieval

        Args:
            message: User message

        Yields:
            Dicts with type and content
        """
        if self.retriever is None:
            yield {
                "type": "error",
                "content": "Vector database not loaded. Please load documents first."
            }
            return

        try:
            # Create search tool
            @tool
            def search_research_papers(query: str) -> str:
                """Search AI research papers for relevant information.

                Use this tool when you need to find information from research papers about:
                - Transformer architectures and attention mechanisms
                - BERT, GPT, and other language models
                - Machine learning and deep learning techniques
                - AI research and innovations

                Args:
                    query: The search query describing what information you need

                Returns:
                    Context from relevant research papers with citations
                """
                docs, scores, expanded_q = self._search_papers(query)

                # Build context from retrieved docs
                context = "\n\n".join([
                    f"[Source: {Path(doc.metadata.get('source', 'Unknown')).name}, Page {doc.metadata.get('page', 'N/A')}]\n{doc.page_content}"
                    for doc in docs
                ])

                # Store for later use
                self._temp_docs = docs
                self._temp_scores = scores
                self._temp_query = expanded_q

                return context

            # Bind tool to LLM
            llm_with_tools = self.llm.bind_tools([search_research_papers])

            # Format recent chat history
            recent_history = ""
            if self.chat_history:
                for msg in self.chat_history[-6:]:  # Last 3 Q&A pairs
                    role = "User" if msg["role"] == "user" else "Assistant"
                    recent_history += f"{role}: {msg['content']}\n"

            # System prompt
            system_msg = """You are an expert AI research assistant specializing in Large Language Models (LLMs).

You have access to a tool that searches cutting-edge research papers on LLMs, transformers, prompt engineering, fine-tuning, and related topics.

Guidelines:
- Use the search_research_papers tool when you need information from research papers
- You can decide whether to use the tool based on the question
- Always cite specific papers and page numbers when using information from the papers
- Provide accurate, technical explanations of LLM concepts
- If asked a general question that doesn't need paper lookup, answer directly
- If the papers don't contain enough information, say so clearly"""

            # Build messages
            messages = [HumanMessage(content=f"{system_msg}\n\n{recent_history}\n\nUser question: {message}")]

            # Invoke with tools
            self._temp_docs = []
            self._temp_scores = []
            self._temp_query = message

            response = llm_with_tools.invoke(messages)

            # Check if tool was called
            if response.tool_calls:
                # Execute tool calls
                for tool_call in response.tool_calls:
                    tool_output = search_research_papers.invoke(tool_call["args"])
                    messages.append(response)
                    messages.append(ToolMessage(content=tool_output, tool_call_id=tool_call["id"]))

                # Get final response with context
                full_response = ""
                for chunk in llm_with_tools.stream(messages):
                    if hasattr(chunk, 'content') and chunk.content:
                        token = chunk.content
                        full_response += token
                        yield {
                            "type": "token",
                            "content": token
                        }

                # Build sources from temp storage
                sources = [
                    self._build_source_payload(doc, self._temp_scores[idx], self._temp_query, rank=idx + 1)
                    for idx, doc in enumerate(self._temp_docs)
                ]

                yield {
                    "type": "sources",
                    "content": sources
                }
            else:
                # No tool call - stream direct response
                full_response = response.content
                for i, char in enumerate(full_response):
                    yield {
                        "type": "token",
                        "content": char
                    }

                # No sources for direct responses
                sources = []

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

    def _retrieve_with_scores(self, query: str, k: int = 10) -> Tuple[List, List[Optional[float]]]:
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
            page_int: Optional[int] = None
            try:
                page_int = max(1, int(page))
            except Exception:
                page_int = None

            relative_str_text = str(relative_str)
            if relative_str_text.startswith(("http://", "https://")):
                source_url = (
                    f"{relative_str_text}#page={page_int}"
                    if page_int
                    else relative_str_text
                )
            else:
                # Link directly to GitHub-hosted PDFs
                github_base = "https://github.com/kyle-mirich/ai-business-automation-demo/blob/main/data/papers"
                paper_param = quote(filename, safe="")
                source_url = f"{github_base}/{paper_param}"
                # Note: GitHub's PDF viewer doesn't support #page= fragment, but opens the PDF

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
