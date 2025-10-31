"""
Simple RAG Chatbot Agent with LangChain

Features:
- ChromaDB vector store for local PDF documents
- Streaming responses
- Conversation memory
- Source citations
"""

from typing import List, Dict, Any, Optional, Generator
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

    def load_documents(self, force_reload: bool = False) -> Dict[str, Any]:
        """
        Load PDFs and create/load vector store

        Args:
            force_reload: Force recreation of vector store

        Returns:
            Status dict with success, message, and stats
        """
        try:
            # Setup persist directory
            persist_directory = self.documents_path.parent / "chroma_db"
            persist_directory.mkdir(parents=True, exist_ok=True)

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
                except:
                    doc_count = 0

                self.retriever = self.vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 4}
                )

                return {
                    "success": True,
                    "message": f"Loaded existing vector database with {doc_count} document chunks",
                    "chunk_count": doc_count,
                    "is_new": False
                }

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

            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            chunks = text_splitter.split_documents(documents)

            # Create vector store
            self.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                collection_name=self.collection_name,
                persist_directory=str(persist_directory)
            )

            # Setup retriever
            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4}
            )

            return {
                "success": True,
                "message": f"Successfully loaded {len(pdf_files)} papers and created {len(chunks)} chunks",
                "pdf_count": len(pdf_files),
                "chunk_count": len(chunks),
                "is_new": True,
                "papers": [pdf.name for pdf in pdf_files]
            }

        except Exception as e:
            return {
                "success": False,
                "message": f"Error loading documents: {str(e)}"
            }

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
            # Retrieve relevant documents
            docs = self.retriever.invoke(message)

            # Extract and yield sources
            sources = []
            for doc in docs:
                source_info = {
                    "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                    "source": Path(doc.metadata.get("source", "Unknown")).name,
                    "page": doc.metadata.get("page", "N/A")
                }
                sources.append(source_info)

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
