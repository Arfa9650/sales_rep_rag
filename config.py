"""
Configuration for the LangChain RAG sales-rep agent.
Ollama, steps, confidence, embedding model, vector store, chunk/retrieval params.
"""

import os

MAX_STEPS = 15
MEMORY_RECENT_K = 10
# Only allow the agent to stop when confidence >= this (so it iterates with tools until satisfied)
MIN_CONFIDENCE_TO_STOP = 0.8
DECIDE_PARSE_RETRIES = 1
MODEL_ERROR_RETRIES = 2

# Ollama configuration (local or remote GPU)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "deepseek-r1:8b")
MODEL_NAME = OLLAMA_MODEL

# OpenAI — used by the consult_oracle tool (higher-intelligence synthesis)
# Paste your key in .env or set the environment variable before running.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# LangGraph
ENABLE_CHECKPOINTING = bool(os.getenv("ENABLE_CHECKPOINTING", ""))  # set to "1" to enable MemorySaver
GRAPH_RECURSION_LIMIT = int(os.getenv("GRAPH_RECURSION_LIMIT", str(MAX_STEPS * 8 + 20)))

# RAG: embeddings (local, no API key). sentence-transformers model name.
# Set OLLAMA_EMBEDDING_MODEL to use Ollama embeddings instead (e.g. nomic-embed-text).
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "")  # If set, use Ollama for embeddings

# Vector store: "memory" = InMemoryVectorStore (no path); "chroma" = Chroma (persisted to path)
VECTOR_STORE_TYPE = os.getenv("VECTOR_STORE_TYPE", "memory")
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "./chroma_sales_rep")

# Chunking and retrieval
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
RETRIEVER_TOP_K = int(os.getenv("RETRIEVER_TOP_K", "8"))

# API (FastAPI)
API_CORS_ORIGINS = os.getenv("API_CORS_ORIGINS", "")  # e.g. "http://localhost:3000,http://127.0.0.1:3000"
