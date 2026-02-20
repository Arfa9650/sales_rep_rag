"""
RAG: Build documents (my_company + prospect + optional search), chunk, create vector store and retriever.
"""

import logging
from typing import Any, List, Optional

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore, VectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config

logger = logging.getLogger("sales_rep.rag")


def _get_embeddings():
    """Return embeddings: Ollama if OLLAMA_EMBEDDING_MODEL set, else HuggingFace (sentence-transformers)."""
    if getattr(config, "OLLAMA_EMBEDDING_MODEL", "") and config.OLLAMA_EMBEDDING_MODEL.strip():
        try:
            from langchain_community.embeddings import OllamaEmbeddings
            return OllamaEmbeddings(
                base_url=config.OLLAMA_BASE_URL,
                model=config.OLLAMA_EMBEDDING_MODEL.strip(),
            )
        except Exception:
            pass
        try:
            from langchain_ollama import OllamaEmbeddings
            return OllamaEmbeddings(
                base_url=config.OLLAMA_BASE_URL,
                model=config.OLLAMA_EMBEDDING_MODEL.strip(),
            )
        except Exception:
            pass
    from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
    )


def _get_text_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        length_function=len,
        strip_whitespace=True,
    )


def _docs_my_company(my_company_description: str) -> List[Document]:
    """Source 1: Who you represent."""
    text = (my_company_description or "").strip()
    if not text:
        return []
    return [
        Document(
            page_content=text,
            metadata={"source": "my_company"},
        )
    ]


def _docs_prospect(
    prospect_profile_text: str,
    prospect_company_name: Optional[str] = None,
    prospect_industry: Optional[str] = None,
) -> List[Document]:
    """Source 2: Prospect profile."""
    parts = []
    if prospect_company_name or prospect_industry:
        parts.append(f"Company: {prospect_company_name or 'N/A'}, Industry: {prospect_industry or 'N/A'}")
    parts.append((prospect_profile_text or "").strip())
    text = "\n".join(p for p in parts if p).strip()
    if not text:
        return []
    metadata: dict = {"source": "prospect_profile"}
    if prospect_company_name:
        metadata["company_name"] = prospect_company_name
    if prospect_industry:
        metadata["industry"] = prospect_industry
    return [Document(page_content=text, metadata=metadata)]


def _docs_search(results: List[dict]) -> List[Document]:
    """Source 3: Search results. Each item: title, body/snippet, href/url, optional query."""
    docs = []
    for r in results:
        title = r.get("title") or ""
        body = r.get("body") or r.get("snippet") or ""
        url = r.get("href") or r.get("url") or ""
        content = f"{title}\n{body}".strip()
        if not content:
            continue
        metadata = {"source": "search", "url": url}
        if r.get("query"):
            metadata["query"] = r["query"]
        docs.append(Document(page_content=content, metadata=metadata))
    return docs


def search_results_as_docs(query: str, max_results: int = 5) -> List[Document]:
    """Run DuckDuckGo search and return results as LangChain Documents for RAG."""
    logger.info("RAG search: query=%s max_results=%s", query, max_results)
    try:
        from ddgs import DDGS
    except ImportError:
        logger.warning("RAG search: ddgs not available")
        return []
    try:
        ddgs = DDGS()
        raw = list(ddgs.text(query, max_results=max_results))
    except Exception as e:
        logger.warning("RAG search failed: %s", e)
        return []
    results = [
        {"title": r.get("title") or "", "body": r.get("body") or "", "href": r.get("href") or "", "query": query}
        for r in raw
    ]
    docs = _docs_search(results)
    logger.info("RAG search: got %d raw results -> %d docs for vector store", len(raw), len(docs))
    return docs


def build_documents(
    my_company_description: str,
    prospect_company_name: Optional[str],
    prospect_industry: Optional[str],
    prospect_profile_text: str,
    search_docs: Optional[List[Document]] = None,
) -> List[Document]:
    """Build all documents to index: my_company + prospect + optional search."""
    all_docs: List[Document] = []
    my_docs = _docs_my_company(my_company_description)
    prospect_docs = _docs_prospect(prospect_profile_text, prospect_company_name, prospect_industry)
    all_docs.extend(my_docs)
    all_docs.extend(prospect_docs)
    if search_docs:
        all_docs.extend(search_docs)
    logger.info(
        "RAG build_documents: my_company=%d prospect=%d search=%d -> total=%d",
        len(my_docs), len(prospect_docs), len(search_docs) if search_docs else 0, len(all_docs),
    )
    return all_docs


def chunk_documents(documents: List[Document]) -> List[Document]:
    """Split documents with RecursiveCharacterTextSplitter."""
    if not documents:
        return []
    splitter = _get_text_splitter()
    return splitter.split_documents(documents)


def create_vector_store(documents: List[Document]) -> VectorStore:
    """Create vector store from (optionally chunked) documents. Uses config for store type and path."""
    embeddings = _get_embeddings()
    if not documents:
        if config.VECTOR_STORE_TYPE == "chroma":
            try:
                from langchain_community.vectorstores import Chroma
                return Chroma(
                    collection_name="sales_rep",
                    embedding_function=embeddings,
                    persist_directory=config.VECTOR_STORE_PATH,
                )
            except Exception:
                pass
        return InMemoryVectorStore.from_texts([], embeddings)  # empty store
    texts = [d.page_content for d in documents]
    metadatas = [d.metadata for d in documents]
    if config.VECTOR_STORE_TYPE == "chroma":
        try:
            from langchain_community.vectorstores import Chroma
            return Chroma.from_texts(
                texts=texts,
                embedding=embeddings,
                metadatas=metadatas,
                collection_name="sales_rep",
                persist_directory=config.VECTOR_STORE_PATH,
            )
        except Exception:
            pass
    return InMemoryVectorStore.from_texts(texts, embeddings, metadatas=metadatas)


def get_retriever(vector_store: VectorStore, top_k: Optional[int] = None) -> Any:
    """Return a retriever over the vector store with top_k documents."""
    k = top_k if top_k is not None else config.RETRIEVER_TOP_K
    return vector_store.as_retriever(search_kwargs={"k": k})
