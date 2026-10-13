"""
Wrapper for langchain vector store
"""
from typing import List, Dict
import logging

from langchain_core.documents import Document
from langchain_community.vectorstores import VectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

class VectorStoreWrapper:
    """
    A Wrapper for LangchainVectorStore to handle GPT-Researcher Document Type
    """
    def __init__(self, vector_store: VectorStore, chunking_strategy: str = "semantic"):
        self.vector_store = vector_store
        self.chunking_strategy = chunking_strategy

    def load(self, documents):
        """
        Load the documents into vector_store
        Translate to langchain doc type, split to chunks then load
        """
        langchain_documents = self._create_langchain_documents(documents)
        splitted_documents = self._split_documents(langchain_documents)
        self.vector_store.add_documents(splitted_documents)
    
    def _create_langchain_documents(self, data: List[Dict[str, str]]) -> List[Document]:
        """Convert GPT Researcher Document to Langchain Document.

        Skip non-dict rows and entries without usable ``raw_content``. Missing
        ``url`` still yields a document with empty source rather than KeyError.
        """
        docs: List[Document] = []
        if not data:
            return docs
        for item in data:
            if not isinstance(item, dict):
                continue
            content = item.get("raw_content")
            if content is None:
                continue
            docs.append(
                Document(
                    page_content=str(content),
                    metadata={"source": item.get("url") or ""},
                )
            )
        return docs

    def _split_documents(self, documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
        """
        Split documents into smaller chunks
        """
        if self.chunking_strategy == "semantic":
            embeddings = (
                getattr(self.vector_store, "embeddings", None)
                or getattr(self.vector_store, "_embeddings", None)
                or getattr(self.vector_store, "_embedding_function", None)
            )
            try:
                from langchain_experimental.text_splitter import SemanticChunker

                if embeddings is not None:
                    return SemanticChunker(
                        embeddings,
                        breakpoint_threshold_type="percentile",
                    ).split_documents(documents)
                logger.warning("Semantic chunking requested but vector store exposes no embeddings; using recursive chunks")
            except (ImportError, AttributeError, TypeError) as exc:
                logger.warning("Semantic chunking unavailable (%s); using recursive chunks", exc)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return text_splitter.split_documents(documents)

    async def asimilarity_search(self, query, k, filter):
        """Return query by vector store"""
        results = await self.vector_store.asimilarity_search(query=query, k=k, filter=filter)
        return results
