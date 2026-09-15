from typing import Any, Dict, Optional
from langchain_core.documents import Document
from config import settings
from rag.registry import registry


class RAGIngestionPipeline:
    """Phase 1: Offline Ingestion — quét file, chunk, embed, lưu vào pgvector."""

    def __init__(
        self,
        data_dir: str = "./data/Document",
        loader_name: str = "unstructured",
        chunker_name: str = "recursive",
        embedder_name: str = "ollama",
        vector_store_name: str = "pgvector",
    ):
        self.data_dir = data_dir

        loader_cls = registry.get_loader(loader_name)
        if not loader_cls:
            raise ValueError(f"Loader '{loader_name}' chưa được đăng ký!")
        self.loader = loader_cls()

        chunker_cls = registry.get_chunker(chunker_name)
        if not chunker_cls:
            raise ValueError(f"Chunker '{chunker_name}' chưa được đăng ký!")

        try:
            self.chunker = chunker_cls(
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP,
            )
        except TypeError:
            self.chunker = chunker_cls()

        embedder_cls = registry.get_embedder(embedder_name)
        if not embedder_cls:
            raise ValueError(f"Embedder '{embedder_name}' chưa được đăng ký!")
        self.embedder = embedder_cls()

        vector_store_cls = registry.get_vector_store(vector_store_name)
        if not vector_store_cls:
            raise ValueError(f"VectorStore '{vector_store_name}' chưa được đăng ký!")
        self.vector_store = vector_store_cls(embeddings=self.embedder)

    def ingest(self, overwrite: bool = False) -> Dict[str, Any]:
        if overwrite and hasattr(self.vector_store, "clear"):
            self.vector_store.clear()

        documents = self.loader.load_all()
        if not documents:
            return {
                "status": "no_documents",
                "message": f"Không tìm thấy tài liệu trong '{self.data_dir}'.",
            }

        chunks = self.chunker.split(documents)
        if not chunks:
            return {
                "status": "no_chunks",
                "message": "Không tạo được chunk từ tài liệu.",
            }

        self.vector_store.add_documents(chunks)

        return {
            "status": "success",
            "total_documents": len(documents),
            "total_chunks": len(chunks),
            "message": f"Đã nạp thành công {len(chunks)} chunks vào database.",
        }
