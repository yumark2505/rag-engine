from typing import Optional, Dict, Any
from rag.pipeline.ingestion import RAGIngestionPipeline
from rag.pipeline.serving import RAGServingPipeline


class Pipeline:
    """Unified RAG Pipeline — orchestrates ingestion + serving."""

    def __init__(
        self,
        data_dir: str = "./data/Document",
        loader_name: str = "unstructured",
        chunker_name: str = "recursive",
        embedder_name: str = "ollama",
        vector_store_name: str = "pgvector",
        top_k: int = 8,
        top_n: int = 3,
    ):
        self.ingestion = RAGIngestionPipeline(
            data_dir=data_dir,
            loader_name=loader_name,
            chunker_name=chunker_name,
            embedder_name=embedder_name,
            vector_store_name=vector_store_name,
        )
        self.serving = RAGServingPipeline(
            vector_store=self.ingestion.vector_store,
            top_k=top_k,
            top_n=top_n,
        )

    def ingest(self, overwrite: bool = False) -> Dict[str, Any]:
        return self.ingestion.ingest(overwrite=overwrite)

    def run(
        self,
        query: str,
        session_id: str = "default_session",
        selected_document: Optional[str] = None,
        llm_provider: str = "ollama",
        top_k: Optional[int] = None,
        top_n: Optional[int] = None,
        pre_retrieval_strategy: str = "identity",
        retrieval_strategy: str = "vector",
        post_retrieval_strategy: str = "rerank",
        **kwargs,
    ) -> Dict[str, Any]:
        return self.serving.run(
            query=query,
            session_id=session_id,
            selected_document=selected_document,
            llm_provider=llm_provider,
            top_k=top_k,
            top_n=top_n,
            pre_retrieval_strategy=pre_retrieval_strategy,
            retrieval_strategy=retrieval_strategy,
            post_retrieval_strategy=post_retrieval_strategy,
            **kwargs,
        )

    def stream_answer(
        self,
        query: str,
        session_id: str = "default_session",
        selected_document: Optional[str] = None,
        llm_provider: str = "ollama",
        top_k: int = 8,
        top_n: int = 5,
        pre_retrieval_strategy: str = "identity",
        retrieval_strategy: str = "vector",
        post_retrieval_strategy: str = "rerank",
        **kwargs,
    ):
        return self.serving.stream_answer(
            query=query,
            session_id=session_id,
            selected_document=selected_document,
            llm_provider=llm_provider,
            top_k=top_k,
            top_n=top_n,
            pre_retrieval_strategy=pre_retrieval_strategy,
            retrieval_strategy=retrieval_strategy,
            post_retrieval_strategy=post_retrieval_strategy,
            **kwargs,
        )

    def stream(self, query: str, **kwargs):
        return self.serving.stream(query=query, **kwargs)
