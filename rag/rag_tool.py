from typing import Optional, Dict, Any
from config import settings
from rag.pipeline.serving import RAGServingPipeline
from rag.registry import registry
from rag.memory import chat_memory

_rag_pipeline: Optional[RAGServingPipeline] = None


def get_rag_pipeline(
    top_k: int = settings.TOP_K,
    top_n: int = settings.DEFAULT_TOP_N,
) -> RAGServingPipeline:
    global _rag_pipeline
    if _rag_pipeline is None:
        vector_store_cls = registry.get_vector_store("pgvector")
        if not vector_store_cls:
            raise ValueError("VectorStore 'pgvector' chưa được đăng ký!")
        embedder_cls = registry.get_embedder("ollama")
        if not embedder_cls:
            raise ValueError("Embedder 'ollama' chưa được đăng ký!")
        embedder = embedder_cls()
        vector_store = vector_store_cls(embeddings=embedder)
        _rag_pipeline = RAGServingPipeline(
            vector_store=vector_store, top_k=top_k, top_n=top_n
        )
    return _rag_pipeline


def rag_search_tool(
    query: str,
    doc_id: Optional[str] = None,
    session_id: str = "default_session",
    llm_provider: str = settings.DEFAULT_LLM_PROVIDER,
    retrieval_strategy: str = settings.DEFAULT_RETRIEVAL_STRATEGY,
    post_strategy: str = settings.DEFAULT_POST_RETRIEVAL_STRATEGY,
    top_k: Optional[int] = None,
    top_n: Optional[int] = None,
) -> str:
    pipeline = get_rag_pipeline()
    result = pipeline.run(
        query=query,
        session_id=session_id,
        selected_document=doc_id,
        llm_provider=llm_provider,
        retrieval_strategy=retrieval_strategy,
        post_retrieval_strategy=post_strategy,
        top_k=top_k,
        top_n=top_n,
    )
    return result["answer"]


def rag_search_with_sources(
    query: str,
    doc_id: Optional[str] = None,
    session_id: str = "default_session",
    llm_provider: str = settings.DEFAULT_LLM_PROVIDER,
    retrieval_strategy: str = settings.DEFAULT_RETRIEVAL_STRATEGY,
    post_strategy: str = settings.DEFAULT_POST_RETRIEVAL_STRATEGY,
    top_k: Optional[int] = None,
    top_n: Optional[int] = None,
) -> Dict[str, Any]:
    pipeline = get_rag_pipeline()
    return pipeline.run(
        query=query,
        session_id=session_id,
        selected_document=doc_id,
        llm_provider=llm_provider,
        retrieval_strategy=retrieval_strategy,
        post_retrieval_strategy=post_strategy,
        top_k=top_k,
        top_n=top_n,
    )


def rag_stream_tool(
    query: str,
    session_id: str = "default_session",
    doc_id: Optional[str] = None,
    llm_provider: str = settings.DEFAULT_LLM_PROVIDER,
    retrieval_strategy: str = settings.DEFAULT_RETRIEVAL_STRATEGY,
    post_strategy: str = settings.DEFAULT_POST_RETRIEVAL_STRATEGY,
    top_k: int = settings.TOP_K,
    top_n: int = settings.DEFAULT_TOP_N,
):
    pipeline = get_rag_pipeline()
    return pipeline.stream_answer(
        query=query,
        session_id=session_id,
        selected_document=doc_id,
        llm_provider=llm_provider,
        retrieval_strategy=retrieval_strategy,
        post_retrieval_strategy=post_strategy,
        top_k=top_k,
        top_n=top_n,
    )


def clear_chat_history(session_id: str = "default_session") -> None:
    chat_memory.clear(session_id)
