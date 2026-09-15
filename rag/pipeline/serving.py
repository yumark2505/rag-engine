from typing import Any, Dict, Iterator, List, Optional
from langchain_core.documents import Document
from config import settings
from rag.memory import chat_memory
from rag.registry import registry


class RAGServingPipeline:
    """Phase 2: Online Serving — retrieve + rerank + memory + multi-llm generation."""

    def __init__(
        self,
        vector_store,
        top_k: int = settings.TOP_K,
        top_n: int = settings.DEFAULT_TOP_N,
    ):
        self.vector_store = vector_store
        self.top_k = top_k
        self.top_n = top_n

    def _get_generator(self, provider: str = settings.DEFAULT_LLM_PROVIDER):
        generator_cls = registry.get_generator(provider)
        if not generator_cls:
            generator_cls = registry.get_generator("ollama")
        if not generator_cls:
            raise ValueError(f"Provider '{provider}' chưa được đăng ký trong registry!")
        return generator_cls()

    def _retrieve_context(
        self,
        query: str,
        top_k: Optional[int] = None,
        top_n: Optional[int] = None,
        pre_strategy: str = "identity",
        retrieval_strategy: str = "vector",
        post_strategy: str = "rerank",
        selected_document: Optional[str] = None,
    ) -> tuple[List[str], List[Document]]:
        effective_top_k = top_k if top_k is not None else self.top_k
        effective_top_n = top_n if top_n is not None else self.top_n

        filter_dict = None
        if selected_document and selected_document != "Tất cả":
            filter_dict = {"filename": selected_document}

        pre_strategy_key = "query_optimizer" if pre_strategy == "query_transform" else pre_strategy
        pre_cls = registry.get_retriever(f"pre_retrieval_{pre_strategy_key}")
        if pre_cls:
            optimizer = pre_cls()
            sub_queries = optimizer.optimize(query) if hasattr(optimizer, "optimize") else [query]
        else:
            sub_queries = [query]

        raw_docs: List[Document] = []
        seen_texts = set()

        retriever_cls = registry.get_retriever(f"retrieval_{retrieval_strategy}")
        if retriever_cls:
            try:
                retriever = retriever_cls(self.vector_store, top_k=effective_top_k)
            except TypeError:
                retriever = retriever_cls(self.vector_store)

            for q in sub_queries:
                if hasattr(retriever, "search"):
                    try:
                        docs = retriever.search(q, top_k=effective_top_k, filter_dict=filter_dict)
                    except TypeError:
                        docs = retriever.search(q, top_k=effective_top_k)
                elif hasattr(retriever, "retrieve"):
                    docs = retriever.retrieve(q, top_k=effective_top_k)
                else:
                    docs = []

                for doc in docs:
                    if doc.page_content not in seen_texts:
                        seen_texts.add(doc.page_content)
                        raw_docs.append(doc)
        else:
            for q in sub_queries:
                if hasattr(self.vector_store, "search"):
                    docs = self.vector_store.search(query=q, top_k=effective_top_k, filter_dict=filter_dict)
                elif hasattr(self.vector_store, "similarity_search"):
                    docs = self.vector_store.similarity_search(query=q, top_k=effective_top_k, filter_dict=filter_dict)
                else:
                    docs = []

                for doc in docs:
                    if doc.page_content not in seen_texts:
                        seen_texts.add(doc.page_content)
                        raw_docs.append(doc)

        post_cls = registry.get_retriever(f"post_retrieval_{post_strategy}")
        if post_cls:
            processor = post_cls(top_n=effective_top_n)
            if hasattr(processor, "process"):
                final_docs = processor.process(query, raw_docs, top_n=effective_top_n)
            elif hasattr(processor, "rerank"):
                final_docs = processor.rerank(query, raw_docs, top_n=effective_top_n)
            else:
                final_docs = raw_docs[:effective_top_n]
        else:
            final_docs = raw_docs[:effective_top_n]

        return sub_queries, final_docs

    def run(
        self,
        query: str,
        session_id: str = "default_session",
        selected_document: Optional[str] = None,
        llm_provider: str = settings.DEFAULT_LLM_PROVIDER,
        top_k: Optional[int] = None,
        top_n: Optional[int] = None,
        pre_retrieval_strategy: str = "identity",
        retrieval_strategy: str = settings.DEFAULT_RETRIEVAL_STRATEGY,
        post_retrieval_strategy: str = settings.DEFAULT_POST_RETRIEVAL_STRATEGY,
        **kwargs,
    ) -> Dict[str, Any]:
        history = chat_memory.get_messages(session_id)

        pre_strategy_key = "query_optimizer" if pre_retrieval_strategy == "query_transform" else pre_retrieval_strategy
        pre_cls = registry.get_retriever(f"pre_retrieval_{pre_strategy_key}")
        optimizer = pre_cls() if pre_cls else None
        standalone_query = query
        if optimizer and hasattr(optimizer, "contextualize_query") and pre_retrieval_strategy != "identity":
            standalone_query = optimizer.contextualize_query(query, history)

        sub_queries, final_docs = self._retrieve_context(
            query=standalone_query,
            top_k=top_k,
            top_n=top_n,
            pre_strategy=pre_retrieval_strategy,
            retrieval_strategy=retrieval_strategy,
            post_strategy=post_retrieval_strategy,
            selected_document=selected_document,
        )

        history_text = "\n".join([f"{msg.role}: {msg.content}" for msg in history])
        chat_memory.add_user_message(session_id, query)

        generator = self._get_generator(llm_provider)
        answer = generator.run(query=query, context_docs=final_docs, chat_history=history_text)
        chat_memory.add_ai_message(session_id, answer)

        clean_sources = []
        for doc in final_docs:
            clean_meta = {}
            for k, v in doc.metadata.items():
                clean_meta[k] = v.item() if hasattr(v, "item") else v
            clean_sources.append({"content": doc.page_content, "metadata": clean_meta})

        return {
            "query": query,
            "sub_queries": sub_queries,
            "answer": answer,
            "sources": clean_sources,
        }

    def stream_answer(
        self,
        query: str,
        session_id: str = "default_session",
        selected_document: Optional[str] = None,
        llm_provider: str = settings.DEFAULT_LLM_PROVIDER,
        top_k: int = settings.TOP_K,
        top_n: int = settings.DEFAULT_TOP_N,
        pre_retrieval_strategy: str = "identity",
        retrieval_strategy: str = settings.DEFAULT_RETRIEVAL_STRATEGY,
        post_retrieval_strategy: str = settings.DEFAULT_POST_RETRIEVAL_STRATEGY,
        **kwargs,
    ) -> Iterator[str]:
        history = chat_memory.get_messages(session_id)

        pre_strategy_key = "query_optimizer" if pre_retrieval_strategy == "query_transform" else pre_retrieval_strategy
        pre_cls = registry.get_retriever(f"pre_retrieval_{pre_strategy_key}")
        optimizer = pre_cls() if pre_cls else None
        standalone_query = query
        if optimizer and hasattr(optimizer, "contextualize_query") and pre_retrieval_strategy != "identity":
            standalone_query = optimizer.contextualize_query(query, history)

        _, final_docs = self._retrieve_context(
            query=standalone_query,
            top_k=top_k,
            top_n=top_n,
            pre_strategy=pre_retrieval_strategy,
            retrieval_strategy=retrieval_strategy,
            post_strategy=post_retrieval_strategy,
            selected_document=selected_document,
        )

        if not final_docs:
            yield "Tôi không tìm thấy thông tin này trong tài liệu được cung cấp."
            return

        history_text = "\n".join([f"{msg.role}: {msg.content}" for msg in history])
        chat_memory.add_user_message(session_id, query)

        generator = self._get_generator(llm_provider)
        collected_tokens: List[str] = []

        for token in generator.stream(
            query=query,
            context_docs=final_docs,
            chat_history=history_text,
        ):
            text = getattr(token, "content", str(token))
            collected_tokens.append(text)
            yield text

        chat_memory.add_ai_message(session_id, "".join(collected_tokens))

    def stream(
        self,
        query: str,
        top_k: Optional[int] = None,
        top_n: Optional[int] = None,
        selected_document: Optional[str] = None,
        **kwargs,
    ) -> Iterator[str]:
        return self.stream_answer(
            query=query,
            top_k=top_k or self.top_k,
            top_n=top_n or self.top_n,
            selected_document=selected_document,
            **kwargs,
        )
