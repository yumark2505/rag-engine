from typing import Any, Dict, Iterator, List, Optional
from langchain_core.documents import Document
from config import settings
from rag.memory import chat_memory
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
        """Quét data_dir -> load -> chunk -> lưu vào pgvector."""
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


class RAGServingPipeline:
    """Phase 2: Online Serving — retrieve + rerank + memory + multi-llm generation."""

    def __init__(
        self,
        vector_store,
        top_k: int = settings.TOP_K,
        top_n: int = 3,
    ):
        self.vector_store = vector_store
        self.top_k = top_k
        self.top_n = top_n

    def _get_generator(self, provider: str = "ollama"):
        """Lấy class Generator trực tiếp từ registry, fallback về ollama nếu không tìm thấy."""
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

        # Xây dựng filter theo metadata filename
        filter_dict = None
        if selected_document and selected_document != "Tất cả":
            filter_dict = {"filename": selected_document}

        # ── 1. PRE-RETRIEVAL ──────────────────────────
        pre_strategy_key = "query_optimizer" if pre_strategy == "query_transform" else pre_strategy
        pre_cls = registry.get_retriever(f"pre_retrieval_{pre_strategy_key}")
        if pre_cls:
            optimizer = pre_cls()
            sub_queries = optimizer.optimize(query) if hasattr(optimizer, "optimize") else [query]
        else:
            sub_queries = [query]

        # ── 2. RETRIEVAL ──────────────────────────────
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
                    # Kiểm tra xem search() của retriever có nhận filter_dict hay không
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

        # ── 3. POST-RETRIEVAL ─────────────────────────
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
        llm_provider: str = "ollama",
        top_k: Optional[int] = None,
        top_n: Optional[int] = None,
        pre_retrieval_strategy: str = "identity",
        retrieval_strategy: str = "vector",
        post_retrieval_strategy: str = "rerank",
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
        llm_provider: str = "ollama",
        top_k: int = 8,
        top_n: int = 5,
        pre_retrieval_strategy: str = "identity",
        retrieval_strategy: str = "vector",
        post_retrieval_strategy: str = "rerank",
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