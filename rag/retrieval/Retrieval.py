from typing import List, Union, Dict, Any, Optional
from collections import defaultdict
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from rag.registry import registry

@registry.retriever("retrieval_vector")
class VectorSearchRetriever:
    def __init__(self, vector_store, top_k: int = 5):
        """
        :param vector_store: Instance của PGVectorStore hoặc ChromaStore
        :param top_k: Số lượng tài liệu tối đa cần lấy cho mỗi query
        """
        self.vector_store = vector_store
        self.top_k = top_k

    def retrieve(self, queries: Union[str, List[str]]) -> List[Document]:
        """
        Nhận 1 câu hỏi hoặc danh sách sub-queries từ Pre-retrieval,
        truy xuất từ Vector Store và khử trùng lặp (Deduplication).
        """
        if isinstance(queries, str):
            queries = [queries]

        all_documents: List[Document] = []
        seen_contents = set()

        for query in queries:
            docs = self.vector_store.similarity_search(query=query, top_k=self.top_k)
            
            for doc in docs:
                if doc.page_content not in seen_contents:
                    seen_contents.add(doc.page_content)
                    all_documents.append(doc)

        return all_documents

@registry.retriever("retrieval_bm25")
class BM25SearchRetriever:
    """Sparse Keyword Search an toàn với danh sách query linh hoạt."""

    def __init__(self, vector_store, top_k: int = 8, **kwargs):
        self.vector_store = vector_store
        self.top_k = top_k
        self._corpus_docs: List[Document] = []
        self._bm25 = None
        self._init_bm25()

    def _init_bm25(self):
        try:
            # Lấy corpus mẫu từ database
            if hasattr(self.vector_store, "similarity_search"):
                self._corpus_docs = self.vector_store.similarity_search(query="a", top_k=1000)
            elif hasattr(self.vector_store, "search"):
                self._corpus_docs = self.vector_store.search(query="a", top_k=1000)
        except Exception:
            self._corpus_docs = []

        if self._corpus_docs:
            tokenized_corpus = [
                doc.page_content.lower().split() for doc in self._corpus_docs if doc.page_content
            ]
            if tokenized_corpus:
                self._bm25 = BM25Okapi(tokenized_corpus)

    def search(
        self,
        queries: Union[str, List[str]],
        top_k: Optional[int] = None,
    ) -> List[Document]:
        if not self._bm25 or not self._corpus_docs:
            self._init_bm25()
        if not self._bm25 or not self._corpus_docs:
            return []

        effective_top_k = top_k if top_k is not None else self.top_k
        if isinstance(queries, str):
            queries = [queries]

        doc_scores: Dict[int, float] = defaultdict(float)
        for q in queries:
            if not q or not q.strip():
                continue
            tokenized_query = q.lower().split()
            scores = self._bm25.get_scores(tokenized_query)
            for idx, score in enumerate(scores):
                doc_scores[idx] += score

        ranked_indices = sorted(doc_scores.keys(), key=lambda i: doc_scores[i], reverse=True)
        return [self._corpus_docs[i] for i in ranked_indices[:effective_top_k]]


@registry.retriever("retrieval_hybrid")
class HybridSearchRetriever:

    def __init__(
        self,
        vector_store,
        bm25_retriever=None,
        top_k: int = 8,
        k: int = 60,
        **kwargs,
    ):
        self.vector_store = vector_store
        self.bm25_retriever = bm25_retriever
        self.top_k = top_k
        self.k = k  # Hằng số RRF (Reciprocal Rank Fusion)

    def search(
        self,
        queries: Union[str, List[str]],
        top_k: Optional[int] = None,
    ) -> List[Document]:
        if isinstance(queries, str):
            queries = [queries]

        effective_top_k = top_k if top_k is not None else self.top_k
        all_hybrid_docs: List[Document] = []
        seen_texts = set()

        for query in queries:
            if not query or not query.strip():
                continue

            # 1. Lấy kết quả từ Vector Search an toàn
            vector_docs: List[Document] = []
            try:
                if hasattr(self.vector_store, "similarity_search"):
                    vector_docs = self.vector_store.similarity_search(
                        query=query, top_k=effective_top_k
                    )
                elif hasattr(self.vector_store, "search"):
                    vector_docs = self.vector_store.search(
                        query=query, top_k=effective_top_k
                    )
            except Exception:
                vector_docs = []

            # 2. Lấy kết quả từ BM25 Search an toàn
            bm25_docs: List[Document] = []
            if self.bm25_retriever:
                try:
                    if hasattr(self.bm25_retriever, "search"):
                        bm25_docs = self.bm25_retriever.search(
                            query=query, top_k=effective_top_k
                        )
                    elif hasattr(self.bm25_retriever, "invoke"):
                        bm25_docs = self.bm25_retriever.invoke(query)
                except Exception:
                    bm25_docs = []

            # 3. Hợp nhất bằng RRF (Reciprocal Rank Fusion)
            doc_scores: Dict[str, float] = {}
            doc_map: Dict[str, Document] = {}

            for rank, doc in enumerate(vector_docs):
                key = doc.page_content.strip()
                doc_map[key] = doc
                doc_scores[key] = doc_scores.get(key, 0.0) + (
                    1.0 / (self.k + rank + 1)
                )

            for rank, doc in enumerate(bm25_docs):
                key = doc.page_content.strip()
                doc_map[key] = doc
                doc_scores[key] = doc_scores.get(key, 0.0) + (
                    1.0 / (self.k + rank + 1)
                )

            sorted_keys = sorted(
                doc_scores.keys(), key=lambda x: doc_scores[x], reverse=True
            )

            for k_text in sorted_keys[:effective_top_k]:
                if k_text not in seen_texts:
                    seen_texts.add(k_text)
                    all_hybrid_docs.append(doc_map[k_text])

        return all_hybrid_docs

    def retrieve(
        self,
        queries: Union[str, List[str]],
        top_k: Optional[int] = None,
    ) -> List[Document]:
        return self.search(queries=queries, top_k=top_k)