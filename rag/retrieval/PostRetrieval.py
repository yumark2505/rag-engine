from typing import List, Optional, Union
from langchain_core.documents import Document
from flashrank import Ranker, RerankRequest
from rag.registry import registry
import re

@registry.retriever("post_retrieval_reranker")
class FlashReranker:

    def __init__(
        self, model_name: str = "ms-marco-TinyBERT-L-2-v2", top_n: int = 3
    ):
        """Khởi tạo Reranker siêu nhẹ chạy bằng CPU."""
        self.ranker = Ranker(model_name=model_name, cache_dir="./opt")
        self.top_n = top_n

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_n: Optional[int] = None, 
    ) -> List[Document]:
        if not documents:
            return []

        # Ưu tiên lấy top_n truyền vào từ request, fallback về self.top_n
        effective_top_n = top_n if top_n is not None else self.top_n

        passages = [
            {"id": idx, "text": doc.page_content, "meta": doc.metadata}
            for idx, doc in enumerate(documents)
        ]

        rerank_request = RerankRequest(query=query, passages=passages)
        results = self.ranker.rerank(rerank_request)

        reranked_docs = []
        for res in results[:effective_top_n]:
            reranked_docs.append(
                Document(
                    page_content=res["text"],
                    metadata={**res["meta"], "rerank_score": res["score"]},
                )
            )
        return reranked_docs

@registry.retriever("post_retrieval_contextual_compression")
class ContextualCompressor:
    """Nén ngữ cảnh: Lọc bỏ các câu không chứa từ khóa của câu hỏi."""

    def __init__(self, top_n: int = 3, **kwargs):
        self.top_n = top_n

    def process(
        self,
        query: Union[str, List[str]],
        documents: List[Document],
        top_n: Optional[int] = None,
    ) -> List[Document]:
        if not documents:
            return []
        effective_top_n = top_n if top_n is not None else self.top_n

        if isinstance(query, list):
            search_text = " ".join([q for q in query if isinstance(q, str)])
        else:
            search_text = str(query)

        keywords = set(re.findall(r"\w+", search_text.lower()))
        compressed_docs = []

        for doc in documents[:effective_top_n]:
            sentences = re.split(r"(?<=[.!?\n])\s+", doc.page_content)
            salient_sentences = [
                s.strip()
                for s in sentences
                if any(kw in s.lower() for kw in keywords if len(kw) > 2)
            ]

            compressed_text = (
                " ".join(salient_sentences) if salient_sentences else doc.page_content
            )
            compressed_docs.append(
                Document(
                    page_content=compressed_text,
                    metadata={**doc.metadata, "compressed": True},
                )
            )

        return compressed_docs

    def rerank(
        self,
        query: Union[str, List[str]],
        documents: List[Document],
        top_n: Optional[int] = None,
    ) -> List[Document]:
        return self.process(query=query, documents=documents, top_n=top_n)