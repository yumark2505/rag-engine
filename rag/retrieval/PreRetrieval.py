import json
from typing import List
import ollama
from rag.registry import registry
from config import settings
from rag.memory import ChatMessage

@registry.retriever("pre_retrieval_identity")
class IdentityQueryOptimizer:
    """Chiến lược Identity: Giữ nguyên câu hỏi gốc, không qua LLM để tối ưu latency."""

    def optimize(self, query: str) -> List[str]:
        return [query]

    def contextualize_query(self, query: str, history: List[ChatMessage]) -> str:
        return query

@registry.retriever("pre_retrieval_query_optimizer")
class QueryOptimizer:
    def __init__(
        self,
        model_name: str = settings.LLM_MODEL,
        base_url: str = settings.OLLAMA_BASE_URL,
    ):
        self.model_name = model_name
        self.client = ollama.Client(host=base_url)

    def rewrite_query(self, query: str) -> str:
        prompt = (
            f"You are an expert in search query optimization for a RAG system.\n"
            f"Rewrite the following question so that it is clear, preserves the original intent, "
            f"and is as easy as possible to use for retrieving relevant text.\n"
            f"Return only the rewritten question. Do not provide any explanation or additional text.\n"
            f"Original question: {query}"
        )
        response = self.client.generate(model=self.model_name, prompt=prompt)
        return response["response"].strip()

    def decompose_query(self, query: str) -> List[str]:
        prompt = (
            f"Decompose the following question into 2 or 3 smaller sub-questions "
            f"to retrieve sufficient context to answer the original question.\n"
            f"Return the result as a plain JSON list (do not use a markdown ```json block):\n"
            f'["question 1", "question 2"]\n'
            f"Question: {query}"
        )
        response = self.client.generate(model=self.model_name, prompt=prompt)
        raw_text = response["response"].strip()
        try:
            if "```" in raw_text:
                raw_text = raw_text.split("```")[1]
                if raw_text.startswith("json"):
                    raw_text = raw_text[4:]
            sub_queries = json.loads(raw_text.strip())
            if isinstance(sub_queries, list):
                return sub_queries
        except Exception:
            pass
        return [query]

    def contextualize_query(self, query: str, history: List[ChatMessage]) -> str:
        if not history:
            return query

        history_text = "\n".join(
            [f"{msg.role.upper()}: {msg.content}" for msg in history]
        )

        prompt = (
            f"<|im_start|>system\n"
            f"Dựa vào lịch sử hội thoại dưới đây, hãy viết lại câu hỏi mới nhất của người dùng thành một câu hỏi độc lập, "
            f"đầy đủ chủ ngữ, vị ngữ và ngữ cảnh để hệ thống tìm kiếm tài liệu có thể hiểu được.\n"
            f"Nếu câu hỏi đã rõ ràng và không phụ thuộc vào lịch sử, hãy giữ nguyên.\n"
            f"CHỈ TRẢ VỀ CÂU HỎI ĐÃ VIẾT LẠI, không thêm lời giải thích hay hội thoại thừa.<|im_end|>\n"
            f"<|im_start|>user\n"
            f"LỊCH SỬ HỘI THOẠI:\n{history_text}\n\n"
            f"CÂU HỎI MỚI: {query}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        try:
            res = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                options={"temperature": 0.0},
            )
            stand_alone_query = res["response"].strip()
            return stand_alone_query if stand_alone_query else query
        except Exception:
            return query

@registry.retriever("pre_retrieval_hyde")
class HyDEOptimizer:
    """Chiến lược HyDE: Tạo tài liệu giả định để tăng cường độ tương đồng vector."""

    def __init__(
        self,
        model_name: str = settings.LLM_MODEL,
        base_url: str = settings.OLLAMA_BASE_URL,
    ):
        self.model_name = model_name
        self.client = ollama.Client(host=base_url)

    def optimize(self, query: str) -> List[str]:
        prompt = (
            "Bạn là một chuyên gia. Hãy viết một đoạn văn ngắn (dưới 80 từ) trả lời hoặc "
            f"giải thích trực tiếp câu hỏi sau đây:\nCâu hỏi: {query}\nĐoạn văn:"
        )
        try:
            res = self.client.generate(model=self.model_name, prompt=prompt)
            hypo_doc = res["response"].strip()
            return [query, hypo_doc] if hypo_doc else [query]
        except Exception:
            return [query]

    def contextualize_query(self, query: str, history: List[ChatMessage]) -> str:
        return query