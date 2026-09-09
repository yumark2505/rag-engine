from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


class RagQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Câu hỏi gửi lên RAG")
    session_id: str = Field(default="default_session", description="Định danh phiên hội thoại")
    selected_document: Optional[str] = Field(default=None, description="Lọc theo tên tệp tài liệu cụ thể")
    
    llm_provider: Literal["ollama", "openai", "google", "vllm"] = Field(
        default="ollama",
        description="Mô hình LLM được sử dụng để sinh câu trả lời",
    )

    top_k: int = Field(default=5, ge=1, le=20, description="Số lượng chunk tìm kiếm ban đầu")
    top_n: int = Field(default=3, ge=1, le=10, description="Số lượng chunk giữ lại sau rerank")

    # Các chiến lược tương tác tùy chọn
    pre_retrieval_strategy: Literal["identity", "query_transform", "query_optimizer", "hyde"] = Field(
        default="identity",
        description="Chiến lược tiền xử lý truy vấn",
    )
    retrieval_strategy: Literal["vector", "bm25", "hybrid"] = Field(
        default="vector",
        description="Chiến lược công cụ tìm kiếm",
    )
    post_retrieval_strategy: Literal["rerank", "contextual_compression"] = Field(
        default="rerank",
        description="Chiến lược hậu xử lý văn bản",
    )


class DocumentSource(BaseModel):
    content: str
    metadata: Dict[str, Any] = {}

    @field_validator("metadata", mode="before")
    @classmethod
    def sanitize_metadata_types(cls, v: Any) -> Any:
        """Tự động khử các kiểu dữ liệu NumPy sang kiểu Python gốc."""
        if not isinstance(v, dict):
            return v

        def clean_val(val: Any) -> Any:
            if hasattr(val, "item"):
                return val.item()
            if hasattr(val, "tolist"):
                return val.tolist()
            if isinstance(val, dict):
                return {k: clean_val(sub_v) for k, sub_v in val.items()}
            if isinstance(val, list):
                return [clean_val(sub_v) for sub_v in val]
            return val

        return {key: clean_val(val) for key, val in v.items()}


class RagQueryResponse(BaseModel):
    query: str
    sub_queries: List[str]
    answer: str
    sources: List[DocumentSource]


class IngestionRequest(BaseModel):
    data_dir: str = "./data/Document"
    loader_name: str = "unstructured"
    chunker_name: str = "recursive"
    embedder_name: str = "ollama"
    vector_store_name: str = "pgvector"