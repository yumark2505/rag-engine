from typing import List, Optional, Literal
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from rag.registry import registry
from config import settings

@registry.chunker("recursive")
class RecursiveChunker: 
    def __init__(
            self,
            chunk_size: int = settings.CHUNK_SIZE,
            chunk_overlap: int = settings.CHUNK_OVERLAP
    ):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
    def split(self, documents: List[Document]) -> List[Document]:
        return self.splitter.split_documents(documents=documents)

@registry.chunker("tiktoken")
class TiktokenChunker:

    def __init__(
        self,
        chunk_size: int = 500,  # 500 tokens ~ 1500-2000 ký tự
        chunk_overlap: int = 50,
        encoding_name: str = "cl100k_base",
    ):
        self.splitter = TokenTextSplitter(
            encoding_name=encoding_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def split(self, documents: List[Document]) -> List[Document]:
        return self.splitter.split_documents(documents)

BreakpointThresholdType = Literal[
    "percentile", "standard_deviation", "interquartile", "gradient"
]

@registry.chunker("semantic")  #[cite: 1]
class DynamicSemanticChunker:

    def __init__(
        self,
        embedder=None,
        breakpoint_threshold_type: BreakpointThresholdType = "percentile",
        breakpoint_threshold_amount: float = 85.0,
    ):
        if embedder is None:
            embedder_cls = registry.get_embedder("ollama")
            if not embedder_cls:
                raise ValueError("Embedder 'ollama' chưa được đăng ký trong registry!")  
            self.embedder = embedder_cls()
        else:
            self.embedder = embedder

        self.splitter = SemanticChunker(
            embeddings=self.embedder,
            breakpoint_threshold_type=breakpoint_threshold_type,
            breakpoint_threshold_amount=breakpoint_threshold_amount,
        )

    def split(self, documents: List[Document]) -> List[Document]:
        return self.splitter.split_documents(documents)