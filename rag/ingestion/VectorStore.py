from typing import Any, Dict, List, Optional
from langchain_community.vectorstores import PGVector
from langchain_core.documents import Document
from sqlalchemy import text
from config import settings
from rag.registry import registry


@registry.vector_store("pgvector")
class PGVectorStore:

    def __init__(
        self,
        embeddings,
        collection_name: str = "rag_collection",
        connection_string: str = settings.PGVECTOR_URL,
    ):
        self.embeddings = embeddings
        self.collection_name = collection_name
        self.connection_string = connection_string
        self.store: Optional[PGVector] = None

    def build(self, documents: Optional[List[Document]] = None) -> PGVector:
        if documents:
            self.store = PGVector.from_documents(
                documents=documents,
                embedding=self.embeddings,
                collection_name=self.collection_name,
                connection_string=self.connection_string,
                use_jsonb=True,
            )
        else:
            self.store = PGVector(
                embedding_function=self.embeddings,
                collection_name=self.collection_name,
                connection_string=self.connection_string,
                use_jsonb=True,
            )
        return self.store

    def clear(self):
        """Xóa toàn bộ chunks/vectors thuộc collection này mà không làm mất bản ghi collection."""
        if self.store is None:
            self.build([])
        assert self.store is not None

        with self.store._make_session() as session:
            query_col = text(
                "SELECT uuid FROM langchain_pg_collection WHERE name = :name"
            )
            col_record = session.execute(
                query_col, {"name": self.collection_name}
            ).fetchone()

            if col_record:
                col_uuid = col_record[0]
                delete_stmt = text(
                    "DELETE FROM langchain_pg_embedding WHERE collection_id = :col_uuid"
                )
                session.execute(delete_stmt, {"col_uuid": col_uuid})
                session.commit()
                print(
                    f"--> [AUTO CLEAN] Đã dọn sạch các vector cũ của collection '{self.collection_name}'."
                )

    def add_documents(self, documents: List[Document]):
        """Thêm tài liệu vào store."""
        if self.store is None:
            self.build(documents)
        else:
            self.store.add_documents(documents=documents)

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        if self.store is None:
            self.build([])
        assert self.store is not None
        if filter_dict:
            return self.store.similarity_search(query=query, k=top_k, filter=filter_dict)
        return self.store.similarity_search(query=query, k=top_k)

    def similarity_search(
        self,
        query: str,
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        return self.search(query=query, top_k=top_k, filter_dict=filter_dict)