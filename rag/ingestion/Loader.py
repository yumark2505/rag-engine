from rag.registry import registry
from unstructured.partition.auto import partition
from typing import List
from langchain_core.documents import Document
import os
from pathlib import Path


@registry.loader("unstructured")
class UnstructuredLoader:
    SUPPORTED_EXTENSIONS = {".pdf"}

    def __init__(self, data_dir: str = "./data/Document"):
        self.data_dir = data_dir

    def load(self, file_path: str) -> List[Document]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Không tìm thấy file: {file_path}")

        elements = partition(filename=file_path)
        full_text = "\n\n".join([str(el) for el in elements if str(el).strip()])
        return [
            Document(
                page_content=full_text,
                metadata={"source": file_path, "filename": Path(file_path).name}
            )
        ]

    def load_all(self) -> List[Document]:
        all_docs = []
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir, exist_ok=True)
            return all_docs

        for filename in os.listdir(self.data_dir):
            file_path = os.path.join(self.data_dir, filename)
            ext = Path(file_path).suffix.lower()
            if ext in self.SUPPORTED_EXTENSIONS:
                docs = self.load(file_path)
                all_docs.extend(docs)
        return all_docs
