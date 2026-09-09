from typing import List
from tqdm import tqdm
import ollama
from langchain_core.embeddings import Embeddings
from rag.registry import registry
from config import settings


@registry.embedder("ollama")
class OllamaEmbedder(Embeddings):
    def __init__(
        self,
        model_name: str = settings.EMBEDDING_MODEL,
        base_url: str = settings.OLLAMA_BASE_URL,
        batch_size: int = settings.BATCH_SIZE,
    ):
        self.model_name = model_name
        self.client = ollama.Client(host=base_url)
        self.batch_size = batch_size

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        all_embeddings: List[List[float]] = []
        total_texts = len(texts)

        print(f"Embedding {total_texts} chunks (batch_size={self.batch_size})")

        for i in tqdm(range(0, total_texts, self.batch_size)):
            batch = texts[i : i + self.batch_size]
            try:
                response = self.client.embed(
                    model=self.model_name,
                    input=batch,
                )
                all_embeddings.extend(response["embeddings"])
            except Exception as e:
                raise RuntimeError(f"Embedding failed for batch {i}: {e}")
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        response = self.client.embed(
            model=self.model_name,
            input=text,
        )
        return response["embeddings"][0]