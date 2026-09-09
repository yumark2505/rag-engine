from typing import Iterator, List
from langchain_core.documents import Document
import ollama
from config import settings  
from rag.generation.base import BaseGenerator
from rag.registry import registry  


@registry.generator("ollama")
class OllamaGenerator(BaseGenerator):

    def __init__(
        self,
        model_name: str = settings.LLM_MODEL,  
        base_url: str = settings.OLLAMA_BASE_URL,  
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.client = ollama.Client(host=base_url)

    def run(
        self, query: str, context_docs: List[Document], chat_history: str = ""
    ) -> str:
        prompt = self._build_prompt_text(query, context_docs, chat_history)
        res = self.client.generate(
            model=self.model_name,
            prompt=prompt,
            options={"temperature": self.temperature},
        )
        return res["response"].strip()

    def stream(
        self, query: str, context_docs: List[Document], chat_history: str = ""
    ) -> Iterator[str]:
        prompt = self._build_prompt_text(query, context_docs, chat_history)
        stream_res = self.client.generate(
            model=self.model_name,
            prompt=prompt,
            stream=True,
            options={"temperature": self.temperature},
        )
        for chunk in stream_res:
            yield chunk.get("response", "")