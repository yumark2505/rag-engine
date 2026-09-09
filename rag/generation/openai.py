from typing import Iterator, List
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from rag.generation.base import BaseGenerator
from rag.registry import registry


@registry.generator("openai")
class OpenAIGenerator(BaseGenerator):

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        api_key: str = "",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.client = ChatOpenAI(
            model=model_name,
            api_key=SecretStr(api_key),
            temperature=self.temperature,
        )

    def run(
        self, query: str, context_docs: List[Document], chat_history: str = ""
    ) -> str:
        messages = self._build_messages(query, context_docs, chat_history)
        response = self.client.invoke(messages)
        return str(response.content)

    def stream(
        self, query: str, context_docs: List[Document], chat_history: str = ""
    ) -> Iterator[str]:
        messages = self._build_messages(query, context_docs, chat_history)
        for chunk in self.client.stream(messages):
            if chunk.content:
                yield str(chunk.content)