from abc import ABC, abstractmethod
from typing import Iterator, List
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from rag.registry import registry


class BaseGenerator(ABC):

    def __init__(
        self,
        prompt_name: str = "conversational",
        temperature: float = 0.0,
        **kwargs,
    ):
        prompt_cls = registry.get_prompt(prompt_name)
        if prompt_cls is None:
            prompt_cls = registry.get_prompt("basic")
        if prompt_cls is None:
            raise ValueError(
                f"Không tìm thấy template prompt '{prompt_name}' hoặc 'basic' trong registry!"
            )

        self.prompt_instance = prompt_cls()
        self.temperature = temperature

    def _build_prompt_text(
        self,
        query: str,
        context_docs: List[Document],
        chat_history: str = "",
    ) -> str:
        prompt_template = self.prompt_instance.build(
            context_docs=context_docs,
            chat_history=chat_history,
        )
        prompt_value = prompt_template.format_prompt(question=query)
        return prompt_value.to_string()

    def _build_messages(
        self,
        query: str,
        context_docs: List[Document],
        chat_history: str = "",
    ) -> List[HumanMessage]:
        prompt_text = self._build_prompt_text(query, context_docs, chat_history)
        return [HumanMessage(content=prompt_text)]

    @abstractmethod
    def run(
        self,
        query: str,
        context_docs: List[Document],
        chat_history: str = "",
    ) -> str:
        pass

    @abstractmethod
    def stream(
        self,
        query: str,
        context_docs: List[Document],
        chat_history: str = "",
    ) -> Iterator[str]:
        pass