from abc import ABC, abstractmethod
from typing import List
from langchain_core.documents import Document
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from rag.registry import registry

SYSTEM_INSTRUCTION = (
    "Bạn là một trợ lý AI thông minh, trung thực và chuyên nghiệp.\n"
    "Hãy trả lời câu hỏi của người dùng CHỈ dựa trên thông tin được cung cấp trong phần NGỮ CẢNH dưới đây.\n"
    "Nếu thông tin trong ngữ cảnh không đủ để trả lời, hãy thành thật nói: "
    "'Tôi không tìm thấy thông tin này trong tài liệu được cung cấp', tuyệt đối không tự bịa đặt câu trả lời.\n"
    "Luôn trích dẫn nguồn [filename:page] khi áp dụng."
)


class BasePrompt(ABC):

    @abstractmethod
    def build(self, *args, **kwargs) -> ChatPromptTemplate:
        pass

    @staticmethod
    def _format_context(documents: List[Document]) -> str:
        if not documents:
            return "Không có tài liệu ngữ cảnh nào được tìm thấy."
        blocks = []
        for idx, doc in enumerate(documents, 1):
            source = doc.metadata.get(
                "filename", doc.metadata.get("source", "Tài liệu")
            )
            blocks.append(
                f"[Đoạn trích {idx} - Nguồn: {source}]\n{doc.page_content.strip()}"
            )
        return "\n\n".join(blocks)


@registry.prompt("basic")
class BasicPrompt(BasePrompt):

    def build(self, context_docs: List[Document], chat_history: str = "") -> ChatPromptTemplate:
        context_text = self._format_context(context_docs)
        # Escape các ngoặc nhọn trong tài liệu (tránh lỗi cú pháp công thức LaTeX như {crit})
        safe_context = context_text.replace("{", "{{").replace("}", "}}")

        system_message = f"{SYSTEM_INSTRUCTION}\n\nNGỮ CẢNH TÀI LIỆU:\n{safe_context}"

        return ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(system_message),
                HumanMessagePromptTemplate.from_template("{question}"),
            ]
        )


@registry.prompt("conversational")
class ConversationalPrompt(BasePrompt):

    def build(
        self, context_docs: List[Document], chat_history: str = ""
    ) -> ChatPromptTemplate:
        context_text = self._format_context(context_docs)
        # Escape các ngoặc nhọn trong cả context và chat_history
        safe_context = context_text.replace("{", "{{").replace("}", "}}")
        safe_history = (chat_history or "Chưa có lịch sử.").replace("{", "{{").replace("}", "}}")

        system_message = (
            f"{SYSTEM_INSTRUCTION}\n\n"
            f"NGỮ CẢNH TÀI LIỆU:\n{safe_context}\n\n"
            f"LỊCH SỬ ĐÀM THOẠI GẦN NHẤT:\n{safe_history}"
        )

        return ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(system_message),
                HumanMessagePromptTemplate.from_template("{question}"),
            ]
        )