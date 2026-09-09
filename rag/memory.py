from typing import List
from dataclasses import dataclass, field


@dataclass
class ChatMessage:
    role: str
    content: str


class ChatMemory:
    def __init__(self):
        self._messages: dict[str, List[ChatMessage]] = {}

    def get_messages(self, session_id: str) -> List[ChatMessage]:
        return self._messages.get(session_id, [])

    def add_user_message(self, session_id: str, content: str):
        self._messages.setdefault(session_id, []).append(
            ChatMessage(role="user", content=content)
        )

    def add_ai_message(self, session_id: str, content: str):
        self._messages.setdefault(session_id, []).append(
            ChatMessage(role="assistant", content=content)
        )


chat_memory = ChatMemory()