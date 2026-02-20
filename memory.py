"""
Agent memory: append-only list with a get_summary() for the Reason/Decide prompts.
"""

from typing import List


class AgentMemory:
    def __init__(self, max_items: int = 50) -> None:
        self._items: List[str] = []
        self._max_items = max_items

    def add(self, item: str) -> None:
        self._items.append(item)
        if len(self._items) > self._max_items:
            self._items = self._items[-self._max_items :]

    def get_summary(self) -> str:
        if not self._items:
            return "(no memory yet)"
        return "\n".join(self._items)

    def __len__(self) -> int:
        return len(self._items)
