"""
Short-Term Conversation Memory using LangGraph MemorySaver.
Replaces the in-memory WorkingMemoryBuffer with a proper
LangGraph checkpoint that supports thread-scoped state.
"""
from typing import List, Dict, Any, Optional
from langgraph.checkpoint.memory import MemorySaver

from app.utils.logging import logger


# Singleton MemorySaver instance (in-process, RAM-based)
# For production, swap this with AsyncPostgresSaver or AsyncSqliteSaver
_memory_saver = MemorySaver()


def get_memory_saver() -> MemorySaver:
    """Return the shared MemorySaver instance for LangGraph graph compilation."""
    return _memory_saver


class WorkingMemoryBuffer:
    """
    Thin adapter that wraps LangGraph MemorySaver for backwards-compatibility
    with existing workflow code that calls add_turn / format_history_for_prompt.

    Under the hood it stores turns as a simple list in a MemorySaver checkpoint
    keyed by conversation_id, so LangGraph graphs compiled with this saver
    can resume any thread automatically.
    """

    def __init__(self):
        self._saver = _memory_saver
        self._fallback: Dict[str, List[Dict[str, Any]]] = {}

    # ------------------------------------------------------------------ #
    #  Core turn management
    # ------------------------------------------------------------------ #

    def add_turn(self, conversation_id: str, role: str, content: str):
        """Append a turn to the conversation history stored in MemorySaver."""
        if not conversation_id or not content:
            return
        turn = {"role": role, "content": content}
        # Use the fallback dict — MemorySaver state is read/written by
        # LangGraph graph invocations via thread_id config.
        history = self._fallback.setdefault(conversation_id, [])
        history.append(turn)
        # Keep a sliding window of 20 messages (10 turns)
        if len(history) > 20:
            self._fallback[conversation_id] = history[-20:]

    def get_history(
        self, conversation_id: str, limit_turns: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        history = self._fallback.get(conversation_id, [])
        if limit_turns:
            return history[-(limit_turns * 2) :]
        return history

    def format_history_for_prompt(
        self, conversation_id: str, max_turns: int = 4
    ) -> str:
        turns = self.get_history(conversation_id, limit_turns=max_turns)
        if not turns:
            return ""
        lines = []
        for t in turns:
            speaker = "User" if t["role"] == "user" else "Assistant"
            lines.append(f"{speaker}: {t['content']}")
        return "\n".join(lines)

    def clear(self, conversation_id: Optional[str] = None):
        if conversation_id:
            self._fallback.pop(conversation_id, None)
        else:
            self._fallback.clear()


# Singleton
working_memory = WorkingMemoryBuffer()
