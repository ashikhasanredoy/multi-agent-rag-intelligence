import json
from pathlib import Path
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
    Adapter that manages thread-scoped conversation working memory with
    LangGraph MemorySaver checkpointing AND persistent disk storage.

    Stores conversation turns in ./data/chat_history.json so that when the
    app or server is closed and reopened, past conversation history and dialogue
    context are preserved seamlessly.
    """

    def __init__(self, storage_path: str = "./data/chat_history.json"):
        self._saver = _memory_saver
        self.storage_path = Path(storage_path)
        self._fallback: Dict[str, List[Dict[str, Any]]] = {}
        self._load_from_disk()

    def _load_from_disk(self):
        try:
            if self.storage_path.exists():
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    self._fallback = json.load(f)
                logger.info(f"Loaded dialogue history for {len(self._fallback)} conversations from {self.storage_path}")
        except Exception as e:
            logger.warning(f"Could not load conversation history from disk: {e}")
            self._fallback = {}

    def _save_to_disk(self):
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(self._fallback, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Could not persist conversation history to disk: {e}")

    # ------------------------------------------------------------------ #
    #  Core turn management
    # ------------------------------------------------------------------ #

    def add_turn(self, conversation_id: str, role: str, content: str):
        """Append a turn to the conversation history and persist to disk."""
        if not conversation_id or not content:
            return
        turn = {"role": role, "content": content}
        history = self._fallback.setdefault(conversation_id, [])
        history.append(turn)
        # Keep a sliding window of 30 messages (15 turns)
        if len(history) > 30:
            self._fallback[conversation_id] = history[-30:]
        self._save_to_disk()

    def sync_history(self, conversation_id: str, turns: List[Dict[str, Any]]):
        """Restore or backfill conversation history turns from client or previous session."""
        if not conversation_id or not turns:
            return
        cleaned_turns = [
            {"role": t.get("role", "user"), "content": t.get("content", "")}
            for t in turns
            if t.get("content")
        ]
        if not cleaned_turns:
            return
        existing = self._fallback.get(conversation_id, [])
        if len(cleaned_turns) >= len(existing):
            self._fallback[conversation_id] = cleaned_turns[-30:]
            self._save_to_disk()

    def get_history(
        self, conversation_id: str, limit_turns: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        history = self._fallback.get(conversation_id, [])
        if limit_turns:
            return history[-(limit_turns * 2) :]
        return history

    def format_history_for_prompt(
        self, conversation_id: str, max_turns: int = 5
    ) -> str:
        turns = self.get_history(conversation_id, limit_turns=max_turns)
        if not turns:
            return ""
        lines = []
        for t in turns:
            speaker = "User" if t.get("role") == "user" else "Assistant"
            lines.append(f"{speaker}: {t.get('content', '')}")
        return "\n".join(lines)

    def clear(self, conversation_id: Optional[str] = None):
        if conversation_id:
            self._fallback.pop(conversation_id, None)
        else:
            self._fallback.clear()
        self._save_to_disk()


# Singleton
working_memory = WorkingMemoryBuffer()

