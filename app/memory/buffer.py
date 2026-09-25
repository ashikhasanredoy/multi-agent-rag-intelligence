from typing import List, Dict, Any, Optional
from collections import defaultdict
from datetime import datetime


class WorkingMemoryBuffer:
    """
    In-memory dialogue buffer that maintains recent multi-turn conversation history.
    """

    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        # conversation_id -> list of message dicts {"role": "user"|"assistant", "content": str, "timestamp": str}
        self._history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    def add_turn(self, conversation_id: str, role: str, content: str):
        """Append a dialogue turn to conversation working memory."""
        if not conversation_id or not content:
            return
        turn = {
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat()
        }
        self._history[conversation_id].append(turn)
        # Keep within sliding window limit
        if len(self._history[conversation_id]) > self.max_turns * 2:
            self._history[conversation_id] = self._history[conversation_id][-self.max_turns * 2:]

    def get_history(self, conversation_id: str, limit_turns: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieve recent dialogue turns for the conversation."""
        turns = self._history.get(conversation_id, [])
        if limit_turns:
            return turns[-limit_turns * 2:]
        return turns

    def format_history_for_prompt(self, conversation_id: str, max_turns: int = 4) -> str:
        """Format recent dialogue turns into a clean markdown dialogue string."""
        turns = self.get_history(conversation_id, limit_turns=max_turns)
        if not turns:
            return ""
        formatted = []
        for t in turns:
            speaker = "User" if t["role"] == "user" else "Assistant"
            formatted.append(f"{speaker}: {t['content']}")
        return "\n".join(formatted)

    def clear(self, conversation_id: Optional[str] = None):
        """Clear history for a specific conversation or all."""
        if conversation_id:
            if conversation_id in self._history:
                del self._history[conversation_id]
        else:
            self._history.clear()


# Singleton working memory buffer
working_memory = WorkingMemoryBuffer()
