import json
import re
from typing import List, Dict, Any, Optional
from app.memory.models import MemoryCategory
from app.memory.store import memory_store
from app.services.llm import llm_service
from app.utils.logging import logger


class MemoryExtractor:
    """
    Autonomous Memory Consolidation Engine.
    Detects user profile details, persistent preferences, project facts, and custom rules
    from conversations and saves them to Long-Term Memory.
    """

    EXPLICIT_REMEMBER_TRIGGERS = [
        r"\bremember\s+(?:that\s+)?(.+)",
        r"\bkeep\s+in\s+mind\s+(?:that\s+)?(.+)",
        r"\bmy\s+name\s+is\s+(.+)",
        r"\bi\s+(?:am|work\s+as|work\s+at)\s+(.+)",
        r"\bi\s+(?:prefer|like|love|hate|always\s+use)\s+(.+)",
        r"\bnever\s+(?:use|include|show)\s+(.+)",
        r"\balways\s+(?:use|include|respond)\s+(.+)",
    ]

    async def process_turn(self, query: str, assistant_response: str, conversation_id: str):
        """
        Extract facts and preferences from the dialogue turn.
        Uses fast heuristic pattern checking first, and falls back to LLM extraction
        when high-signal memory cues are present.
        """
        clean_q = query.strip()
        if len(clean_q) < 8:
            return

        # 1. Fast Heuristic Extraction for explicit commands
        heuristic_extracted = self._fast_pattern_extract(clean_q, conversation_id)
        if heuristic_extracted:
            for item in heuristic_extracted:
                await memory_store.add_memory(
                    content=item["content"],
                    category=item["category"],
                    importance=item.get("importance", 4),
                    metadata={"source": "heuristic", "conversation_id": conversation_id}
                )
            return

        # 2. Heuristic check to decide if LLM extraction is needed
        # Avoid running LLM extractor on generic questions (e.g. "what is weather", "stock price of TSLA")
        should_extract = self._should_run_llm_extraction(clean_q)
        if not should_extract:
            return

        # 3. LLM-based Memory Extraction
        try:
            extracted_facts = await self._llm_extract_facts(clean_q, assistant_response)
            for fact in extracted_facts:
                cat_str = fact.get("category", "general")
                try:
                    category = MemoryCategory(cat_str)
                except Exception:
                    category = MemoryCategory.GENERAL

                await memory_store.add_memory(
                    content=fact.get("content", ""),
                    category=category,
                    importance=int(fact.get("importance", 3)),
                    metadata={"source": "autonomous_llm", "conversation_id": conversation_id}
                )
        except Exception as e:
            logger.warning(f"Autonomous memory extraction encountered error: {e}")

    def _fast_pattern_extract(self, query: str, conversation_id: str) -> List[Dict[str, Any]]:
        """Fast regex extraction for explicit remember directives."""
        extracted = []
        lower_q = query.lower()

        # Explicit "remember that ..."
        match = re.search(r"\bremember\s+(?:that\s+)?(.+)", query, re.IGNORECASE)
        if match:
            extracted.append({
                "content": match.group(1).strip().rstrip("."),
                "category": MemoryCategory.INSTRUCTION,
                "importance": 5
            })

        # Name detection
        match_name = re.search(r"\bmy\s+name\s+is\s+([a-zA-Z\s]+?)(?:\.|$|,|\sand\b)", query, re.IGNORECASE)
        if match_name:
            extracted.append({
                "content": f"User's name is {match_name.group(1).strip()}",
                "category": MemoryCategory.USER_PROFILE,
                "importance": 5
            })

        # Preference detection
        match_pref = re.search(r"\bi\s+(?:prefer|always\s+use|like)\s+(.+)", query, re.IGNORECASE)
        if match_pref and not match:
            extracted.append({
                "content": f"User preference: {match_pref.group(1).strip().rstrip('.')}",
                "category": MemoryCategory.PREFERENCE,
                "importance": 4
            })

        return extracted

    def _should_run_llm_extraction(self, query: str) -> bool:
        """Check if message contains potential personal facts, enduring instructions or project info."""
        lower_q = query.lower()
        triggers = [
            "i am ", "i'm ", "my project", "we are building", "our stack", "i work",
            "my company", "my email", "i live in", "prefer", "favorite", "usually",
            "from now on", "call me", "keep in mind", "note that"
        ]
        return any(t in lower_q for t in triggers)

    async def _llm_extract_facts(self, query: str, assistant_response: str) -> List[Dict[str, Any]]:
        """Ask LLM to extract structured facts from dialogue."""
        prompt = f"""You are an autonomous long-term memory extraction system.
Analyze the user's message and identify any permanent facts about the user, their preferences, their projects, or persistent instructions that should be remembered across conversations.

Ignore transient questions, generic greetings, search requests, or temporary tasks.

User Message: "{query}"

If there are enduring facts or preferences to remember, return ONLY a valid JSON array of objects with keys:
- "content": concise declarative fact (e.g. "User is a Senior Python Engineer", "User prefers concise answers with type hints")
- "category": one of ["user_profile", "preference", "fact", "project", "instruction", "general"]
- "importance": integer from 1 to 5

If nothing needs to be permanently remembered, return an empty array `[]`.
JSON Output:"""

        response = await llm_service.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        
        # Parse JSON
        text = response.strip()
        if "[" in text and "]" in text:
            json_str = text[text.find("["):text.rfind("]") + 1]
            data = json.loads(json_str)
            if isinstance(data, list):
                return [d for d in data if isinstance(d, dict) and d.get("content")]
        return []


# Singleton memory extractor
memory_extractor = MemoryExtractor()
