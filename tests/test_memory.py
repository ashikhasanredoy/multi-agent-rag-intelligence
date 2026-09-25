import pytest
from app.memory.models import MemoryCategory, MemoryItem
from app.memory.store import LongTermMemoryStore
from app.memory.buffer import WorkingMemoryBuffer
from app.memory.extractor import MemoryExtractor


@pytest.mark.asyncio
async def test_working_memory_buffer():
    buffer = WorkingMemoryBuffer(max_turns=3)
    conv_id = "test_conv_123"
    
    buffer.add_turn(conv_id, "user", "Hello assistant")
    buffer.add_turn(conv_id, "assistant", "Hello! How can I help you today?")
    
    history = buffer.get_history(conv_id)
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"
    
    formatted = buffer.format_history_for_prompt(conv_id)
    assert "User: Hello assistant" in formatted
    assert "Assistant: Hello!" in formatted


@pytest.mark.asyncio
async def test_long_term_memory_store(tmp_path):
    storage_file = str(tmp_path / "test_memories.json")
    store = LongTermMemoryStore(storage_path=storage_file)
    
    # 1. Add Memory
    item = await store.add_memory(
        content="User prefers Python and uses macOS.",
        category=MemoryCategory.PREFERENCE,
        importance=4,
        embed=False
    )
    assert item.id.startswith("mem_")
    assert item.content == "User prefers Python and uses macOS."
    assert item.category == MemoryCategory.PREFERENCE
    assert len(store.memories) == 1
    
    # 2. List Memories
    memories = store.list_memories()
    assert len(memories) == 1
    
    # 3. Keyword recall test (without vector embedding)
    recalled = await store.recall_memories(query="What programming language does user prefer?", min_score=0.1)
    assert len(recalled) >= 1
    assert recalled[0].memory.id == item.id
    
    # 4. Delete Memory
    deleted = await store.delete_memory(item.id)
    assert deleted is True
    assert len(store.list_memories()) == 0


@pytest.mark.asyncio
async def test_memory_extractor_fast_patterns():
    extractor = MemoryExtractor()
    
    # Explicit remember
    facts = extractor._fast_pattern_extract("Remember that our database port is 5432.", "conv_1")
    assert len(facts) >= 1
    assert "database port is 5432" in facts[0]["content"]
    
    # Name extraction
    facts_name = extractor._fast_pattern_extract("Hi, my name is Alex.", "conv_1")
    assert len(facts_name) >= 1
    assert "Alex" in facts_name[0]["content"]
