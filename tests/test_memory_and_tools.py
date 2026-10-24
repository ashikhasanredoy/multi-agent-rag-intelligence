import os
import json
import pytest
from pathlib import Path

from app.memory.buffer import WorkingMemoryBuffer
from app.memory.pg_store import PostgresLongTermMemoryStore
from app.memory.models import MemoryCategory, MemoryItem
from app.tools.code_tool import code_tool
from app.tools.finance_tool import finance_tool
from app.models.schemas import ChatRequest


@pytest.mark.asyncio
async def test_code_tool_safe_math_evaluation():
    """Verify CodeTool evaluates allowed mathematical expressions safely."""
    res1 = code_tool.evaluate_math_expression("2 + 2 * 10")
    assert res1["success"] is True
    assert res1["result"] == 22

    res2 = code_tool.evaluate_math_expression("sqrt(144) + pow(2, 3)")
    assert res2["success"] is True
    assert res2["result"] == 20.0


@pytest.mark.asyncio
async def test_code_tool_blocks_dangerous_execution():
    """Verify CodeTool refuses to evaluate unauthorized or malicious system code."""
    res = code_tool.evaluate_math_expression("__import__('os').system('echo hack')")
    assert res["success"] is False
    assert "error" in res


@pytest.mark.asyncio
async def test_finance_tool_ticker_mapping():
    """Verify FinanceTool resolves both names and symbols accurately."""
    assert finance_tool.resolve_ticker("Apple") == "AAPL"
    assert finance_tool.resolve_ticker("samsung") == "005930.KS"
    assert finance_tool.resolve_ticker("Tesla") == "TSLA"
    assert finance_tool.resolve_ticker("NVDA") == "NVDA"
    assert finance_tool.resolve_ticker("Microsoft Corporation") == "MSFT"


@pytest.mark.asyncio
async def test_working_memory_disk_persistence(tmp_path):
    """Verify WorkingMemoryBuffer persists dialogue turns to disk across instances."""
    test_file = tmp_path / "test_chat_history.json"
    
    # Instance 1: write turns
    buffer1 = WorkingMemoryBuffer(storage_path=str(test_file))
    buffer1.add_turn("conv_test_1", "user", "Hello, who are you?")
    buffer1.add_turn("conv_test_1", "assistant", "I am your AI pair programmer.")
    
    assert test_file.exists()
    
    # Instance 2: reload from same disk file
    buffer2 = WorkingMemoryBuffer(storage_path=str(test_file))
    history = buffer2.get_history("conv_test_1")
    assert len(history) == 2
    assert history[0]["content"] == "Hello, who are you?"
    assert history[1]["content"] == "I am your AI pair programmer."
    
    prompt = buffer2.format_history_for_prompt("conv_test_1")
    assert "User: Hello, who are you?" in prompt
    assert "Assistant: I am your AI pair programmer." in prompt


@pytest.mark.asyncio
async def test_working_memory_sync_history(tmp_path):
    """Verify WorkingMemoryBuffer syncs prior conversation turns from client."""
    test_file = tmp_path / "test_chat_history_sync.json"
    buffer = WorkingMemoryBuffer(storage_path=str(test_file))
    
    prior_turns = [
        {"role": "user", "content": "I live in Tokyo"},
        {"role": "assistant", "content": "Got it, Tokyo is great."}
    ]
    buffer.sync_history("conv_sync_1", prior_turns)
    
    history = buffer.get_history("conv_sync_1")
    assert len(history) == 2
    assert history[0]["content"] == "I live in Tokyo"


@pytest.mark.asyncio
async def test_long_term_memory_persistent_fallback(tmp_path):
    """Verify long term memory store falls back to disk and recalls items correctly."""
    store = PostgresLongTermMemoryStore()
    
    # Verify list_memories works (via PG or disk fallback)
    memories = await store.list_memories()
    assert isinstance(memories, list)
    
    # Add a memory
    test_content = "User prefers Rust for performance critical code."
    added = await store.add_memory(
        content=test_content,
        category=MemoryCategory.PREFERENCE,
        importance=5,
        embed=False
    )
    assert added.content == test_content
    assert added.category == MemoryCategory.PREFERENCE
    
    # Recall memories
    recalled = await store.recall_memories("What programming language does the user prefer?", top_k=3, min_score=0.1)
    assert len(recalled) > 0
    contents = [r.memory.content for r in recalled]
    assert any("Rust" in c or "Python" in c for c in contents)


def test_chat_request_schema_with_history():
    """Verify ChatRequest schema validates message and optional history turns."""
    req = ChatRequest(
        conversation_id="conv_abc",
        message="Can you help me?",
        history=[
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"}
        ]
    )
    assert req.conversation_id == "conv_abc"
    assert req.message == "Can you help me?"
    assert len(req.history) == 2
