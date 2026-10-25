import pytest
import json
import httpx
from fastapi.testclient import TestClient

from app.main import create_app
from app.memory.buffer import working_memory
from app.memory.pg_store import memory_store
from app.memory.models import MemoryCategory

# TestClient instance for synchronous integration tests
app = create_app()
client = TestClient(app)


def test_integration_health_endpoint():
    """Verify GET /health returns status code 200 and healthy services schema."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "services" in data
    assert "llm" in data["services"]


def test_integration_frontend_and_static_assets():
    """Verify index.html and static CSS/JS assets are served correctly."""
    # Test Root HTML
    root_res = client.get("/")
    assert root_res.status_code == 200
    assert "text/html" in root_res.headers.get("content-type", "")
    assert "<title>Multi-Agent Chat</title>" in root_res.text

    # Test CSS Static File
    css_res = client.get("/static/css/styles.css")
    assert css_res.status_code == 200
    assert "text/css" in css_res.headers.get("content-type", "")

    # Test JS Static File
    js_res = client.get("/static/js/app.js")
    assert js_res.status_code == 200
    assert "javascript" in js_res.headers.get("content-type", "")


def test_integration_documents_list_endpoint():
    """Verify GET /documents returns document list and status 200."""
    response = client.get("/documents")
    assert response.status_code == 200
    data = response.json()
    assert "documents" in data
    assert "total_count" in data
    assert isinstance(data["documents"], list)


def test_integration_memory_full_crud_lifecycle():
    """Verify complete Memory REST API lifecycle: Create -> List -> Recall -> Delete."""
    test_content = "Integration Test Fact: System uses LangGraph orchestration."

    # 1. POST /memory (Create)
    create_res = client.post("/memory", json={
        "content": test_content,
        "category": "fact",
        "importance": 4,
        "metadata": {"source": "integration_test"}
    })
    assert create_res.status_code == 200
    created_item = create_res.json()
    assert created_item["content"] == test_content
    assert created_item["category"] == "fact"
    memory_id = created_item["id"]

    # 2. GET /memory (List)
    list_res = client.get("/memory")
    assert list_res.status_code == 200
    list_data = list_res.json()
    assert any(m["id"] == memory_id for m in list_data["memories"])

    # 3. POST /memory/recall (Semantic & Keyword Query)
    recall_res = client.post("/memory/recall", json={
        "query": "LangGraph orchestration",
        "top_k": 3,
        "min_score": 0.1
    })
    assert recall_res.status_code == 200
    recall_data = recall_res.json()
    assert len(recall_data) > 0
    assert any(r["memory"]["id"] == memory_id for r in recall_data)

    # 4. DELETE /memory/{id} (Delete)
    del_res = client.delete(f"/memory/{memory_id}")
    assert del_res.status_code == 200
    assert del_res.json()["status"] == "success"

    # Verify deleted
    verify_list = client.get("/memory").json()
    assert not any(m["id"] == memory_id for m in verify_list["memories"])


def test_integration_chat_non_streaming():
    """Verify POST /chat non-streaming returns complete structured ChatResponse."""
    payload = {
        "conversation_id": "conv_integration_nonstream",
        "message": "Hello, please reply with 'OK' for integration test.",
        "stream": False,
        "history": [
            {"role": "user", "content": "Ping"},
            {"role": "assistant", "content": "Pong"}
        ]
    }
    response = client.post("/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert len(data["answer"]) > 0
    assert "agent" in data
    assert "latency_seconds" in data
    assert data["conversation_id"] == "conv_integration_nonstream"


def test_integration_chat_streaming_sse():
    """Verify POST /chat streaming returns SSE event-stream with metadata and tokens."""
    payload = {
        "conversation_id": "conv_integration_stream",
        "message": "Say hello in one sentence.",
        "stream": True
    }
    with client.stream("POST", "/chat", json=payload) as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")
        
        events = []
        for line in response.iter_lines():
            if line.startswith("data: "):
                event_data = json.loads(line[6:])
                events.append(event_data)
        
        event_types = [e.get("type") for e in events]
        assert "status" in event_types
        assert "metadata" in event_types
        assert "token" in event_types
        assert "done" in event_types


def test_integration_chat_history_restoration_across_restarts():
    """Verify dialogue turns sent in ChatRequest restore working memory seamlessly."""
    import uuid
    conv_id = f"conv_restart_{uuid.uuid4().hex[:8]}"
    
    # Simulate a prior session with 2 turns sent from client
    payload = {
        "conversation_id": conv_id,
        "message": "What is my city?",
        "stream": False,
        "history": [
            {"role": "user", "content": "My name is Carlos and I live in Barcelona."},
            {"role": "assistant", "content": "Nice to meet you, Carlos! Barcelona is wonderful."}
        ]
    }
    
    response = client.post("/chat", json=payload)
    assert response.status_code == 200
    
    # Verify turns were restored and preserved in WorkingMemoryBuffer
    history = working_memory.get_history(conv_id)
    assert len(history) >= 2
    prompt_history = working_memory.format_history_for_prompt(conv_id)
    assert "Barcelona" in prompt_history or "Carlos" in prompt_history
