"""
Test-Driven Development (TDD) Suite
=====================================
Cycle: RED -> GREEN -> REFACTOR

Tests are written FIRST to specify exact expected behaviors across 8 domains:
  1.  Supervisor Routing (intent classification contracts)
  2.  FinanceTool (ticker resolution + quote shape)
  3.  CodeTool (safe eval + dangerous input rejection)
  4.  WorkingMemoryBuffer (CRUD + sliding window + persistence)
  5.  Pydantic Schema Validation
  6.  API Error Contracts (422 / 404 / 200)
  7.  FinanceTool Ticker Mapping Completeness
  8.  Supervisor Edge Cases

Run with:
    python -m pytest tests/test_tdd_suite.py -v
"""
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

from app.main import create_app
from app.agents.supervisor import SupervisorAgent
from app.tools.finance_tool import FinanceTool
from app.tools.code_tool import CodeTool
from app.memory.buffer import WorkingMemoryBuffer
from app.memory.models import MemoryCategory, MemoryItem, MemoryAddRequest
from app.models.schemas import (
    RoutingDecision,
    ChatRequest,
    SourceCitation,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    """FastAPI TestClient (module-scoped for performance)."""
    app = create_app()
    return TestClient(app)


@pytest.fixture()
def supervisor():
    """Fresh SupervisorAgent per test."""
    return SupervisorAgent()


@pytest.fixture()
def tmp_memory_buffer(tmp_path):
    """WorkingMemoryBuffer backed by a temp file."""
    storage = str(tmp_path / "test_chat_history.json")
    return WorkingMemoryBuffer(storage_path=storage)


# ===========================================================================
# DOMAIN 1: Supervisor Routing Contracts
# ===========================================================================

class TestSupervisorRouting:
    """TDD contracts for intent classification."""

    @pytest.mark.asyncio
    async def test_finance_routes_to_finance_agent(self, supervisor):
        decision = await supervisor.route("What is Apple stock price today?")
        assert decision.intent == "FINANCE_MARKET"
        assert "finance_agent" in decision.agents
        assert decision.requires_finance is True
        assert decision.confidence >= 0.9

    @pytest.mark.asyncio
    async def test_product_routes_to_product_agent(self, supervisor):
        decision = await supervisor.route("What is the price of iPhone 15 Pro Max?")
        assert decision.intent == "PRODUCT_SEARCH"
        assert "product_agent" in decision.agents
        assert decision.requires_product is True

    @pytest.mark.asyncio
    async def test_rag_routes_to_rag_agent(self, supervisor):
        decision = await supervisor.route("Summarize my uploaded research proposal PDF.")
        assert decision.intent == "DOCUMENT_RAG"
        assert "rag_agent" in decision.agents
        assert decision.requires_documents is True

    @pytest.mark.asyncio
    async def test_code_routes_to_code_agent(self, supervisor):
        decision = await supervisor.route("Write a Bellman-Ford algorithm in Python.")
        assert decision.intent == "CODE_EXECUTION"
        assert "code_agent" in decision.agents

    @pytest.mark.asyncio
    async def test_web_routes_to_web_agent(self, supervisor):
        decision = await supervisor.route("What is the weather today in Dhaka?")
        assert decision.intent == "WEB_SEARCH"
        assert "web_agent" in decision.agents
        assert decision.requires_live_data is True

    @pytest.mark.asyncio
    async def test_general_chat_routes_to_general_agent(self, supervisor):
        decision = await supervisor.route("Hello, who are you?")
        assert decision.intent == "GENERAL_CHAT"
        assert "general_agent" in decision.agents

    @pytest.mark.asyncio
    async def test_multi_agent_finance_and_product(self, supervisor):
        decision = await supervisor.route(
            "Compare Apple stock price with iPhone 18 Pro Max retail price"
        )
        assert decision.intent == "MULTI_AGENT"
        assert "finance_agent" in decision.agents
        assert "product_agent" in decision.agents

    @pytest.mark.asyncio
    async def test_multi_agent_code_and_rag(self, supervisor):
        decision = await supervisor.route(
            "Use my uploaded project files to write a Python class for the data structure."
        )
        assert decision.intent == "MULTI_AGENT"
        assert "rag_agent" in decision.agents
        assert "code_agent" in decision.agents
        assert decision.requires_documents is True

    @pytest.mark.asyncio
    async def test_routing_returns_routing_decision_schema(self, supervisor):
        decision = await supervisor.route("Tell me about large language models.")
        assert isinstance(decision, RoutingDecision)
        assert isinstance(decision.agents, list)
        assert 0.0 <= decision.confidence <= 1.0
        assert isinstance(decision.intent, str) and len(decision.intent) > 0

    @pytest.mark.asyncio
    async def test_routing_always_has_at_least_one_agent(self, supervisor):
        queries = [
            "Hello",
            "What is machine learning?",
            "Show me Tesla stock",
            "Summarize policy document",
        ]
        for q in queries:
            decision = await supervisor.route(q)
            assert len(decision.agents) >= 1, f"No agent assigned for: '{q}'"

    @pytest.mark.asyncio
    async def test_rag_web_combined_routes_multi_agent(self, supervisor):
        decision = await supervisor.route(
            "Compare privacy policy in my document with current web regulations."
        )
        assert decision.intent == "MULTI_AGENT"
        assert "rag_agent" in decision.agents


# ===========================================================================
# DOMAIN 2: FinanceTool Contracts
# ===========================================================================

class TestFinanceToolContracts:
    """TDD contracts for ticker resolution and quote shape."""

    def test_resolve_ticker_apple(self):
        assert FinanceTool.resolve_ticker("apple") == "AAPL"

    def test_resolve_ticker_samsung(self):
        assert FinanceTool.resolve_ticker("samsung") == "005930.KS"

    def test_resolve_ticker_microsoft(self):
        assert FinanceTool.resolve_ticker("microsoft") == "MSFT"

    def test_resolve_ticker_google(self):
        assert FinanceTool.resolve_ticker("google") == "GOOGL"

    def test_resolve_ticker_case_insensitive(self):
        assert FinanceTool.resolve_ticker("APPLE") == "AAPL"
        assert FinanceTool.resolve_ticker("Apple") == "AAPL"
        assert FinanceTool.resolve_ticker("Tesla") == "TSLA"

    def test_resolve_ticker_direct_symbol_passthrough(self):
        assert FinanceTool.resolve_ticker("NVDA") == "NVDA"
        assert FinanceTool.resolve_ticker("META") == "META"

    def test_resolve_ticker_unknown_long_name_returns_none(self):
        result = FinanceTool.resolve_ticker("ThisIsNotACompany_XYZ_123")
        assert result is None

    def test_resolve_ticker_short_alpha_uppercased(self):
        result = FinanceTool.resolve_ticker("aaaa")
        assert result == "AAAA"

    def test_get_stock_quote_shape_when_available(self):
        quote = FinanceTool.get_stock_quote("AAPL")
        if quote is not None:
            assert "symbol" in quote
            assert "price" in quote
            assert "currency" in quote
            assert "source" in quote
            assert isinstance(quote["price"], float)
            assert quote["symbol"] == "AAPL"

    @pytest.mark.asyncio
    async def test_get_quotes_for_query_returns_list(self):
        results = await FinanceTool.get_quotes_for_query("Apple stock price")
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_get_quotes_for_query_item_structure(self):
        results = await FinanceTool.get_quotes_for_query("apple stock price today")
        for item in results:
            assert "title" in item
            assert "snippet" in item
            assert item.get("source_type") == "finance"
            assert "url" in item


# ===========================================================================
# DOMAIN 3: CodeTool Contracts
# ===========================================================================

class TestCodeToolContracts:
    """TDD contracts for the safe math sandbox."""

    def test_simple_addition(self):
        r = CodeTool.evaluate_math_expression("2 + 2")
        assert r["success"] is True
        assert r["result"] == 4

    def test_multiplication(self):
        r = CodeTool.evaluate_math_expression("7 * 8")
        assert r["success"] is True
        assert r["result"] == 56

    def test_float_expression(self):
        r = CodeTool.evaluate_math_expression("3.14 * 2")
        assert r["success"] is True
        assert abs(r["result"] - 6.28) < 0.01

    def test_math_sqrt(self):
        r = CodeTool.evaluate_math_expression("sqrt(144)")
        assert r["success"] is True
        assert r["result"] == 12.0

    def test_math_pi(self):
        r = CodeTool.evaluate_math_expression("pi * 2")
        assert r["success"] is True
        assert abs(r["result"] - 6.2831) < 0.001

    def test_power_expression(self):
        r = CodeTool.evaluate_math_expression("pow(2, 10)")
        assert r["success"] is True
        assert r["result"] == 1024

    def test_dangerous_import_rejected(self):
        r = CodeTool.evaluate_math_expression("__import__('os').system('ls')")
        assert r["success"] is False

    def test_dangerous_open_rejected(self):
        r = CodeTool.evaluate_math_expression("open('/etc/passwd').read()")
        assert r["success"] is False

    def test_dangerous_exec_rejected(self):
        r = CodeTool.evaluate_math_expression("exec('import os')")
        assert r["success"] is False

    def test_malformed_expression_fails_gracefully(self):
        r = CodeTool.evaluate_math_expression("2 +* 3")
        assert r["success"] is False
        assert "error" in r

    def test_result_echoes_expression_key(self):
        r = CodeTool.evaluate_math_expression("1 + 1")
        assert "expression" in r
        assert r["expression"] == "1 + 1"

    def test_empty_string_fails_gracefully(self):
        r = CodeTool.evaluate_math_expression("")
        assert r["success"] is False


# ===========================================================================
# DOMAIN 4: WorkingMemoryBuffer Contracts
# ===========================================================================

class TestWorkingMemoryBuffer:
    """TDD contracts for per-conversation working memory."""

    def test_add_and_retrieve_turn(self, tmp_memory_buffer):
        buf = tmp_memory_buffer
        buf.add_turn("conv-001", "user", "Hello!")
        history = buf.get_history("conv-001")
        assert len(history) == 1
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "Hello!"

    def test_multiple_turns_ordered(self, tmp_memory_buffer):
        buf = tmp_memory_buffer
        buf.add_turn("conv-002", "user", "First message")
        buf.add_turn("conv-002", "assistant", "First reply")
        buf.add_turn("conv-002", "user", "Second message")
        history = buf.get_history("conv-002")
        assert history[0]["content"] == "First message"
        assert history[1]["content"] == "First reply"
        assert history[2]["content"] == "Second message"

    def test_sliding_window_at_30_messages(self, tmp_memory_buffer):
        buf = tmp_memory_buffer
        cid = "conv-window"
        for i in range(40):
            buf.add_turn(cid, "user", f"Message {i}")
        history = buf.get_history(cid)
        assert len(history) <= 30
        assert history[0]["content"] != "Message 0"

    def test_empty_conversation_returns_empty_list(self, tmp_memory_buffer):
        result = tmp_memory_buffer.get_history("nonexistent-conv")
        assert result == []

    def test_sync_history_backfills_from_client(self, tmp_memory_buffer):
        buf = tmp_memory_buffer
        cid = "conv-sync"
        turns = [
            {"role": "user", "content": "Restored turn 1"},
            {"role": "assistant", "content": "Restored reply 1"},
        ]
        buf.sync_history(cid, turns)
        history = buf.get_history(cid)
        assert len(history) == 2
        assert history[0]["content"] == "Restored turn 1"

    def test_sync_history_longer_replaces_shorter(self, tmp_memory_buffer):
        buf = tmp_memory_buffer
        cid = "conv-replace"
        buf.add_turn(cid, "user", "Old turn")
        new_turns = [{"role": "user", "content": f"New turn {i}"} for i in range(5)]
        buf.sync_history(cid, new_turns)
        history = buf.get_history(cid)
        assert len(history) == 5
        assert "New turn" in history[0]["content"]

    def test_clear_single_conversation(self, tmp_memory_buffer):
        buf = tmp_memory_buffer
        buf.add_turn("conv-A", "user", "turn A")
        buf.add_turn("conv-B", "user", "turn B")
        buf.clear("conv-A")
        assert buf.get_history("conv-A") == []
        assert len(buf.get_history("conv-B")) == 1

    def test_clear_all_conversations(self, tmp_memory_buffer):
        buf = tmp_memory_buffer
        buf.add_turn("conv-X", "user", "x")
        buf.add_turn("conv-Y", "user", "y")
        buf.clear()
        assert buf.get_history("conv-X") == []
        assert buf.get_history("conv-Y") == []

    def test_disk_persistence_survives_reload(self, tmp_path):
        storage = str(tmp_path / "persist_test.json")
        buf1 = WorkingMemoryBuffer(storage_path=storage)
        buf1.add_turn("conv-persist", "user", "Survived!")
        buf2 = WorkingMemoryBuffer(storage_path=storage)
        history = buf2.get_history("conv-persist")
        assert len(history) == 1
        assert history[0]["content"] == "Survived!"

    def test_format_history_for_prompt_with_speaker_labels(self, tmp_memory_buffer):
        buf = tmp_memory_buffer
        cid = "conv-fmt"
        buf.add_turn(cid, "user", "What is TDD?")
        buf.add_turn(cid, "assistant", "Test-Driven Development.")
        prompt_str = buf.format_history_for_prompt(cid, max_turns=5)
        assert "User:" in prompt_str
        assert "Assistant:" in prompt_str
        assert "TDD" in prompt_str

    def test_format_history_empty_returns_empty_string(self, tmp_memory_buffer):
        result = tmp_memory_buffer.format_history_for_prompt("no-such-conv", max_turns=5)
        assert result == ""

    def test_add_turn_ignores_empty_content(self, tmp_memory_buffer):
        buf = tmp_memory_buffer
        buf.add_turn("conv-empty", "user", "")
        history = buf.get_history("conv-empty")
        assert len(history) == 0

    def test_limit_turns_in_get_history(self, tmp_memory_buffer):
        buf = tmp_memory_buffer
        cid = "conv-limit"
        for i in range(10):
            buf.add_turn(cid, "user", f"u{i}")
            buf.add_turn(cid, "assistant", f"a{i}")
        result = buf.get_history(cid, limit_turns=2)
        assert len(result) <= 4


# ===========================================================================
# DOMAIN 5: Pydantic Schema Validation
# ===========================================================================

class TestSchemaValidation:
    """TDD contracts for request/response Pydantic models."""

    def test_chat_request_requires_message(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            ChatRequest(stream=False)

    def test_chat_request_empty_message_rejected(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            ChatRequest(message="")

    def test_chat_request_valid_minimal(self):
        req = ChatRequest(message="Hello")
        assert req.message == "Hello"
        assert req.stream is False
        assert req.conversation_id is None

    def test_chat_request_all_optional_fields_accepted(self):
        req = ChatRequest(
            message="What is Python?",
            stream=True,
            conversation_id="conv-abc",
            history=[{"role": "user", "content": "Hi"}]
        )
        assert req.stream is True
        assert req.conversation_id == "conv-abc"
        assert len(req.history) == 1

    def test_routing_decision_confidence_max_1(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            RoutingDecision(intent="GENERAL_CHAT", agents=["general_agent"], confidence=1.5)

    def test_routing_decision_confidence_min_0(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            RoutingDecision(intent="GENERAL_CHAT", agents=["general_agent"], confidence=-0.1)

    def test_routing_decision_valid_instance(self):
        rd = RoutingDecision(
            intent="FINANCE_MARKET",
            agents=["finance_agent"],
            confidence=0.98,
            requires_live_data=True,
            requires_finance=True,
        )
        assert rd.intent == "FINANCE_MARKET"
        assert "finance_agent" in rd.agents

    def test_source_citation_requires_document(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            SourceCitation(source_type="web")

    def test_source_citation_valid_minimal(self):
        sc = SourceCitation(document="Annual Report 2024.pdf")
        assert sc.document == "Annual Report 2024.pdf"
        assert sc.source_type == "document"

    def test_memory_item_id_starts_with_mem(self):
        item = MemoryItem(content="TDD is red-green-refactor.")
        assert item.id.startswith("mem_")

    def test_memory_item_importance_min_1(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            MemoryItem(content="test", importance=0)

    def test_memory_item_importance_max_5(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            MemoryItem(content="test", importance=6)

    def test_memory_add_request_min_length_2(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            MemoryAddRequest(content="x")

    def test_memory_category_contains_all_expected_values(self):
        categories = {c.value for c in MemoryCategory}
        for expected in ("fact", "preference", "user_profile", "project", "instruction", "general"):
            assert expected in categories


# ===========================================================================
# DOMAIN 6: API Error Contract Tests
# ===========================================================================

class TestAPIErrorContracts:
    """TDD: define what the API MUST return on bad inputs."""

    def test_post_chat_empty_message_returns_422(self, client):
        response = client.post("/chat", json={"message": "", "stream": False})
        assert response.status_code == 422

    def test_post_chat_missing_message_field_returns_422(self, client):
        response = client.post("/chat", json={"stream": False})
        assert response.status_code == 422

    def test_post_memory_missing_content_returns_422(self, client):
        response = client.post("/memory", json={"category": "fact"})
        assert response.status_code == 422

    def test_delete_nonexistent_memory_returns_404(self, client):
        response = client.delete("/memory/mem_doesnotexist_xyz")
        assert response.status_code == 404

    def test_get_document_file_nonexistent_returns_404(self, client):
        response = client.get("/documents/file/nonexistent_file_xyz.pdf")
        assert response.status_code == 404

    def test_health_endpoint_always_200(self, client):
        for _ in range(3):
            response = client.get("/health")
            assert response.status_code == 200

    def test_post_chat_valid_minimal_returns_200(self, client):
        response = client.post("/chat", json={"message": "Hi", "stream": False})
        assert response.status_code == 200

    def test_post_chat_response_has_answer_and_agent_keys(self, client):
        response = client.post("/chat", json={"message": "What is 2+2?", "stream": False})
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "agent" in data
        assert len(data["answer"]) > 0

    def test_post_memory_recall_empty_query_returns_422(self, client):
        response = client.post("/memory/recall", json={"query": "", "top_k": 5})
        assert response.status_code == 422

    def test_get_memory_returns_memories_and_total_count(self, client):
        response = client.get("/memory")
        assert response.status_code == 200
        data = response.json()
        assert "memories" in data
        assert "total_count" in data
        assert isinstance(data["memories"], list)

    def test_get_documents_returns_documents_list_and_total_count(self, client):
        response = client.get("/documents")
        assert response.status_code == 200
        data = response.json()
        assert "documents" in data
        assert "total_count" in data
        assert isinstance(data["documents"], list)


# ===========================================================================
# DOMAIN 7: FinanceTool Ticker Map Completeness
# ===========================================================================

class TestFinanceToolTickerCompleteness:
    """TDD: ensure the COMPANY_TO_TICKER mapping covers all expected companies."""

    EXPECTED = {
        "apple": "AAPL",
        "samsung": "005930.KS",
        "microsoft": "MSFT",
        "google": "GOOGL",
        "alphabet": "GOOGL",
        "amazon": "AMZN",
        "tesla": "TSLA",
        "nvidia": "NVDA",
        "meta": "META",
        "facebook": "META",
    }

    def test_all_expected_tickers_resolve_correctly(self):
        for company, expected_ticker in self.EXPECTED.items():
            result = FinanceTool.resolve_ticker(company)
            assert result == expected_ticker, (
                f"Expected '{company}' -> '{expected_ticker}', got '{result}'"
            )

    def test_ticker_map_minimum_size(self):
        assert len(FinanceTool.COMPANY_TO_TICKER) >= 10

    def test_amzn_alias(self):
        assert FinanceTool.resolve_ticker("amzn") == "AMZN"

    def test_tsla_alias(self):
        assert FinanceTool.resolve_ticker("tsla") == "TSLA"


# ===========================================================================
# DOMAIN 8: Supervisor Edge Cases
# ===========================================================================

class TestSupervisorEdgeCases:
    """TDD: supervisor must handle edge inputs gracefully."""

    @pytest.mark.asyncio
    async def test_single_word_query_no_crash(self, supervisor):
        decision = await supervisor.route("stocks")
        assert isinstance(decision, RoutingDecision)

    @pytest.mark.asyncio
    async def test_very_long_query_no_crash(self, supervisor):
        long_q = "What is " + "Python " * 200 + "?"
        decision = await supervisor.route(long_q)
        assert isinstance(decision, RoutingDecision)

    @pytest.mark.asyncio
    async def test_special_characters_no_crash(self, supervisor):
        decision = await supervisor.route("!@#$%^&*() stock price ???")
        assert isinstance(decision, RoutingDecision)

    @pytest.mark.asyncio
    async def test_nasdaq_keyword_routes_finance(self, supervisor):
        decision = await supervisor.route("What companies are listed on NASDAQ?")
        assert decision.intent == "FINANCE_MARKET"

    @pytest.mark.asyncio
    async def test_annual_leave_policy_routes_rag(self, supervisor):
        decision = await supervisor.route("What is the annual leave policy?")
        assert decision.intent == "DOCUMENT_RAG"

    @pytest.mark.asyncio
    async def test_privacy_policy_routes_rag(self, supervisor):
        decision = await supervisor.route("Explain the privacy policy in the document.")
        assert decision.intent == "DOCUMENT_RAG"

    @pytest.mark.asyncio
    async def test_python_class_routes_code(self, supervisor):
        decision = await supervisor.route("Write a Python class for a binary tree.")
        assert decision.intent in ("CODE_EXECUTION", "MULTI_AGENT")
        assert "code_agent" in decision.agents
