import pytest
import asyncio
from app.agents.supervisor import supervisor_agent
from app.agents.code_agent import code_agent
from app.rag.hybrid import hybrid_search
from app.tools.finance_tool import finance_tool


@pytest.mark.asyncio
async def test_scenario_a_annual_leave_policy():
    """Scenario A: Internal policy question -> DOCUMENT_RAG (rag_agent), web & code skipped."""
    decision = await supervisor_agent.route("What is our company's annual leave policy?")
    assert decision.intent == "DOCUMENT_RAG"
    assert "rag_agent" in decision.agents
    assert "code_agent" not in decision.agents
    assert "web_agent" not in decision.agents


@pytest.mark.asyncio
async def test_scenario_b_cpp_file_loading():
    """Scenario B: General C++ code generation -> CODE_EXECUTION (code_agent), RAG skipped."""
    decision = await supervisor_agent.route("Give me C++ code for file loading.")
    assert decision.intent == "CODE_EXECUTION"
    assert "code_agent" in decision.agents
    assert "rag_agent" not in decision.agents
    assert "web_agent" not in decision.agents


@pytest.mark.asyncio
async def test_scenario_c_bellman_ford():
    """Scenario C: Algorithm implementation -> CODE_EXECUTION (code_agent)."""
    decision = await supervisor_agent.route("Give me C++ code for the Bellman-Ford algorithm.")
    assert decision.intent == "CODE_EXECUTION"
    assert "code_agent" in decision.agents


@pytest.mark.asyncio
async def test_scenario_d_current_samsung_device():
    """Scenario D: Current Samsung device -> PRODUCT_SEARCH (product_agent) with web search."""
    decision = await supervisor_agent.route("What is Samsung's latest mobile phone?")
    assert decision.intent == "PRODUCT_SEARCH"
    assert "product_agent" in decision.agents
    assert "finance_agent" not in decision.agents


@pytest.mark.asyncio
async def test_scenario_e_current_product_price():
    """Scenario E: Current product price -> PRODUCT_SEARCH without stock API confusion."""
    decision = await supervisor_agent.route("What is the current price of the iPhone 18 Pro Max?")
    assert decision.intent == "PRODUCT_SEARCH"
    assert "product_agent" in decision.agents
    assert "finance_agent" not in decision.agents


@pytest.mark.asyncio
async def test_scenario_f_polymorphism_in_cpp():
    """Scenario F: Explain polymorphism in C++ -> CODE_EXECUTION / GENERAL_CHAT without blocking."""
    decision = await supervisor_agent.route("Explain polymorphism in C++.")
    assert decision.intent in ["CODE_EXECUTION", "GENERAL_CHAT"]
    assert "rag_agent" not in decision.agents


@pytest.mark.asyncio
async def test_scenario_g_mixed_privacy_policy_and_regulations():
    """Scenario G: Compare internal privacy policy with current privacy regulations -> MULTI_AGENT."""
    decision = await supervisor_agent.route("Compare our internal privacy policy with current privacy regulations.")
    assert decision.intent == "MULTI_AGENT"
    assert "rag_agent" in decision.agents
    assert "web_agent" in decision.agents


@pytest.mark.asyncio
async def test_scenario_h_project_specific_code_modification():
    """Scenario H: Project-specific code modification -> MULTI_AGENT (rag_agent + code_agent)."""
    decision = await supervisor_agent.route("Use my uploaded project files to improve this Python function.")
    assert decision.intent == "MULTI_AGENT"
    assert "rag_agent" in decision.agents
    assert "code_agent" in decision.agents
    assert "web_agent" not in decision.agents


@pytest.mark.asyncio
async def test_scenario_j_conditional_cancellation_logic():
    """Scenario J: Task cancellation logic - simulates cancelling web task if RAG is sufficient."""
    web_cancelled = False
    rag_done = False

    async def simulated_rag():
        nonlocal rag_done
        await asyncio.sleep(0.01)
        rag_done = True
        return {"status": "sufficient", "answer": "Policy says 25 days annual leave."}

    async def simulated_web():
        try:
            await asyncio.sleep(5.0)
            return {"status": "web_data"}
        except asyncio.CancelledError:
            nonlocal web_cancelled
            web_cancelled = True
            raise

    web_task = asyncio.create_task(simulated_web())
    rag_task = asyncio.create_task(simulated_rag())

    # RAG completes first with sufficient policy evidence
    rag_res = await rag_task
    assert rag_res["status"] == "sufficient"
    
    # Conditional cancellation: cancel web task because internal policy is sufficient
    web_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await web_task
    assert web_cancelled is True


@pytest.mark.asyncio
async def test_scenario_k_tool_failure_handling():
    """Scenario K: Simulated web failure while RAG succeeds."""
    async def failing_web():
        raise ConnectionError("DuckDuckGo timed out")

    async def succeeding_rag():
        return {"draft_answer": "Company policy allows 15 sick leaves.", "sources": []}

    rag_task = asyncio.create_task(succeeding_rag())
    web_task = asyncio.create_task(failing_web())

    results = await asyncio.gather(rag_task, web_task, return_exceptions=True)
    assert isinstance(results[0], dict)
    assert isinstance(results[1], ConnectionError)
    assert "Company policy" in results[0]["draft_answer"]
