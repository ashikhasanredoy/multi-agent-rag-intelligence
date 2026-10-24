import pytest
import asyncio
from app.agents.supervisor import supervisor_agent
from app.agents.finance_agent import finance_agent
from app.agents.product_agent import product_agent
from app.rag.hybrid import hybrid_search
from app.tools.finance_tool import finance_tool


@pytest.mark.asyncio
async def test_routing_apple_stock_price():
    decision = await supervisor_agent.route("What is Apple's stock price?")
    assert decision.intent == "FINANCE_MARKET"
    assert "finance_agent" in decision.agents


@pytest.mark.asyncio
async def test_routing_samsung_stock_price():
    decision = await supervisor_agent.route("What is Samsung's stock price?")
    assert decision.intent == "FINANCE_MARKET"
    assert "finance_agent" in decision.agents


@pytest.mark.asyncio
async def test_routing_iphone_18_pro_max():
    decision = await supervisor_agent.route("What is the current price of iPhone 18 Pro Max?")
    assert decision.intent == "PRODUCT_SEARCH"
    assert "product_agent" in decision.agents
    assert "finance_agent" not in decision.agents


@pytest.mark.asyncio
async def test_routing_iphone_duo():
    decision = await supervisor_agent.route("What is the price of iPhone Duo?")
    assert decision.intent == "PRODUCT_SEARCH"
    assert "product_agent" in decision.agents


@pytest.mark.asyncio
async def test_routing_compare_apple_and_samsung_stocks():
    decision = await supervisor_agent.route("Compare Apple and Samsung stock prices.")
    assert decision.intent == "FINANCE_MARKET"
    assert "finance_agent" in decision.agents


@pytest.mark.asyncio
async def test_routing_compare_iphone_and_samsung_smartphone_prices():
    decision = await supervisor_agent.route("Compare iPhone and Samsung smartphone prices.")
    assert decision.intent == "PRODUCT_SEARCH"
    assert "product_agent" in decision.agents


@pytest.mark.asyncio
async def test_routing_explain_research_proposal():
    decision = await supervisor_agent.route("Explain my uploaded research proposal.")
    assert decision.intent == "DOCUMENT_RAG"
    assert "rag_agent" in decision.agents


@pytest.mark.asyncio
async def test_routing_hypotheses_in_research_proposal():
    decision = await supervisor_agent.route("What are the hypotheses in the research proposal?")
    assert decision.intent == "DOCUMENT_RAG"
    assert "rag_agent" in decision.agents


@pytest.mark.asyncio
async def test_routing_experimental_results_proposal():
    decision = await supervisor_agent.route("What experimental results does the research proposal report?")
    assert decision.intent == "DOCUMENT_RAG"
    assert "rag_agent" in decision.agents


@pytest.mark.asyncio
async def test_routing_annual_leave_policy():
    decision = await supervisor_agent.route("What is the company's annual leave policy?")
    assert decision.intent == "DOCUMENT_RAG"
    assert "rag_agent" in decision.agents


@pytest.mark.asyncio
async def test_routing_summarize_privacy_policy():
    decision = await supervisor_agent.route("Summarize the uploaded privacy policy.")
    assert decision.intent == "DOCUMENT_RAG"
    assert "rag_agent" in decision.agents


@pytest.mark.asyncio
async def test_routing_latest_ai_news():
    decision = await supervisor_agent.route("What is the latest news about artificial intelligence?")
    assert decision.intent == "WEB_SEARCH"
    assert "web_agent" in decision.agents


@pytest.mark.asyncio
async def test_routing_compare_document_with_web():
    decision = await supervisor_agent.route("Compare information in my uploaded document with current web information.")
    assert decision.intent == "MULTI_AGENT"
    assert "rag_agent" in decision.agents
    assert "web_agent" in decision.agents


@pytest.mark.asyncio
async def test_routing_weather_today():
    decision = await supervisor_agent.route("What is the weather today?")
    assert decision.intent == "WEB_SEARCH"
    assert "web_agent" in decision.agents


@pytest.mark.asyncio
async def test_routing_general_chat_rag_definition():
    decision = await supervisor_agent.route("Explain what Retrieval-Augmented Generation means.")
    assert decision.intent == "GENERAL_CHAT"
    assert "general_agent" in decision.agents


@pytest.mark.asyncio
async def test_rejection_of_unrelated_documents():
    # An unrelated query should NOT match our PDF documents
    results = await hybrid_search.search("What is the current price of the iPhone 18 Pro Max and iPhone Duo?", top_k=4)
    assert len(results) == 0, f"Expected 0 results for unrelated query, got {len(results)}"


@pytest.mark.asyncio
async def test_finance_tool_ticker_resolution():
    assert finance_tool.resolve_ticker("apple") == "AAPL"
    assert finance_tool.resolve_ticker("samsung") == "005930.KS"
    assert finance_tool.resolve_ticker("microsoft") == "MSFT"
    assert finance_tool.resolve_ticker("iphone 18") is None
