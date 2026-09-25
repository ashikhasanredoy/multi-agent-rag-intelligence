import time
import asyncio
from typing import Dict, Any, List, Optional
from app.graph.state import AgentState
from app.agents.supervisor import supervisor_agent
from app.agents.web_agent import web_agent
from app.agents.rag_agent import rag_agent
from app.agents.code_agent import code_agent
from app.agents.general_agent import general_agent
from app.agents.critic_agent import critic_agent
from app.agents.final_agent import final_agent
from app.models.schemas import ChatResponse
from app.memory.store import memory_store
from app.memory.buffer import working_memory
from app.memory.extractor import memory_extractor
from app.utils.logging import logger


class MultiAgentWorkflow:
    """
    Multi-Agent LangGraph Workflow Orchestrator with Long-Term Memory & Working Dialogue Memory.
    Controls memory recall, routing, agent execution, validation, synthesis, and memory consolidation.
    """

    async def run(self, conversation_id: str, query: str) -> ChatResponse:
        start_time = time.perf_counter()
        logger.info(f"Starting Multi-Agent Workflow for query: '{query}' (Conv: {conversation_id})")

        # 1. Recall Long-Term Memories & Working Dialogue Context
        recalled_memories = await memory_store.recall_memories(query=query, top_k=5, min_score=0.30)
        memory_strings = [
            f"- [{rm.memory.category.value.upper()}] {rm.memory.content}"
            for rm in recalled_memories
        ]
        memory_context = "\n".join(memory_strings) if memory_strings else None
        dialogue_context = working_memory.format_history_for_prompt(conversation_id, max_turns=4) or None

        if memory_strings:
            logger.info(f"Recalled {len(memory_strings)} relevant long-term memories for query.")

        # 2. Initialize State
        state: AgentState = {
            "conversation_id": conversation_id,
            "user_query": query,
            "retry_count": 0,
            "sources": [],
            "agent_results": {},
            "confidence": 1.0,
            "active_agent": "general_agent",
            "relevant_memories": memory_strings,
            "working_dialogue": dialogue_context
        }

        # 3. Supervisor Node (Routing)
        decision = await supervisor_agent.route(query)
        state["intent"] = decision.intent
        state["selected_agents"] = decision.agents
        logger.info(f"Supervisor routed intent: '{decision.intent}' to agents: {decision.agents}")

        # 4. Agent Execution Node (with Memory Awareness)
        if "web_agent" in decision.agents:
            result = await web_agent.execute(query)
        elif "rag_agent" in decision.agents:
            result = await rag_agent.execute(query)
        elif "code_agent" in decision.agents:
            result = await code_agent.execute(
                query=query,
                memory_context=memory_context,
                dialogue_context=dialogue_context
            )
        else:
            result = await general_agent.execute(
                query=query,
                memory_context=memory_context,
                dialogue_context=dialogue_context
            )

        state["draft_answer"] = result.get("draft_answer", "")
        state["sources"] = result.get("sources", [])
        state["active_agent"] = result.get("agent", "general_agent")
        state["confidence"] = result.get("confidence", 1.0)

        # 5. Critic Node (Grounding & Hallucination Check)
        critique = await critic_agent.evaluate(
            query=query,
            draft_answer=state["draft_answer"],
            sources=state["sources"],
            agent_name=state["active_agent"]
        )
        state["critique"] = critique

        # If critic rejected due to cutoff excuses when sources exist, refine
        if not critique.get("valid", True) and state["sources"]:
            logger.info("Critic triggered rewrite to remove cutoff excuse.")
            lines = [line for line in state["draft_answer"].split("\n") if not any(w in line.lower() for w in ["knowledge cutoff", "december 2023"])]
            state["draft_answer"] = "\n".join(lines).strip()

        latency = round(time.perf_counter() - start_time, 3)
        state["latency_seconds"] = latency

        # 6. Final Synthesis Node
        response = await final_agent.format_response(
            conversation_id=conversation_id,
            query=query,
            draft_answer=state["draft_answer"],
            agent_name=state["active_agent"],
            sources=state["sources"],
            confidence=state["confidence"],
            latency=latency
        )

        # 7. Memory Updates:
        # A. Update Working Short-Term Dialogue Buffer
        working_memory.add_turn(conversation_id, "user", query)
        working_memory.add_turn(conversation_id, "assistant", response.answer)

        # B. Asynchronous Autonomous Long-Term Memory Extraction
        asyncio.create_task(
            memory_extractor.process_turn(
                query=query,
                assistant_response=response.answer,
                conversation_id=conversation_id
            )
        )

        return response


workflow_engine = MultiAgentWorkflow()
