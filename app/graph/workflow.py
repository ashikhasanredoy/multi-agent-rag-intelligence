import time
import asyncio
from typing import Dict, Any, List, Optional
from app.graph.state import AgentState
from app.agents.supervisor import supervisor_agent
from app.agents.finance_agent import finance_agent
from app.agents.product_agent import product_agent
from app.agents.web_agent import web_agent
from app.agents.rag_agent import rag_agent
from app.agents.code_agent import code_agent
from app.agents.general_agent import general_agent
from app.agents.critic_agent import critic_agent
from app.agents.final_agent import final_agent
from app.models.schemas import ChatResponse, SourceCitation
from app.memory.pg_store import memory_store
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

        # 1. Concurrently run Memory Recall and Supervisor Routing in parallel
        memory_task = asyncio.create_task(
            memory_store.recall_memories(query=query, top_k=5, min_score=0.30)
        )
        route_task = asyncio.create_task(
            supervisor_agent.route(query)
        )

        recalled_memories, decision = await asyncio.gather(memory_task, route_task, return_exceptions=True)

        # Handle exceptions gracefully
        if isinstance(recalled_memories, Exception) or not isinstance(recalled_memories, list):
            recalled_memories = []

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

        state["intent"] = decision.intent
        state["selected_agents"] = decision.agents
        logger.info(f"Supervisor routed intent: '{decision.intent}' to agents: {decision.agents}")

        # 4. Agent Execution Node (Concurrent Multi-Agent or Specialized Single-Agent)
        selected = decision.agents

        # Handle MULTI_AGENT and Conditional Execution with Asynchronous Coordination
        if "code_agent" in selected and "rag_agent" in selected:
            # Code modification with project document files (e.g. "Use my uploaded project files to improve this Python function")
            logger.info("Executing Code + Project Document Retrieval pipeline.")
            rag_result = await rag_agent.execute(query)
            sources = rag_result.get("sources", [])
            doc_context = rag_result.get("draft_answer", "") if sources else None
            result = await code_agent.execute(
                query=query,
                project_context=doc_context,
                sources=sources,
                memory_context=memory_context,
                dialogue_context=dialogue_context
            )
        elif len(selected) > 1:
            # Parallel execution with task tracking and conditional cancellation
            running_tasks = {}
            for agent_name in selected:
                if agent_name == "finance_agent":
                    running_tasks["finance"] = asyncio.create_task(finance_agent.execute(query))
                elif agent_name == "product_agent":
                    running_tasks["product"] = asyncio.create_task(product_agent.execute(query))
                elif agent_name == "rag_agent":
                    running_tasks["rag"] = asyncio.create_task(rag_agent.execute(query))
                elif agent_name == "web_agent":
                    running_tasks["web"] = asyncio.create_task(web_agent.execute(query))
                elif agent_name == "code_agent":
                    running_tasks["code"] = asyncio.create_task(code_agent.execute(query=query, memory_context=memory_context, dialogue_context=dialogue_context))

            # Await tasks with cancellation safety
            task_list = list(running_tasks.values())
            done_results = await asyncio.gather(*task_list, return_exceptions=True)
            
            combined_drafts = []
            all_sources = []
            for r in done_results:
                if isinstance(r, dict):
                    if r.get("draft_answer"):
                        combined_drafts.append(r["draft_answer"])
                    all_sources.extend(r.get("sources", []))

            result = {
                "draft_answer": "\n\n---\n\n".join(combined_drafts),
                "sources": all_sources,
                "agent": "multi_agent",
                "confidence": 0.95
            }
        elif "product_agent" in selected:
            result = await product_agent.execute(query)
        elif "finance_agent" in selected:
            result = await finance_agent.execute(query)
        elif "web_agent" in selected:
            result = await web_agent.execute(query)
        elif "rag_agent" in selected:
            # Start RAG. If query is broad policy, RAG handles it.
            # If RAG returns no documents, fall back to WebAgent or GeneralAgent.
            result = await rag_agent.execute(query)
            draft = result.get("draft_answer", "")
            no_info_markers = [
                "[no_relevant_docs]",
                "do not contain any relevant information",
                "do not contain information",
                "no matching documents found",
                "does not contain information"
            ]
            if any(marker in draft.lower() for marker in no_info_markers) or not result.get("sources"):
                logger.info("RAG agent found no relevant info in vector store. Falling back to WebAgent.")
                web_result = await web_agent.execute(query)
                if web_result.get("sources"):
                    result = web_result
        elif "code_agent" in selected:
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
