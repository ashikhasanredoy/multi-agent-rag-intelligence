import uuid
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from app.models.schemas import ChatRequest, ChatResponse
from app.graph.workflow import workflow_engine
from app.services.llm import llm_service
from app.utils.logging import logger

router = APIRouter(tags=["Chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Main Chat API Endpoint.
    Orchestrates the Multi-Agent LangGraph workflow (Supervisor -> Specialized Agent -> Critic -> Final Agent).
    """
    conv_id = request.conversation_id or f"conv_{uuid.uuid4().hex[:8]}"

    if request.stream:
        async def event_generator():
            try:
                # Streaming through LLM directly or agent pipeline
                messages = [{"role": "user", "content": request.message}]
                async for chunk in llm_service.stream_chat(messages=messages):
                    yield chunk
            except Exception as e:
                logger.error(f"Streaming error in /chat: {str(e)}")
                yield f"\n[Error: {str(e)}]"

        return StreamingResponse(event_generator(), media_type="text/plain")

    try:
        # Run Multi-Agent Orchestration Workflow
        response = await workflow_engine.run(
            conversation_id=conv_id,
            query=request.message
        )
        return response
    except Exception as e:
        logger.error(f"Chat workflow execution failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
