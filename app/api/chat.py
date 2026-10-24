import uuid
import json
import asyncio
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from app.models.schemas import ChatRequest, ChatResponse
from app.graph.workflow import workflow_engine
from app.utils.logging import logger

router = APIRouter(tags=["Chat"])


@router.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Main Chat API Endpoint with Real-Time Streaming Support.
    When stream=true (default in Web UI), streams tokens word-by-word via Server-Sent Events (SSE)
    so the user watches the response generate live in real-time.
    """
    conv_id = request.conversation_id or f"conv_{uuid.uuid4().hex[:8]}"

    if request.stream:
        async def sse_generator():
            try:
                # 1. Run multi-agent pipeline
                yield f"data: {json.dumps({'type': 'status', 'content': 'Analyzing & retrieving evidence...'})}\n\n"
                
                response = await workflow_engine.run(
                    conversation_id=conv_id,
                    query=request.message,
                    history=request.history
                )

                # 2. Emit active agent and sources metadata
                sources_data = [s.model_dump() for s in response.sources] if response.sources else []
                meta_event = {
                    "type": "metadata",
                    "agent": response.agent,
                    "sources": sources_data,
                    "latency": response.latency_seconds
                }
                yield f"data: {json.dumps(meta_event)}\n\n"

                # 3. Stream text content tokens naturally
                full_text = response.answer or ""
                words = full_text.split(" ")
                for i, word in enumerate(words):
                    token = word + (" " if i < len(words) - 1 else "")
                    yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
                    # Subtle pacing for smooth human reading animation
                    await asyncio.sleep(0.015)

                yield f"data: {json.dumps({'type': 'done'})}\n\n"
            except Exception as e:
                logger.error(f"Streaming error in /chat: {str(e)}")
                yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

        return StreamingResponse(sse_generator(), media_type="text/event-stream")

    try:
        # Non-streaming fallback
        response = await workflow_engine.run(
            conversation_id=conv_id,
            query=request.message,
            history=request.history
        )
        return response
    except Exception as e:
        logger.error(f"Chat workflow execution failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
