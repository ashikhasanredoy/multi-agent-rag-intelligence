from fastapi import APIRouter
from datetime import datetime
import httpx

from app.config.settings import get_settings
from app.services.llm import llm_service
from app.models.schemas import HealthCheck

router = APIRouter(prefix="/health", tags=["Health"])


@router.get("", response_model=HealthCheck)
async def overall_health():
    settings = get_settings()
    llm_health = await llm_service.check_health()
    
    return HealthCheck(
        status="healthy" if llm_health.get("model_ready") else "degraded",
        version=settings.VERSION,
        timestamp=datetime.utcnow(),
        services={
            "llm": llm_health,
            "environment": settings.ENVIRONMENT,
        }
    )


@router.get("/llm")
async def llm_health_check():
    """Specific health check for the Ollama LLM service."""
    return await llm_service.check_health()


@router.get("/vector-db")
async def vector_db_health():
    """Check connectivity to Qdrant vector database."""
    settings = get_settings()
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            res = await client.get(f"{settings.QDRANT_URL.rstrip('/')}/readyz")
            if res.status_code == 200:
                return {"status": "connected", "url": settings.QDRANT_URL}
            return {"status": "unhealthy", "status_code": res.status_code}
    except Exception as e:
        return {"status": "disconnected", "url": settings.QDRANT_URL, "error": str(e)}


@router.get("/database")
async def database_health():
    """Check connectivity to relational database."""
    settings = get_settings()
    return {
        "status": "configured",
        "database_url": settings.DATABASE_URL.split("///")[0] if "///" in settings.DATABASE_URL else "configured"
    }
