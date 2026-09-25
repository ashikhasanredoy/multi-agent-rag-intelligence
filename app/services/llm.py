import json
import httpx
from typing import List, Dict, Any, Optional, AsyncGenerator
from app.config.settings import get_settings
from app.utils.logging import logger


class LLMService:
    def __init__(self):
        self.settings = get_settings()
        self.base_url = self.settings.OLLAMA_BASE_URL.rstrip("/")
        self.default_model = self.settings.LLM_MODEL
        self.timeout = self.settings.LLM_TIMEOUT_SECONDS

    async def check_health(self) -> Dict[str, Any]:
        """Check connection to Ollama and verify model availability."""
        url = f"{self.base_url}/api/tags"
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(url)
                if response.status_code == 200:
                    data = response.json()
                    models = [m.get("name") for m in data.get("models", [])]
                    model_available = any(
                        self.default_model in m or m.startswith(self.default_model.split(":")[0])
                        for m in models
                    )
                    return {
                        "status": "connected" if model_available else "model_missing",
                        "available_models": models,
                        "default_model": self.default_model,
                        "model_ready": model_available,
                    }
                return {
                    "status": "error",
                    "status_code": response.status_code,
                    "default_model": self.default_model,
                    "model_ready": False,
                }
        except Exception as e:
            logger.warning(f"Ollama health check failed: {str(e)}")
            return {
                "status": "unreachable",
                "error": str(e),
                "default_model": self.default_model,
                "model_ready": False,
            }

    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        system: Optional[str] = None,
    ) -> str:
        """Send a chat completion request to Ollama and return the full assistant response text."""
        url = f"{self.base_url}/api/chat"
        selected_model = model or self.default_model
        temp = temperature if temperature is not None else self.settings.LLM_TEMPERATURE

        payload_messages = []
        if system:
            payload_messages.append({"role": "system", "content": system})
        payload_messages.extend(messages)

        payload = {
            "model": selected_model,
            "messages": payload_messages,
            "stream": False,
            "options": {
                "temperature": temp
            }
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
                return data.get("message", {}).get("content", "")
        except httpx.HTTPStatusError as e:
            logger.error(f"Ollama HTTP error {e.response.status_code}: {e.response.text}")
            raise RuntimeError(f"Ollama API returned status {e.response.status_code}: {e.response.text}") from e
        except Exception as e:
            logger.error(f"Error during Ollama chat request: {str(e)}")
            raise RuntimeError(f"Failed to communicate with LLM service: {str(e)}") from e

    async def stream_chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        system: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream chat completion chunks from Ollama."""
        url = f"{self.base_url}/api/chat"
        selected_model = model or self.default_model
        temp = temperature if temperature is not None else self.settings.LLM_TEMPERATURE

        payload_messages = []
        if system:
            payload_messages.append({"role": "system", "content": system})
        payload_messages.extend(messages)

        payload = {
            "model": selected_model,
            "messages": payload_messages,
            "stream": True,
            "options": {
                "temperature": temp
            }
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream("POST", url, json=payload) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        chunk = json.loads(line)
                        content = chunk.get("message", {}).get("content", "")
                        if content:
                            yield content
                        if chunk.get("done", False):
                            break
        except Exception as e:
            logger.error(f"Error streaming from Ollama: {str(e)}")
            yield f"\n[Error communicating with LLM service: {str(e)}]"


# Singleton instance
llm_service = LLMService()
