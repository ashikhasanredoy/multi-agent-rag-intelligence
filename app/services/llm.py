import json
import httpx
from typing import List, Dict, Any, Optional, AsyncGenerator
from app.config.settings import get_settings
from app.utils.logging import logger


class LLMService:
    def __init__(self):
        self.settings = get_settings()
        self.provider = getattr(self.settings, "LLM_PROVIDER", "openrouter").lower()
        self.ollama_base_url = self.settings.OLLAMA_BASE_URL.rstrip("/")
        self.openrouter_base_url = getattr(self.settings, "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
        self.openrouter_api_key = getattr(self.settings, "OPENROUTER_API_KEY", None)
        self.default_model = self.settings.LLM_MODEL
        self.timeout = self.settings.LLM_TIMEOUT_SECONDS

    async def check_health(self) -> Dict[str, Any]:
        """Check connection to configured LLM provider and verify model availability."""
        if self.provider == "openrouter":
            url = f"{self.openrouter_base_url}/auth/key"
            headers = {
                "Authorization": f"Bearer {self.openrouter_api_key}",
            }
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get(url, headers=headers)
                    if response.status_code == 200:
                        key_info = response.json().get("data", {})
                        return {
                            "status": "connected",
                            "provider": "openrouter",
                            "default_model": self.default_model,
                            "model_ready": True,
                            "key_label": key_info.get("label"),
                            "free_model_daily_requests": key_info.get("free_model_daily_requests", {}),
                        }
                    return {
                        "status": "error",
                        "provider": "openrouter",
                        "status_code": response.status_code,
                        "default_model": self.default_model,
                        "model_ready": False,
                    }
            except Exception as e:
                logger.warning(f"OpenRouter health check failed: {str(e)}")
                return {
                    "status": "unreachable",
                    "provider": "openrouter",
                    "error": str(e),
                    "default_model": self.default_model,
                    "model_ready": False,
                }
        else:
            url = f"{self.ollama_base_url}/api/tags"
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
                            "provider": "ollama",
                            "available_models": models,
                            "default_model": self.default_model,
                            "model_ready": model_available,
                        }
                    return {
                        "status": "error",
                        "provider": "ollama",
                        "status_code": response.status_code,
                        "default_model": self.default_model,
                        "model_ready": False,
                    }
            except Exception as e:
                logger.warning(f"Ollama health check failed: {str(e)}")
                return {
                    "status": "unreachable",
                    "provider": "ollama",
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
        """Send a chat completion request to the active LLM provider (OpenRouter or Ollama)."""
        selected_model = model or self.default_model
        temp = temperature if temperature is not None else self.settings.LLM_TEMPERATURE

        payload_messages = []
        if system:
            payload_messages.append({"role": "system", "content": system})
        payload_messages.extend(messages)

        timeout_sec = self.settings.LLM_TIMEOUT_SECONDS
        timeout_config = httpx.Timeout(timeout_sec, connect=15.0, read=timeout_sec, write=15.0)

        if self.provider == "openrouter":
            url = f"{self.openrouter_base_url}/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.openrouter_api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost:8000",
                "X-Title": "Multi-Agent RAG Intelligence",
            }
            payload = {
                "model": selected_model,
                "messages": payload_messages,
                "temperature": temp,
            }
            try:
                async with httpx.AsyncClient(timeout=timeout_config) as client:
                    response = await client.post(url, json=payload, headers=headers)
                    response.raise_for_status()
                    data = response.json()
                    choices = data.get("choices", [])
                    if choices:
                        return choices[0].get("message", {}).get("content", "")
                    return ""
            except httpx.HTTPStatusError as e:
                logger.error(f"OpenRouter HTTP error {e.response.status_code}: {e.response.text}")
                raise RuntimeError(f"OpenRouter API returned status {e.response.status_code}: {e.response.text}") from e
            except Exception as e:
                err_detail = f"{type(e).__name__}: {str(e)}" if str(e) else type(e).__name__
                logger.error(f"Error during OpenRouter chat request: {err_detail}")
                raise RuntimeError(f"Failed to communicate with OpenRouter LLM: {err_detail}") from e
        else:
            url = f"{self.ollama_base_url}/api/chat"
            payload = {
                "model": selected_model,
                "messages": payload_messages,
                "stream": False,
                "options": {
                    "temperature": temp,
                    "num_predict": 768,
                    "num_ctx": 4096,
                    "top_p": 0.9,
                },
            }
            try:
                async with httpx.AsyncClient(timeout=timeout_config) as client:
                    response = await client.post(url, json=payload)
                    response.raise_for_status()
                    data = response.json()
                    return data.get("message", {}).get("content", "")
            except httpx.HTTPStatusError as e:
                logger.error(f"Ollama HTTP error {e.response.status_code}: {e.response.text}")
                raise RuntimeError(f"Ollama API returned status {e.response.status_code}: {e.response.text}") from e
            except Exception as e:
                err_detail = f"{type(e).__name__}: {str(e)}" if str(e) else type(e).__name__
                logger.error(f"Error during Ollama chat request: {err_detail}")
                raise RuntimeError(f"Failed to communicate with LLM service: {err_detail}") from e

    async def stream_chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        system: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream chat completion chunks from the active LLM provider."""
        selected_model = model or self.default_model
        temp = temperature if temperature is not None else self.settings.LLM_TEMPERATURE

        payload_messages = []
        if system:
            payload_messages.append({"role": "system", "content": system})
        payload_messages.extend(messages)

        if self.provider == "openrouter":
            url = f"{self.openrouter_base_url}/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.openrouter_api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost:8000",
                "X-Title": "Multi-Agent RAG Intelligence",
            }
            payload = {
                "model": selected_model,
                "messages": payload_messages,
                "stream": True,
                "temperature": temp,
            }
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    async with client.stream("POST", url, json=payload, headers=headers) as response:
                        response.raise_for_status()
                        async for line in response.aiter_lines():
                            if not line or not line.startswith("data: "):
                                continue
                            data_str = line[6:].strip()
                            if data_str == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data_str)
                                choices = chunk.get("choices", [])
                                if choices:
                                    delta = choices[0].get("delta", {})
                                    content = delta.get("content", "")
                                    if content:
                                        yield content
                            except json.JSONDecodeError:
                                continue
            except Exception as e:
                logger.error(f"Error streaming from OpenRouter: {str(e)}")
                yield f"\n[Error communicating with OpenRouter service: {str(e)}]"
        else:
            url = f"{self.ollama_base_url}/api/chat"
            payload = {
                "model": selected_model,
                "messages": payload_messages,
                "stream": True,
                "options": {
                    "temperature": temp,
                },
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
