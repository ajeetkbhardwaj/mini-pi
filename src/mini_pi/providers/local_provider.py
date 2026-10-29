from __future__ import annotations

import os
from typing import Any, AsyncIterator

import httpx

from mini_pi.agent.types import Message
from mini_pi.providers.base import (
    CompletionRequest,
    CompletionResponse,
    ModelClient,
    StreamChunk,
)
from mini_pi.providers.openai_provider import OpenAIClient

LOCAL_PRESETS = {
    "lmstudio": {
        "base_url": "http://localhost:1234/v1",
        "default_model": "local-model",
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "default_model": "qwen2.5-coder",
    },
    "llamacpp": {
        "base_url": "http://localhost:8080/v1",
        "default_model": "default",
    },
}


class LocalOpenAIClient(ModelClient):
    """Client for local OpenAI-compatible backends like LMStudio, llama.cpp, and Ollama."""

    def __init__(
        self,
        base_url: str | None = None,
        model_id: str | None = None,
        api_key: str = "local-key",
        backend: str = "lmstudio",
    ) -> None:
        self._preset = LOCAL_PRESETS.get(backend.lower(), LOCAL_PRESETS["lmstudio"])
        self._base_url = (
            base_url
            or os.environ.get("LOCAL_OPENAI_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or self._preset["base_url"]
        )
        self._model_id = (
            model_id
            or os.environ.get("LOCAL_OPENAI_MODEL")
            or self._preset["default_model"]
        )
        self._api_key = api_key
        self._backend = backend

        self._rebuild_client()

    def _rebuild_client(self) -> None:
        self._inner_client = OpenAIClient(
            model_id=self._model_id,
            api_key=self._api_key,
            base_url=self._base_url,
        )

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def backend(self) -> str:
        return self._backend

    def set_model(self, model_id: str) -> None:
        self._model_id = model_id
        self._rebuild_client()

    def set_base_url(self, base_url: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._rebuild_client()

    def switch_backend(self, backend: str, custom_url: str | None = None) -> None:
        self._backend = backend.lower()
        if self._backend in LOCAL_PRESETS:
            preset = LOCAL_PRESETS[self._backend]
            self._base_url = custom_url or preset["base_url"]
            self._model_id = preset["default_model"]
        elif custom_url:
            self._base_url = custom_url
        self._rebuild_client()

    async def list_available_models(self) -> list[str]:
        """Fetch list of available models from the local server's /models endpoint."""
        url = f"{self._base_url.rstrip('/')}/models"
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(url, headers={"Authorization": f"Bearer {self._api_key}"})
            if resp.status_code != 200:
                raise ConnectionError(f"HTTP {resp.status_code} from {url}")
            data = resp.json()

            models: list[str] = []
            if "data" in data and isinstance(data["data"], list):
                for item in data["data"]:
                    if isinstance(item, dict) and "id" in item:
                        models.append(item["id"])
            elif "models" in data and isinstance(data["models"], list):
                for item in data["models"]:
                    if isinstance(item, dict) and "name" in item:
                        models.append(item["name"])
            return models

    async def test_connection(self) -> tuple[bool, str, list[str]]:
        """Test connection to the local backend and retrieve available models."""
        try:
            models = await self.list_available_models()
            return True, f"Connected to {self._backend.upper()} at {self._base_url}", models
        except Exception as e:
            return False, f"Connection to {self._base_url} failed: {e}", []

    async def stream_complete(
        self, request: CompletionRequest
    ) -> AsyncIterator[StreamChunk]:
        try:
            async for chunk in self._inner_client.stream_complete(request):
                yield chunk
        except Exception as e:
            err_msg = str(e)
            if "Connection refused" in err_msg or "ConnectError" in err_msg or "Failed to connect" in err_msg:
                raise ConnectionError(
                    f"Unable to connect to local model backend at '{self._base_url}'.\n"
                    f"Make sure your local server ({self._backend.upper()}, LM Studio, Ollama, or llama.cpp) "
                    "is running and has the OpenAI-compatible local server enabled."
                ) from e
            raise

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        try:
            return await self._inner_client.complete(request)
        except Exception as e:
            err_msg = str(e)
            if "Connection refused" in err_msg or "ConnectError" in err_msg or "Failed to connect" in err_msg:
                raise ConnectionError(
                    f"Unable to connect to local model backend at '{self._base_url}'.\n"
                    f"Make sure your local server ({self._backend.upper()}, LM Studio, Ollama, or llama.cpp) "
                    "is running and has the OpenAI-compatible local server enabled."
                ) from e
            raise

    @classmethod
    async def probe_local_servers() -> str | None:
        """Probe common ports to detect if a local server is running."""
        candidates = [
            ("lmstudio", "http://localhost:1234/v1/models"),
            ("ollama", "http://localhost:11434/v1/models"),
            ("llamacpp", "http://localhost:8080/v1/models"),
        ]
        async with httpx.AsyncClient(timeout=0.8) as client:
            for name, url in candidates:
                try:
                    res = await client.get(url)
                    if res.status_code == 200:
                        return name
                except Exception:
                    pass
        return None
