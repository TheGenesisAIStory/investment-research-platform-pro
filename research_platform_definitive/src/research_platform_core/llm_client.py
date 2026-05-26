"""Small Ollama client for local or private-cloud research assistance."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LlmResponse:
    status: str
    model: str
    content: str
    raw: dict[str, Any]
    error: str = ""


class OllamaClient:
    """HTTP client for Ollama `/api/generate` and `/api/chat`.

    The base URL can point to localhost, a VPN-only VM, or a secured container
    endpoint. Callers should keep prompts data-minimized and treat responses as
    research assistance, not investment advice.
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: int = 60,
        retries: int = 2,
        api_key: str | None = None,
    ) -> None:
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434").rstrip("/")
        self.model = model or os.getenv("OLLAMA_MODEL") or "llama3.1"
        self.timeout = int(timeout)
        self.retries = int(retries)
        self.api_key = api_key or os.getenv("OLLAMA_API_KEY", "")

    def _post(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}{endpoint}"
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        last_error: Exception | None = None
        for attempt in range(max(1, self.retries + 1)):
            try:
                request = urllib.request.Request(url, data=data, headers=headers, method="POST")
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(min(2.0, 0.25 * (attempt + 1)))
        raise RuntimeError(f"Ollama request failed: {type(last_error).__name__}: {last_error}")

    def generate_completion(
        self,
        prompt: str,
        *,
        system: str = "",
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        response_format: str | dict[str, Any] | None = None,
    ) -> LlmResponse:
        payload: dict[str, Any] = {
            "model": model or self.model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {"temperature": float(temperature)},
        }
        if max_tokens:
            payload["options"]["num_predict"] = int(max_tokens)
        if response_format is not None:
            payload["format"] = response_format
        try:
            raw = self._post("/api/generate", payload)
            return LlmResponse("OK", payload["model"], str(raw.get("response", "")).strip(), raw)
        except Exception as exc:
            return LlmResponse("UNAVAILABLE", payload["model"], "", {}, f"{type(exc).__name__}: {exc}")

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        response_format: str | dict[str, Any] | None = None,
    ) -> LlmResponse:
        payload: dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": float(temperature)},
        }
        if max_tokens:
            payload["options"]["num_predict"] = int(max_tokens)
        if response_format is not None:
            payload["format"] = response_format
        try:
            raw = self._post("/api/chat", payload)
            message = raw.get("message", {}) if isinstance(raw, dict) else {}
            return LlmResponse("OK", payload["model"], str(message.get("content", "")).strip(), raw)
        except Exception as exc:
            return LlmResponse("UNAVAILABLE", payload["model"], "", {}, f"{type(exc).__name__}: {exc}")
