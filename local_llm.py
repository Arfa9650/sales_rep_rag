"""
Local LLM client for the scratch agent using Ollama.
All planning, reasoning, and tool-selection decisions happen through this interface.
"""

import json
import urllib.error
import urllib.request
from typing import Any, Optional

from config import OLLAMA_BASE_URL, OLLAMA_MODEL


def _ollama_url(path: str) -> str:
    """Build full Ollama URL for a given path (e.g. /api/generate)."""
    base = OLLAMA_BASE_URL.rstrip("/")
    return f"{base}{path}"


def _post_generate(body: dict) -> dict:
    """
    Call Ollama's /api/generate endpoint with a JSON body and return the parsed JSON response.
    Uses non-streaming mode (stream: false).
    """
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        _ollama_url("/api/generate"),
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Ollama HTTP {e.code}: {err_body}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Ollama connection error: {e}") from e

    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Ollama returned non-JSON response: {raw[:200]}") from e


def complete(prompt: str, **kwargs: Any) -> str:
    """
    Send a prompt to the local LLM (Ollama) and return the raw text response.
    This is the single call site for reasoning, decision, and reflection.
    """
    options = kwargs.get("options") or {}
    temperature = kwargs.get("temperature")
    if temperature is not None:
        # do not overwrite if user already provided in options
        options.setdefault("temperature", float(temperature))

    body: dict[str, Any] = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }
    if options:
        body["options"] = options

    resp = _post_generate(body)
    # Non-streaming generate returns a single object with a `response` field
    return str(resp.get("response", ""))


def complete_structured(prompt: str, schema: Optional[dict] = None) -> dict:
    """
    Request a structured JSON response from Ollama.

    - Sends `format: \"json\"` so Ollama validates JSON.
    - Expects the model to return a JSON object in `response`.
    - If parsing fails, returns an empty dict; callers fall back to text parsing.
    """
    body: dict[str, Any] = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json",
    }
    # `schema` can be used in the prompt; we don't send it separately here
    _ = schema

    resp = _post_generate(body)
    text = str(resp.get("response", "")).strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}

