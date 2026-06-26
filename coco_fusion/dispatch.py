"""CoCo Fusion — public model dispatch (httpx only).

Supports:
  - OpenAI-compatible chat completions (OpenRouter, OpenAI, DeepSeek, Kimi, Ollama/LM Studio)
  - Anthropic Messages API (direct)
"""
from __future__ import annotations

import os
from typing import Any

import httpx

from coco_fusion.lane_registry import LOCAL_RESIDENT_MODEL, Lane

DEFAULT_TIMEOUT_S = 120.0


class DispatchError(RuntimeError):
    pass


def _api_key(lane: Lane) -> str:
    return os.environ.get(lane.api_key_env, "").strip()


def _chat_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def _resolve_model(lane: Lane) -> str:
    if lane.model_id == LOCAL_RESIDENT_MODEL:
        # Ask the local server which model is loaded (first in /models).
        base = (lane.base_url or "").rstrip("/")
        models_url = f"{base}/models" if base.endswith("/v1") else f"{base}/v1/models"
        key = _api_key(lane)
        headers: dict[str, str] = {}
        if key and not key.startswith("http"):
            headers["Authorization"] = f"Bearer {key}"
        r = httpx.get(models_url, headers=headers, timeout=10.0)
        r.raise_for_status()
        data = r.json()
        models = data.get("data") or []
        if models and isinstance(models[0], dict):
            mid = models[0].get("id")
            if mid:
                return str(mid)
        raise DispatchError("local server returned no resident model")
    return lane.model_id


def _openai_reasoning_effort(effort: str) -> str | None:
    e = (effort or "").strip().lower()
    if e in ("high", "medium", "low"):
        return e
    return None


def dispatch_openai_compat(
    lane: Lane,
    system: str,
    prompt: str,
    *,
    max_tokens: int,
    effort: str = "medium",
    timeout: float = DEFAULT_TIMEOUT_S,
) -> str:
    """POST to an OpenAI-compatible ``/v1/chat/completions`` endpoint."""
    if not lane.base_url:
        raise DispatchError(f"lane {lane.lane_id} has no base_url")
    key = _api_key(lane)
    if not key:
        raise DispatchError(f"lane {lane.lane_id}: env {lane.api_key_env!r} not set")

    model = _resolve_model(lane)
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.2,
        "stream": False,
    }
    if lane.supports_effort:
        reff = _openai_reasoning_effort(effort)
        if reff:
            payload["reasoning_effort"] = reff

    headers: dict[str, str] = {"Content-Type": "application/json"}
    if not key.startswith("http"):
        headers["Authorization"] = f"Bearer {key}"

    r = httpx.post(
        _chat_url(lane.base_url),
        json=payload,
        headers=headers,
        timeout=timeout,
    )
    if r.status_code != 200:
        raise DispatchError(f"{lane.lane_id} HTTP {r.status_code}: {r.text[:300]}")
    data = r.json()
    msg = (data.get("choices") or [{}])[0].get("message") or {}
    # Reasoning models may put chain-of-thought in reasoning_content.
    text = (msg.get("content") or msg.get("reasoning_content") or "").strip()
    if not text:
        raise DispatchError(f"{lane.lane_id} returned empty content")
    return text


def dispatch_anthropic(
    lane: Lane,
    system: str,
    prompt: str,
    *,
    max_tokens: int,
    effort: str = "medium",
    timeout: float = DEFAULT_TIMEOUT_S,
) -> str:
    """POST to the Anthropic Messages API."""
    key = _api_key(lane)
    if not key:
        raise DispatchError(f"lane {lane.lane_id}: env {lane.api_key_env!r} not set")

    payload: dict[str, Any] = {
        "model": lane.model_id,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        payload["system"] = system

    # Map effort to thinking budget when supported (best-effort).
    if lane.supports_effort and effort == "high":
        payload["thinking"] = {"type": "enabled", "budget_tokens": min(8000, max_tokens)}

    base = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com").rstrip("/")
    r = httpx.post(
        f"{base}/v1/messages",
        json=payload,
        headers={
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        timeout=timeout,
    )
    if r.status_code != 200:
        raise DispatchError(f"{lane.lane_id} HTTP {r.status_code}: {r.text[:300]}")
    data = r.json()
    blocks = data.get("content") or []
    parts: list[str] = []
    for block in blocks:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    text = "".join(parts).strip()
    if not text:
        raise DispatchError(f"{lane.lane_id} returned empty content")
    return text


def default_dispatch(
    lane: Lane,
    system: str,
    prompt: str,
    *,
    max_tokens: int,
    effort: str = "medium",
) -> str:
    """Dispatch one call on *lane*; return response text."""
    if lane.provider == "anthropic":
        return dispatch_anthropic(lane, system, prompt, max_tokens=max_tokens, effort=effort)
    return dispatch_openai_compat(lane, system, prompt, max_tokens=max_tokens, effort=effort)
