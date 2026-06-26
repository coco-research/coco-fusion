"""CoCo Fusion — public lane registry.

A *lane* is one reachable way to call a model: provider endpoint, model id,
architecture family, and the env var that supplies the API key (or base URL for
local). Lanes are built from environment variables or an explicit config dict.

Tiers:
  - ``frontier`` — high/medium stakes (strongest models per family)
  - ``budget``   — low stakes only
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

TIER_FRONTIER = "frontier"
TIER_BUDGET = "budget"

# Family selection priority for distinct-architecture panels. Lower = picked first.
_FAMILY_PRIORITY: dict[str, int] = {
    "claude": 0,
    "gpt": 1,
    "gemini": 2,
    "deepseek": 3,
    "llama": 4,
    "local": 5,
}

# Default model IDs per preset (override via env).
_ENV_MODEL = {
    "anthropic": "COCO_FUSION_ANTHROPIC_MODEL",
    "openai": "COCO_FUSION_OPENAI_MODEL",
    "openrouter_claude": "COCO_FUSION_OPENROUTER_CLAUDE_MODEL",
    "openrouter_gemini": "COCO_FUSION_OPENROUTER_GEMINI_MODEL",
    "deepseek": "COCO_FUSION_DEEPSEEK_MODEL",
    "moonshot": "COCO_FUSION_MOONSHOT_MODEL",
    "local": "COCO_FUSION_LOCAL_MODEL",
}

_DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-4-20250514",
    "openai": "gpt-4o",
    "openrouter_claude": "anthropic/claude-sonnet-4",
    "openrouter_gemini": "google/gemini-2.5-pro-preview",
    "deepseek": "deepseek-chat",
    "moonshot": "moonshot-v1-128k",
    "local": "__resident__",
}


@dataclass(frozen=True)
class Lane:
    """One reachable model lane."""

    lane_id: str
    provider: str  # openai_compat | anthropic
    model_id: str
    architecture_family: str  # claude | gpt | gemini | llama | deepseek | local
    api_key_env: str
    base_url: str | None = None  # OpenAI-compatible chat completions base (…/v1)
    tier: str = TIER_FRONTIER
    rank: int = 0
    supports_effort: bool = False

    @property
    def client_hint(self) -> str:
        """Dispatch hint consumed by ``dispatch.py``."""
        return "anthropic" if self.provider == "anthropic" else "openai_compat"


LOCAL_RESIDENT_MODEL = "__resident__"


def _model(preset: str) -> str:
    env_key = _ENV_MODEL.get(preset, "")
    if env_key:
        val = os.environ.get(env_key, "").strip()
        if val:
            return val
    return _DEFAULT_MODELS[preset]


def _preset_lanes() -> list[dict[str, Any]]:
    """Built-in lane presets (activated when their api_key_env is set)."""
    return [
        {
            "lane_id": "anthropic-sonnet",
            "provider": "anthropic",
            "model_id": _model("anthropic"),
            "architecture_family": "claude",
            "api_key_env": "ANTHROPIC_API_KEY",
            "tier": TIER_FRONTIER,
            "rank": 0,
            "supports_effort": True,
        },
        {
            "lane_id": "openai-gpt",
            "provider": "openai_compat",
            "base_url": os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            "api_key_env": "OPENAI_API_KEY",
            "model_id": _model("openai"),
            "architecture_family": "gpt",
            "tier": TIER_FRONTIER,
            "rank": 0,
            "supports_effort": True,
        },
        {
            "lane_id": "openrouter-claude",
            "provider": "openai_compat",
            "base_url": os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            "api_key_env": "OPENROUTER_API_KEY",
            "model_id": _model("openrouter_claude"),
            "architecture_family": "claude",
            "tier": TIER_FRONTIER,
            "rank": 1,
        },
        {
            "lane_id": "openrouter-gemini",
            "provider": "openai_compat",
            "base_url": os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            "api_key_env": "OPENROUTER_API_KEY",
            "model_id": _model("openrouter_gemini"),
            "architecture_family": "gemini",
            "tier": TIER_FRONTIER,
            "rank": 0,
        },
        {
            "lane_id": "deepseek-chat",
            "provider": "openai_compat",
            "base_url": os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
            "api_key_env": "DEEPSEEK_API_KEY",
            "model_id": _model("deepseek"),
            "architecture_family": "deepseek",
            "tier": TIER_FRONTIER,
            "rank": 0,
        },
        {
            "lane_id": "moonshot-kimi",
            "provider": "openai_compat",
            "base_url": os.environ.get("MOONSHOT_BASE_URL", "https://api.moonshot.cn/v1"),
            "api_key_env": "MOONSHOT_API_KEY",
            "model_id": _model("moonshot"),
            "architecture_family": "gpt",
            "tier": TIER_FRONTIER,
            "rank": 2,
        },
        {
            "lane_id": "local-resident",
            "provider": "openai_compat",
            "base_url": os.environ.get("LOCAL_LLM_BASE_URL", "http://127.0.0.1:11434/v1"),
            "api_key_env": "LOCAL_LLM_BASE_URL",
            "model_id": _model("local"),
            "architecture_family": "local",
            "tier": TIER_BUDGET,
            "rank": 0,
        },
        # Budget tier extras
        {
            "lane_id": "anthropic-haiku",
            "provider": "anthropic",
            "model_id": os.environ.get("COCO_FUSION_ANTHROPIC_BUDGET_MODEL", "claude-haiku-4-5-20251001"),
            "architecture_family": "claude",
            "api_key_env": "ANTHROPIC_API_KEY",
            "tier": TIER_BUDGET,
            "rank": 0,
        },
        {
            "lane_id": "openai-mini",
            "provider": "openai_compat",
            "base_url": os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            "api_key_env": "OPENAI_API_KEY",
            "model_id": os.environ.get("COCO_FUSION_OPENAI_BUDGET_MODEL", "gpt-4o-mini"),
            "architecture_family": "gpt",
            "tier": TIER_BUDGET,
            "rank": 0,
        },
    ]


def lane_from_dict(entry: dict[str, Any]) -> Lane:
    """Build a Lane from a config dict."""
    return Lane(
        lane_id=str(entry["lane_id"]),
        provider=str(entry.get("provider", "openai_compat")),
        model_id=str(entry["model_id"]),
        architecture_family=str(entry["architecture_family"]),
        api_key_env=str(entry["api_key_env"]),
        base_url=entry.get("base_url"),
        tier=str(entry.get("tier", TIER_FRONTIER)),
        rank=int(entry.get("rank", 0)),
        supports_effort=bool(entry.get("supports_effort", False)),
    )


def build_lanes_from_config(config: list[dict[str, Any]]) -> list[Lane]:
    """Build lanes from an explicit config list."""
    return [lane_from_dict(e) for e in config]


def default_lanes() -> list[Lane]:
    """All preset lanes (usable subset filtered by ``usable_lanes``)."""
    return [lane_from_dict(e) for e in _preset_lanes()]


REGISTRY: list[Lane] = default_lanes()


def get_lane(lane_id: str, registry: list[Lane] | None = None) -> Lane | None:
    """Return the lane with this id, or None."""
    reg = registry if registry is not None else REGISTRY
    for lane in reg:
        if lane.lane_id == lane_id:
            return lane
    return None


def lane_usable(lane: Lane) -> bool:
    """True when the lane's credential env var is set (BYO-key gate)."""
    val = os.environ.get(lane.api_key_env, "").strip()
    return bool(val)


def usable_lanes(
    *,
    tier: str | None = None,
    registry: list[Lane] | None = None,
) -> list[Lane]:
    """All registry lanes with credentials configured, optionally tier-filtered."""
    reg = registry if registry is not None else REGISTRY
    out = [lane for lane in reg if lane_usable(lane)]
    if tier is not None:
        out = [lane for lane in out if lane.tier == tier]
    return out


def distinct_architecture_lanes(
    lanes: list[Lane], n: int, *, tier: str | None = None
) -> list[Lane]:
    """Pick up to *n* lanes of distinct architecture family.

    One representative per family (lowest ``rank`` in that family), families
    ordered by ``_FAMILY_PRIORITY``.
    """
    if n <= 0:
        return []
    pool = [lane for lane in lanes if tier is None or lane.tier == tier]
    by_family: dict[str, Lane] = {}
    for lane in sorted(pool, key=lambda x: x.rank):
        by_family.setdefault(lane.architecture_family, lane)
    families = sorted(by_family, key=lambda f: _FAMILY_PRIORITY.get(f, 99))
    return [by_family[f] for f in families[:n]]
