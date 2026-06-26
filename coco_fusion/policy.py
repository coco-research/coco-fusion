"""CoCo Fusion — when-to-fuse policy (public profile).

``should_fuse()`` is the deterministic gate that decides, per call surface,
whether fusion is worth it, which lanes to use, and at what reasoning effort.

Layers on:
  - scheduler load gate (optional, from ``~/.coco/scheduler_gate.py``)
  - usable lanes (BYO-key: env var must be set)
  - stakes → tier + effort mapping

Fail-closed: unknown surface, unreachable load gate, or too few distinct lanes
→ ``fuse=False``. Fusion is the exception; single-model is the default.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from coco_fusion.lane_registry import (
    TIER_BUDGET,
    TIER_FRONTIER,
    Lane,
    distinct_architecture_lanes,
    get_lane,
    lane_usable,
    usable_lanes,
)

log = logging.getLogger(__name__)

FUSION_NONE = "none"
FUSION_SYNTHESIS = "synthesis"
FUSION_MODEL = "model"
FUSION_COUNCIL = "council"

EFFORT_HIGH = "high"
EFFORT_MEDIUM = "medium"
EFFORT_LOW = "low"

_FUSION_SURFACES: dict[str, tuple[str, int]] = {
    "brain_ask": (FUSION_SYNTHESIS, 1),
    "merge_gate": (FUSION_MODEL, 3),
    "squad_plan": (FUSION_COUNCIL, 8),
}

_NEVER_FUSE_SURFACES: frozenset[str] = frozenset(
    {"chat_stream", "knowledge_search_stream", "fact_extract", "memory_tree_seal"}
)

_FUSE_OK_POLICIES: frozenset[str] = frozenset({"normal", "aggressive"})

MAX_PANEL = 4

_HIGH = "high"
_MED = "med"
_LOW = "low"


@dataclass
class FusionDecision:
    fuse: bool
    fusion_type: str
    lanes: list[str] = field(default_factory=list)
    judge_lane: str | None = None
    effort: str = EFFORT_MEDIUM
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "fuse": self.fuse,
            "fusion_type": self.fusion_type,
            "lanes": list(self.lanes),
            "judge_lane": self.judge_lane,
            "effort": self.effort,
            "reason": self.reason,
        }


def _stakes_tier_effort(stakes: str) -> tuple[str, str]:
    s = (stakes or "").strip().lower()
    if s == _LOW:
        return TIER_BUDGET, EFFORT_LOW
    if s == _MED or s == "medium":
        return TIER_FRONTIER, EFFORT_MEDIUM
    return TIER_FRONTIER, EFFORT_HIGH


def _default_load_policy() -> str:
    """Best-effort read of the live scheduler-gate policy."""
    try:
        import sys

        gate_dir = str(Path.home() / ".coco")
        if gate_dir not in sys.path:
            sys.path.insert(0, gate_dir)
        from scheduler_gate import current_policy  # type: ignore

        return str(current_policy())
    except Exception as exc:  # pragma: no cover
        log.info("fusion_load_gate_unreachable: %s", exc)
        return "unknown"


def _no(fusion_type: str, reason: str, effort: str = EFFORT_MEDIUM) -> FusionDecision:
    return FusionDecision(fuse=False, fusion_type=FUSION_NONE, effort=effort, reason=reason)


def _resolve_panel_override(
    panel: list[str], registry: list[Lane] | None
) -> tuple[list[Lane], str | None]:
    if len(panel) > MAX_PANEL:
        return [], f"panel has {len(panel)} lanes; max is {MAX_PANEL}"
    chosen: list[Lane] = []
    for lane_id in panel:
        lane = get_lane(lane_id, registry)
        if lane is None:
            return [], f"unknown lane_id {lane_id!r}"
        if not lane_usable(lane):
            return [], f"lane {lane_id!r} missing API key env {lane.api_key_env!r}"
        chosen.append(lane)
    return chosen, None


def should_fuse(
    surface: str,
    content_source: str = "public",
    *,
    stakes: str = "high",
    tokens_estimate: int = 0,
    panel: list[str] | None = None,
    load_policy_fn: Callable[[], str] | None = None,
    registry: list[Lane] | None = None,
) -> FusionDecision:
    """Decide whether *surface* should fuse, with which lanes, at what effort.

    ``content_source`` is accepted for API compatibility but does not narrow
    lanes in the public profile (BYO-key only).
    """
    del content_source, tokens_estimate  # public profile: no content-source gating

    surf = (surface or "").strip()
    tier, effort = _stakes_tier_effort(stakes)

    if surf in _NEVER_FUSE_SURFACES:
        return _no(FUSION_NONE, f"surface {surf!r} is a no-fuse hot path", effort)

    if surf not in _FUSION_SURFACES:
        return _no(FUSION_NONE, f"surface {surf!r} is not a fusion surface (fail-closed)", effort)

    fusion_type, target_n = _FUSION_SURFACES[surf]

    policy = (load_policy_fn or _default_load_policy)()
    if policy not in _FUSE_OK_POLICIES:
        return _no(fusion_type, f"load gate policy={policy!r}: refusing to fuse", effort)

    if panel is not None:
        chosen, err = _resolve_panel_override(panel, registry)
        if err:
            return _no(fusion_type, f"panel rejected: {err}", effort)
        if fusion_type == FUSION_MODEL and len(chosen) < 2:
            return _no(fusion_type, "user panel has <2 lanes; cannot model-fuse", effort)
        lane_ids = [lane.lane_id for lane in chosen]
        return FusionDecision(
            fuse=True,
            fusion_type=fusion_type,
            lanes=lane_ids,
            judge_lane=lane_ids[0],
            effort=effort,
            reason=f"user panel ({len(lane_ids)} lanes): {', '.join(lane_ids)}",
        )

    available = usable_lanes(tier=tier, registry=registry)
    if not available:
        return _no(fusion_type, f"no usable {tier} lanes (set API key env vars)", effort)

    chosen = distinct_architecture_lanes(available, target_n, tier=tier)

    if fusion_type == FUSION_SYNTHESIS:
        if not chosen:
            return _no(fusion_type, "no usable judge lane for synthesis", effort)
        judge = chosen[0]
        return FusionDecision(
            fuse=True,
            fusion_type=FUSION_SYNTHESIS,
            lanes=[judge.lane_id],
            judge_lane=judge.lane_id,
            effort=effort,
            reason=f"synthesis judge={judge.lane_id} ({judge.architecture_family}, {tier})",
        )

    if fusion_type == FUSION_MODEL:
        if len(chosen) < 2:
            return _no(
                fusion_type,
                f"only {len(chosen)} distinct-architecture {tier} lane(s); need ≥2",
                effort,
            )
        lane_ids = [lane.lane_id for lane in chosen]
        return FusionDecision(
            fuse=True,
            fusion_type=FUSION_MODEL,
            lanes=lane_ids,
            judge_lane=chosen[0].lane_id,
            effort=effort,
            reason=(
                f"{len(lane_ids)} distinct-architecture {tier} lanes: "
                + ", ".join(f"{lane.lane_id}({lane.architecture_family})" for lane in chosen)
            ),
        )

    council_lanes = [lane.lane_id for lane in available]
    families = sorted({lane.architecture_family for lane in available})
    if len(families) < 2:
        return _no(fusion_type, f"council needs ≥2 architectures; have {families}", effort)
    return FusionDecision(
        fuse=True,
        fusion_type=FUSION_COUNCIL,
        lanes=council_lanes,
        judge_lane=None,
        effort=effort,
        reason=(
            f"council pool: {len(council_lanes)} {tier} lanes "
            f"across {len(families)} architectures {families}"
        ),
    )
