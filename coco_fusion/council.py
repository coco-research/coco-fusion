"""CoCo Fusion — F3 Council: persona × model orchestrator (public profile).

Two-axis fusion — persona framing × model architecture.

Flow:
  1. ``should_fuse("squad_plan", ...)`` gates (load + usable lanes).
  2. Select a panel of personas (built-in default or injectable selector).
  3. Assign each persona a lane round-robin across distinct architectures.
  4. Dispatch all personas in parallel via ``dispatch.default_dispatch``.
  5. A judge lane synthesises one grounded verdict (extract → compose → repair).

The verdict is ADVISORY — returned, never auto-executed.
"""
from __future__ import annotations

import concurrent.futures
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Callable

from coco_fusion.dispatch import default_dispatch
from coco_fusion.lane_registry import Lane, distinct_architecture_lanes, get_lane
from coco_fusion.policy import EFFORT_HIGH, FusionDecision, should_fuse

log = logging.getLogger(__name__)

MAX_PANEL_PERSONAS = 16
MAX_ROUNDS = 2
DEFAULT_PERSONAS = 8
_DISPATCH_CONCURRENCY = 4
_PERSONA_MAX_TOKENS = 1200
_EXTRACT_MAX_TOKENS = 2600
_SYNTH_MAX_TOKENS = 4000
_REPAIR_MAX_TOKENS = 4500


class CouncilError(RuntimeError):
    pass


class CouncilSelectionError(CouncilError):
    pass


class CouncilDispatchError(CouncilError):
    pass


@dataclass
class Persona:
    slug: str
    team: str
    cell: str
    display_name: str
    system_prompt: str


@dataclass
class PanelMember:
    persona: str
    team: str
    cell: str
    model_lane: str
    model_id: str
    architecture: str
    position: str
    ok: bool


@dataclass
class CouncilResult:
    verdict: str
    consensus: str
    dissent: list[str] = field(default_factory=list)
    panel: list[PanelMember] = field(default_factory=list)
    rounds_run: int = 0
    fused: bool = False
    degraded: bool = False
    reason: str = ""
    effort: str = EFFORT_HIGH
    cost_usd: float = 0.0
    latency_ms: int = 0

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "consensus": self.consensus,
            "dissent": list(self.dissent),
            "panel": [
                {
                    "persona": m.persona,
                    "team": m.team,
                    "cell": m.cell,
                    "model_lane": m.model_lane,
                    "model_id": m.model_id,
                    "architecture": m.architecture,
                    "position": m.position,
                    "ok": m.ok,
                }
                for m in self.panel
            ],
            "rounds_run": self.rounds_run,
            "fused": self.fused,
            "degraded": self.degraded,
            "reason": self.reason,
            "effort": self.effort,
            "cost_usd": round(self.cost_usd, 6),
            "latency_ms": self.latency_ms,
        }


_JSON_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _parse_json_obj(raw: str) -> dict | None:
    s = _JSON_FENCE.sub("", (raw or "").strip())
    start, end = s.find("{"), s.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(s[start : end + 1])
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def _fallback_verdict(members: list[PanelMember]) -> str:
    bits = []
    for m in members[:5]:
        first = re.split(r"(?<=[.!?])\s", m.position.strip(), maxsplit=1)[0]
        if first:
            bits.append(f"{m.persona}: {first}")
    return " | ".join(bits) if bits else "No council verdict available."


def _synthesize(
    prompt: str,
    members: list[PanelMember],
    judge: Lane,
    dispatcher: Callable[..., str],
    effort: str,
) -> dict:
    panel_text = "\n\n".join(
        f"[{m.persona} · {m.architecture}]: {m.position}" for m in members
    )
    user = f"QUESTION: {prompt}\n\nEXPERT POSITIONS:\n{panel_text}"

    extract_system = (
        "From the expert positions below, extract the UNION of EVERY DISTINCT substantive "
        "consideration, risk, mechanism, or recommendation any expert raised — one concise "
        "bullet each. Be EXHAUSTIVE: do NOT merge, summarize, or drop distinct points; if two "
        "experts raise different risks keep BOTH; capture every specialist point including the "
        "narrow ones. Output a long bulleted list only — completeness is the sole goal."
    )
    union_points = ""
    try:
        union_points = dispatcher(
            judge, extract_system, user, max_tokens=_EXTRACT_MAX_TOKENS, effort=effort
        ).strip()
    except Exception as exc:
        log.warning("council_extract_failed: %s", exc)

    if union_points:
        compose_system = (
            "You are the council synthesiser. Write ONE complete, decision-ready answer to the "
            "QUESTION, given the considerations the expert panel collectively raised. Your answer "
            "MUST substantively address EVERY listed consideration — do NOT drop any to save space, "
            "especially the narrow specialist risks (those are the panel's value). Organize by "
            "domain, then give a clear recommendation and what remains uncertain. Completeness "
            "first, then decisiveness. Prose under headings; no JSON, no meta-commentary."
        )
        compose_user = f"QUESTION: {prompt}\n\nPANEL CONSIDERATIONS (address ALL):\n{union_points}"
    else:
        compose_system = (
            "You are the council synthesiser. Write ONE complete, decision-ready answer that "
            "integrates the expert positions, PRESERVING every distinct specialist risk (do not "
            "average them away). Organize by domain; clear recommendation; what's uncertain. Prose."
        )
        compose_user = user

    answer = ""
    try:
        answer = dispatcher(
            judge, compose_system, compose_user, max_tokens=_SYNTH_MAX_TOKENS, effort=effort
        ).strip()
    except Exception as exc:
        log.warning("council_synth_compose_failed: %s", exc)
    if not answer:
        answer = _fallback_verdict(members)

    if union_points and answer and not answer.startswith(tuple(m.persona for m in members[:1])):
        try:
            gap_system = (
                "You are a coverage auditor. Given the considerations the expert panel raised and "
                "a DRAFT ANSWER, list ONLY the considerations the draft does NOT substantively "
                'address. Respond ONLY with JSON: {"missing": ["...", ...]} — empty list if none.'
            )
            gap_user = f"PANEL CONSIDERATIONS:\n{union_points}\n\nDRAFT ANSWER:\n{answer}"
            gap_raw = dispatcher(judge, gap_system, gap_user, max_tokens=1200, effort="low")
            missing = [
                str(x).strip()
                for x in (_parse_json_obj(gap_raw) or {}).get("missing", [])
                if str(x).strip()
            ]
            if missing:
                repair_system = (
                    "Revise the ANSWER so it ALSO substantively addresses EACH of the MISSING "
                    "considerations, integrating them into the relevant domain sections. Keep "
                    "everything already in the answer — only ADD coverage. Return the full revised "
                    "answer as prose: no JSON, no meta-commentary."
                )
                repair_user = (
                    "MISSING CONSIDERATIONS:\n"
                    + "\n".join(f"- {x}" for x in missing)
                    + f"\n\nANSWER:\n{answer}"
                )
                repaired = dispatcher(
                    judge, repair_system, repair_user, max_tokens=_REPAIR_MAX_TOKENS, effort=effort
                ).strip()
                if repaired:
                    answer = repaired
        except Exception as exc:
            log.info("council_synth_repair_failed: %s", exc)

    consensus, dissent = "", []
    try:
        meta_system = (
            "Read the expert positions. Identify what they AGREE on and the genuine "
            'DISAGREEMENTS. Respond ONLY with JSON: {"consensus": "<one or two sentences>", '
            '"dissent": ["<a specific disagreement and who holds it>", ...]}.'
        )
        raw = dispatcher(
            judge, meta_system, f"EXPERT POSITIONS:\n{panel_text}", max_tokens=600, effort="low"
        )
        parsed = _parse_json_obj(raw)
        if parsed:
            consensus = (parsed.get("consensus") or "").strip()
            dissent = [str(d) for d in (parsed.get("dissent") or []) if str(d).strip()]
    except Exception as exc:
        log.info("council_synth_meta_failed: %s", exc)

    return {"verdict": answer, "consensus": consensus, "dissent": dissent}


def _default_selector(prompt: str, max_personas: int, content_source: str) -> list[Persona]:
    from coco_fusion.personas import default_select_panel

    return default_select_panel(prompt, max_personas, content_source)


def _degrade(
    prompt: str,
    fuse: FusionDecision,
    dispatcher: Callable[..., str],
    *,
    reason: str,
    t0: float,
) -> CouncilResult:
    lane = None
    if fuse.lanes:
        lane = get_lane(fuse.lanes[0])
    if lane is None:
        lane = get_lane("local-resident")
    verdict = ""
    if lane is not None:
        try:
            verdict = dispatcher(
                lane,
                "You are a decisive advisor. Answer the decision concisely.",
                prompt,
                max_tokens=_SYNTH_MAX_TOKENS,
                effort=fuse.effort,
            ).strip()
        except Exception as exc:
            log.warning("council_degrade_dispatch_failed: %s", exc)
    return CouncilResult(
        verdict=verdict or "Council unavailable; no model could be reached.",
        consensus="",
        dissent=[],
        panel=[],
        rounds_run=0,
        fused=False,
        degraded=True,
        reason=reason,
        effort=fuse.effort,
        latency_ms=int((time.monotonic() - t0) * 1000),
    )


def run_council(
    prompt: str,
    *,
    content_source: str = "public",
    stakes: str = "high",
    max_personas: int = DEFAULT_PERSONAS,
    debate_rounds: int = 1,
    selector: Callable[[str, int, str], list[Persona]] | None = None,
    dispatcher: Callable[..., str] | None = None,
    fuse_decision_fn: Callable[..., FusionDecision] | None = None,
    cost_recorder: Callable[[PanelMember, str], float] | None = None,
) -> CouncilResult:
    """Run the persona×model council on *prompt*; return an ADVISORY verdict."""
    t0 = time.monotonic()
    prompt = (prompt or "").strip()
    max_personas = max(1, min(int(max_personas), MAX_PANEL_PERSONAS))
    debate_rounds = max(1, min(int(debate_rounds), MAX_ROUNDS))
    del debate_rounds  # v1: independent positions + synthesis
    selector = selector or _default_selector
    dispatcher = dispatcher or default_dispatch
    fuse_fn = fuse_decision_fn or should_fuse

    if not prompt:
        return CouncilResult(
            verdict="",
            consensus="",
            reason="empty prompt",
            degraded=True,
            latency_ms=int((time.monotonic() - t0) * 1000),
        )

    fuse = fuse_fn("squad_plan", content_source, stakes=stakes)
    if not fuse.fuse:
        return _degrade(prompt, fuse, dispatcher, reason=f"fusion gate closed: {fuse.reason}", t0=t0)

    try:
        personas = selector(prompt, max_personas, content_source)
    except Exception as exc:
        log.warning("council_selection_failed: %s", exc)
        return _degrade(prompt, fuse, dispatcher, reason=f"selection failed: {exc}", t0=t0)
    if not personas:
        return _degrade(prompt, fuse, dispatcher, reason="empty panel", t0=t0)

    pool = [ln for ln in (get_lane(lid) for lid in fuse.lanes) if ln is not None]
    arch_lanes = distinct_architecture_lanes(pool, len(pool))
    if not arch_lanes:
        return _degrade(prompt, fuse, dispatcher, reason="no distinct-architecture lanes", t0=t0)
    assignments = [(p, arch_lanes[i % len(arch_lanes)]) for i, p in enumerate(personas)]

    def _run_one(persona: Persona, lane: Lane) -> PanelMember:
        try:
            pos = dispatcher(
                lane, persona.system_prompt, prompt, max_tokens=_PERSONA_MAX_TOKENS, effort=fuse.effort
            )
            return PanelMember(
                persona.slug,
                persona.team,
                persona.cell,
                lane.lane_id,
                lane.model_id,
                lane.architecture_family,
                pos.strip(),
                True,
            )
        except Exception as exc:
            alt = next((ln for ln in arch_lanes if ln.lane_id != lane.lane_id), None)
            if alt is not None:
                try:
                    pos = dispatcher(
                        alt,
                        persona.system_prompt,
                        prompt,
                        max_tokens=_PERSONA_MAX_TOKENS,
                        effort=fuse.effort,
                    )
                    return PanelMember(
                        persona.slug,
                        persona.team,
                        persona.cell,
                        alt.lane_id,
                        alt.model_id,
                        alt.architecture_family,
                        pos.strip(),
                        True,
                    )
                except Exception:
                    pass
            log.warning("council_persona_failed slug=%s lane=%s: %s", persona.slug, lane.lane_id, exc)
            return PanelMember(
                persona.slug,
                persona.team,
                persona.cell,
                lane.lane_id,
                lane.model_id,
                lane.architecture_family,
                "",
                False,
            )

    with concurrent.futures.ThreadPoolExecutor(max_workers=_DISPATCH_CONCURRENCY) as ex:
        members = list(ex.map(lambda a: _run_one(*a), assignments))

    ok_members = [m for m in members if m.ok and m.position]
    if not ok_members:
        return _degrade(prompt, fuse, dispatcher, reason="all personas failed to respond", t0=t0)

    cost = 0.0
    if cost_recorder:
        for m in ok_members:
            try:
                cost += cost_recorder(m, content_source) or 0.0
            except Exception:
                pass

    judge = arch_lanes[0]
    synth = _synthesize(prompt, ok_members, judge, dispatcher, fuse.effort)

    return CouncilResult(
        verdict=synth["verdict"],
        consensus=synth["consensus"],
        dissent=synth["dissent"],
        panel=members,
        rounds_run=1,
        fused=True,
        degraded=False,
        reason=fuse.reason,
        effort=fuse.effort,
        cost_usd=cost,
        latency_ms=int((time.monotonic() - t0) * 1000),
    )
