"""CoCo Fusion — BYO-key multi-model persona fusion council (public profile)."""

from coco_fusion.council import (
    CouncilResult,
    PanelMember,
    Persona,
    run_council,
)
from coco_fusion.lane_registry import (
    LOCAL_RESIDENT_MODEL,
    REGISTRY,
    Lane,
    build_lanes_from_config,
    default_lanes,
    distinct_architecture_lanes,
    get_lane,
    lane_usable,
    usable_lanes,
)
from coco_fusion.policy import FusionDecision, should_fuse

__all__ = [
    "Lane",
    "REGISTRY",
    "LOCAL_RESIDENT_MODEL",
    "default_lanes",
    "build_lanes_from_config",
    "distinct_architecture_lanes",
    "get_lane",
    "lane_usable",
    "usable_lanes",
    "FusionDecision",
    "should_fuse",
    "CouncilResult",
    "Persona",
    "PanelMember",
    "run_council",
]

__version__ = "0.1.0"
