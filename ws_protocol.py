"""
server/ws_protocol.py - WebSocket Streaming Protocol & Telemetry Frames
VeriPDE Specification (October 2026)
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass
class TelemetryFrame:
    """Standardized WebSocket frame streamed live to frontend mission-control deck."""
    frame_type: str  # "arena_step", "audit_progress", "hitl_state", "ablation_update"
    panel_id: Optional[str] = None  # "naive", "strong_baseline", "fourier"
    step_index: int = 0
    sim_time: float = 0.0
    field_data: Optional[List[float]] = None
    field_shape: Optional[List[int]] = None
    reference_data: Optional[List[float]] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    badge: Optional[Dict[str, Any]] = None

    def to_json(self) -> str:
        d = {
            "frame_type": self.frame_type,
            "panel_id": self.panel_id,
            "step_index": self.step_index,
            "sim_time": round(self.sim_time, 4),
            "field_data": self.field_data,
            "field_shape": self.field_shape,
            "reference_data": self.reference_data,
            "metrics": self.metrics,
            "badge": self.badge,
        }
        return json.dumps(d)
