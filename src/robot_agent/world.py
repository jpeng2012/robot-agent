"""Symbolic world state shared by every env backend, the skills, and the agent loop."""

from enum import StrEnum
import math
from typing import Any

from pydantic import BaseModel

# Reachable tabletop region (x_min, x_max, y_min, y_max), metres.
WORKSPACE = (-0.5, 0.5, -0.5, 0.5)


class Pose(BaseModel):
    x: float
    y: float
    z: float = 0.0

    def dist_xy(self, x: float, y: float) -> float:
        return math.hypot(self.x - x, self.y - y)

    def dist_xyz(self, other: "Pose") -> float:
        """Euclidean distance in 3D."""
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2)


class ObjectObs(BaseModel):
    id: str
    kind: str
    pose: Pose
    visible: bool = True
    confidence: float = 1.0
    observed_at: float = 0.0  # env time of the observation that produced this estimate


class GripperObs(BaseModel):
    pose: Pose
    holding: str | None = None


class WorldState(BaseModel):
    t: float  # env time when this snapshot was taken
    step: int  # number of skills executed so far
    objects: dict[str, ObjectObs]
    gripper: GripperObs


class FailureCode(StrEnum):
    INVALID_ARGS = "INVALID_ARGS"
    UNKNOWN_TOOL = "UNKNOWN_TOOL"
    UNKNOWN_OBJECT = "UNKNOWN_OBJECT"
    OUT_OF_WORKSPACE = "OUT_OF_WORKSPACE"
    NOT_HOLDING = "NOT_HOLDING"
    ALREADY_HOLDING = "ALREADY_HOLDING"
    OBJECT_NOT_AT_POSE = "OBJECT_NOT_AT_POSE"
    GRASP_SLIP = "GRASP_SLIP"
    POSTCONDITION_FAILED = "POSTCONDITION_FAILED"
    GOAL_NOT_MET = "GOAL_NOT_MET"
    SKIPPED = "SKIPPED"  # not executed: an earlier call in the same turn failed
    UNGROUNDED_ACTION = "UNGROUNDED_ACTION"  # pose-dependent call not grounded in a delivered current state
    DESTINATION_OCCUPIED = "DESTINATION_OCCUPIED"  # place would overlap a cube or fill an occupied region


# Failures that are safe to retry with identical arguments (if preconditions still hold).
TRANSIENT = frozenset({FailureCode.GRASP_SLIP})

# Occupancy model (F011). Only cubes are solid; plates and regions are never obstacles. A
# target_region holds at most one cube, judged by cube centre with the same strict box test as
# goals.InRegion.
CUBE_HALF = 0.02    # cube half-size (m)
REGION_HALF = 0.045  # target_region half-size (m)


def placement_conflict(state: "WorldState", held: str, x: float, y: float) -> str | None:
    """Why placing the held object with its centre at (x, y) is impossible, or None if it is free.

    R2 capacity (reported first, the more useful reason for planning): (x, y) lies in a
    target_region that already contains another cube's centre.
    R1 non-overlap: the cube footprint would overlap another resting cube.
    The held object is ignored, so it can always be put back where it came from.
    """
    cubes = {oid: o.pose for oid, o in state.objects.items() if o.kind == "cube" and oid != held}
    for rid, r in state.objects.items():
        if r.kind != "target_region":
            continue
        inside = lambda px, py: abs(px - r.pose.x) < REGION_HALF and abs(py - r.pose.y) < REGION_HALF  # noqa: E731
        if inside(x, y):
            for oid, p in cubes.items():
                if inside(p.x, p.y):
                    return f"region occupied: {rid} already contains {oid}"
    for oid, p in cubes.items():
        if abs(p.x - x) < 2 * CUBE_HALF and abs(p.y - y) < 2 * CUBE_HALF:
            return f"cube overlap: ({x:.3f}, {y:.3f}) overlaps {oid} at ({p.x:.3f}, {p.y:.3f})"
    return None


class Failure(BaseModel):
    code: FailureCode
    detail: str


class SkillOutcome(BaseModel):
    """What the env reports about its own execution. Not trusted on its own; see skills.verify."""

    ok: bool
    failure: Failure | None = None
    info: dict[str, Any] = {}  # backend diagnostics (controller steps, termination, ...), traced verbatim
