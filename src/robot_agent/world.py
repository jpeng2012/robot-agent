"""Symbolic world state shared by every env backend, the skills, and the agent loop."""

from enum import StrEnum
import math

from pydantic import BaseModel

# Reachable tabletop region (x_min, x_max, y_min, y_max), metres.
WORKSPACE = (-0.5, 0.5, -0.5, 0.5)


class Pose(BaseModel):
    x: float
    y: float
    z: float = 0.0

    def dist_xy(self, x: float, y: float) -> float:
        return math.hypot(self.x - x, self.y - y)


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


# Failures that are safe to retry with identical arguments (if preconditions still hold).
TRANSIENT = frozenset({FailureCode.GRASP_SLIP})


class Failure(BaseModel):
    code: FailureCode
    detail: str


class SkillOutcome(BaseModel):
    """What the env reports about its own execution. Not trusted on its own; see skills.verify."""

    ok: bool
    failure: Failure | None = None
