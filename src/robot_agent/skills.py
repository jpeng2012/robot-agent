"""Skill (tool) definitions: argument schemas, validation, preconditions, postconditions.

Skill-level checks only answer "did this action do what it claims?". Whether the
task is done is decided separately by goals.check_goal.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from robot_agent.env import GRASP_TOL
from robot_agent.world import WORKSPACE, Failure, FailureCode, SkillOutcome, WorldState

PLACE_TOL = 0.02


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class ObserveArgs(_Args):
    pass


class PickArgs(_Args):
    object_id: str = Field(description="ID of the object to grasp, from the latest world state.")
    x: float = Field(description="Commanded grasp x (m); use the object's latest observed pose.")
    y: float = Field(description="Commanded grasp y (m); use the object's latest observed pose.")


class PlaceArgs(_Args):
    x: float = Field(description="Target x (m) for the held object.")
    y: float = Field(description="Target y (m) for the held object.")


class FinishArgs(_Args):
    summary: str = Field(description="Why you believe the task is complete.")


Check = Callable[[WorldState, Any], Failure | None]
PostCheck = Callable[[WorldState, WorldState, Any], Failure | None]


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    args: type[_Args]
    precondition: Check | None = None
    postcondition: PostCheck | None = None


def _pick_pre(s: WorldState, a: PickArgs) -> Failure | None:
    if s.gripper.holding is not None:
        return Failure(code=FailureCode.ALREADY_HOLDING, detail=f"already holding {s.gripper.holding}")
    err = s.objects[a.object_id].pose.dist_xy(a.x, a.y)
    if err > GRASP_TOL:
        return Failure(
            code=FailureCode.OBJECT_NOT_AT_POSE,
            detail=f"latest observation puts {a.object_id} {err:.3f} m from ({a.x:.3f}, {a.y:.3f})",
        )
    return None


def _pick_post(before: WorldState, after: WorldState, a: PickArgs) -> Failure | None:
    if after.gripper.holding != a.object_id:
        return Failure(
            code=FailureCode.POSTCONDITION_FAILED,
            detail=f"expected to hold {a.object_id}, holding {after.gripper.holding}",
        )
    return None


def _place_pre(s: WorldState, a: PlaceArgs) -> Failure | None:
    if s.gripper.holding is None:
        return Failure(code=FailureCode.NOT_HOLDING, detail="gripper is empty")
    return None


def _place_post(before: WorldState, after: WorldState, a: PlaceArgs) -> Failure | None:
    oid = before.gripper.holding
    if after.gripper.holding is not None:
        return Failure(code=FailureCode.POSTCONDITION_FAILED, detail=f"still holding {after.gripper.holding}")
    err = after.objects[oid].pose.dist_xy(a.x, a.y)
    if err > PLACE_TOL:
        p = after.objects[oid].pose
        return Failure(
            code=FailureCode.POSTCONDITION_FAILED,
            detail=f"{oid} ended at ({p.x:.3f}, {p.y:.3f}), {err:.3f} m from the commanded place point",
        )
    return None


SKILLS: dict[str, Skill] = {
    s.name: s
    for s in [
        Skill("observe", "Return the current world state.", ObserveArgs),
        Skill("pick", "Move to (x, y) and grasp the object.", PickArgs, _pick_pre, _pick_post),
        Skill("place", "Move the held object to (x, y) and release it.", PlaceArgs, _place_pre, _place_post),
        Skill("finish", "Declare the task complete. The goal is verified independently.", FinishArgs),
    ]
}


def tool_schemas() -> list[dict]:
    """Provider-neutral tool definitions (name, description, JSON Schema input)."""
    return [
        {"name": s.name, "description": s.description, "input_schema": s.args.model_json_schema()}
        for s in SKILLS.values()
    ]


def validate_call(name: str, raw_args: Any, state: WorldState) -> _Args | Failure:
    """Parse and check a tool call against the fresh world state before it can reach the env."""
    skill = SKILLS.get(name)
    if skill is None:
        return Failure(code=FailureCode.UNKNOWN_TOOL, detail=f"unknown tool {name!r}; valid: {sorted(SKILLS)}")
    try:
        args = skill.args.model_validate(raw_args)
    except ValidationError as e:
        errs = "; ".join(f"{'.'.join(map(str, err['loc'])) or 'args'}: {err['msg']}" for err in e.errors())
        return Failure(code=FailureCode.INVALID_ARGS, detail=errs)
    if (oid := getattr(args, "object_id", None)) is not None and oid not in state.objects:
        return Failure(code=FailureCode.UNKNOWN_OBJECT, detail=f"no object {oid!r}; known: {sorted(state.objects)}")
    if hasattr(args, "x"):
        x0, x1, y0, y1 = WORKSPACE
        if not (x0 <= args.x <= x1 and y0 <= args.y <= y1):
            return Failure(
                code=FailureCode.OUT_OF_WORKSPACE,
                detail=f"({args.x}, {args.y}) outside workspace x∈[{x0}, {x1}], y∈[{y0}, {y1}]",
            )
    return precondition(name, args, state) or args


def precondition(name: str, args: _Args, state: WorldState) -> Failure | None:
    check = SKILLS[name].precondition
    return check(state, args) if check else None


def verify(name: str, args: _Args, before: WorldState, outcome: SkillOutcome, after: WorldState) -> Failure | None:
    """Combine the env's report with a postcondition check on the refreshed state."""
    if not outcome.ok:
        return outcome.failure
    check = SKILLS[name].postcondition
    return check(before, after, args) if check else None
