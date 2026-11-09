"""Symbolic dependency planner for unordered in-region goals (F012b-2, mock backend only).

plan(goals, state, bounds=WORKSPACE, max_physical=None, clearance=0.0, staging_margin=PLACE_TOL) -> Plan | PlanningFailure
simulate(state, step) -> WorldState  (the rollout transition, also used by executor.PlanExecutor)

The planner never touches a RobotEnv: it rolls the dependency-aware step policy (policies.decide, aware=True)
forward over a private copy of the given WorldState, using the mock's pick/place kinematics. Goals are validated
(ids, kinds, duplicates, feasibility, region geometry, bounds) and canonicalised, so their presentation order
cannot influence the plan. Each turn the unsatisfied goals are re-prioritised: destination free first, then
goals whose blocker has no goal of its own (it must be staged anyway), then the remaining cycles (one staging
move each); ties fall back to canonical order. Every place is checked with world.placement_conflict.

A plan is open-loop: an executor must still ground every step on fresh state and re-plan on deviation.
"""

import itertools
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel

from robot_agent.env import CARRY_Z, GRASP_TOL
from robot_agent.goals import InRegion
from robot_agent.skills import PLACE_TOL
from robot_agent import policies
from robot_agent.world import REGION_HALF, WORKSPACE, GripperObs, Pose, WorldState, placement_conflict

Bounds = tuple[float, float, float, float]


class PlanningFailureCode(StrEnum):
    INVALID_GOALS = "INVALID_GOALS"
    INFEASIBLE_GOALS = "INFEASIBLE_GOALS"
    OUT_OF_BOUNDS = "OUT_OF_BOUNDS"
    UNSUPPORTED_GEOMETRY = "UNSUPPORTED_GEOMETRY"
    NO_STAGING = "NO_STAGING"
    BLOCKER_NOT_FOUND = "BLOCKER_NOT_FOUND"
    CONFLICTING_GOALS = "CONFLICTING_GOALS"
    BOUND_EXCEEDED = "BOUND_EXCEEDED"
    INVALID_STATE = "INVALID_STATE"  # inconsistent input state (e.g. holding an unknown or non-cube object)
    PLANNER_INTERNAL = "PLANNER_INTERNAL"  # planner/policy fault, never a property of the user's goals

POLICY_STOPS = {"NO_STAGING": PlanningFailureCode.NO_STAGING, "BLOCKER_NOT_FOUND": PlanningFailureCode.BLOCKER_NOT_FOUND,
                "CONFLICTING_GOALS": PlanningFailureCode.CONFLICTING_GOALS}


class PlanStep(BaseModel):
    op: Literal["pick", "place", "finish"]
    purpose: Literal["goal", "clear_blocker", "stage", "finish"]
    object_id: str | None = None
    region: str | None = None  # goal-region placements
    x: float | None = None  # pick: the object's planned pose; place: the release point
    y: float | None = None


class Plan(BaseModel):
    goals: list[tuple[str, str]]  # canonical (object_id, region), sorted
    steps: list[PlanStep]  # ends with finish
    physical_actions: int
    staging_moves: int


class GoalError(BaseModel):
    code: str
    index: int | None = None
    detail: str


class PlanningFailure(BaseModel):
    code: PlanningFailureCode
    subcode: str | None = None
    detail: str
    errors: list[GoalError] = []
    partial_steps: list[PlanStep] = []


def _fail(code, detail, sub=None, errors=(), steps=()):
    return PlanningFailure(code=code, subcode=sub, detail=detail, errors=list(errors), partial_steps=list(steps))


def _inside(b: Bounds, x: float, y: float) -> bool:
    return b[0] <= x <= b[1] and b[2] <= y <= b[3]


def _overlap(state: WorldState, a: str, b: str) -> bool:
    pa, pb = state.objects[a].pose, state.objects[b].pose
    return abs(pa.x - pb.x) < 2 * REGION_HALF and abs(pa.y - pb.y) < 2 * REGION_HALF


def validate(goals, state: WorldState) -> tuple[list[tuple[str, str]] | None, PlanningFailure | None]:
    """Strict goal validation. Returns (canonical goals, None) or (None, failure)."""
    errors, seen = [], set()
    goals = list(goals)
    if not goals:
        errors.append(GoalError(code="EMPTY", detail="no goals"))
    for i, g in enumerate(goals):
        if not isinstance(g, InRegion):
            errors.append(GoalError(code="UNSUPPORTED_PREDICATE", index=i, detail=type(g).__name__))
            continue
        if (g.half_x, g.half_y) != (REGION_HALF, REGION_HALF):
            errors.append(GoalError(code="UNSUPPORTED_REGION_SIZE", index=i, detail=f"{g.half_x}, {g.half_y}"))
        o, r = g.obj, g.region
        if o not in state.objects:
            errors.append(GoalError(code="UNKNOWN_OBJECT", index=i, detail=o))
        elif state.objects[o].kind != "cube":
            errors.append(GoalError(code="NOT_A_CUBE", index=i, detail=f"{o} is {state.objects[o].kind}"))
        if r not in state.objects:
            errors.append(GoalError(code="UNKNOWN_REGION", index=i, detail=r))
        elif state.objects[r].kind != "target_region":
            errors.append(GoalError(code="NOT_A_REGION", index=i, detail=f"{r} is {state.objects[r].kind}"))
        if (o, r) in seen:
            errors.append(GoalError(code="DUPLICATE", index=i, detail=f"in_region({o}, {r})"))
        seen.add((o, r))
    if errors:
        return None, _fail(PlanningFailureCode.INVALID_GOALS, "invalid goals", errors[0].code, errors)
    canonical = sorted(seen)
    for (o1, r1), (o2, r2) in itertools.combinations(canonical, 2):
        if o1 == o2:
            return None, _fail(PlanningFailureCode.INFEASIBLE_GOALS, f"{o1} cannot be in {r1} and {r2}",
                               "OBJECT_CONFLICT")
        if r1 == r2:
            return None, _fail(PlanningFailureCode.INFEASIBLE_GOALS, f"{r1} holds at most one cube ({o1}, {o2})",
                               "REGION_CONFLICT")
    return canonical, None


class InvalidStep(ValueError):
    """simulate() was asked for a transition the mock could not execute successfully."""


def simulate(state: WorldState, step: PlanStep) -> WorldState:
    """The successful transition of one physical plan step under mock kinematics (env.py), on a copy.

    pick lifts the cube to CARRY_Z at the commanded (x, y) and the gripper holds it there; place sets the
    held cube at (x, y, 0) and releases it. Pure and deterministic; the input is never modified. Steps the
    mock would reject (and finish, which has no physical effect) raise InvalidStep. This is the planner's
    own rollout transition, so plans and executor predictions share one model.
    """
    g = state.gripper
    if step.op not in ("pick", "place"):
        raise InvalidStep(f"{step.op} has no physical transition")
    if step.x is None or step.y is None:
        raise InvalidStep(f"{step.op} needs x and y")
    x, y = step.x, step.y
    objs = {k: o.model_copy(deep=True) for k, o in state.objects.items()}
    if step.op == "pick":
        oid = step.object_id
        if g.holding is not None:
            raise InvalidStep(f"pick {oid}: already holding {g.holding}")
        if oid not in objs or objs[oid].kind != "cube":
            raise InvalidStep(f"pick {oid!r}: not a known cube")
        if objs[oid].pose.dist_xy(x, y) > GRASP_TOL:
            raise InvalidStep(f"pick {oid}: ({x}, {y}) is more than GRASP_TOL from its pose")
        objs[oid].pose = Pose(x=x, y=y, z=CARRY_Z)
        grip = GripperObs(pose=Pose(x=x, y=y, z=CARRY_Z), holding=oid)
    else:
        held = g.holding
        if held is None:
            raise InvalidStep("place: gripper is empty")
        if step.object_id is not None and step.object_id != held:
            raise InvalidStep(f"place {step.object_id}: holding {held}")
        if (why := placement_conflict(state, held, x, y)) is not None:
            raise InvalidStep(f"place {held}: {why}")
        objs[held].pose = Pose(x=x, y=y, z=0.0)
        grip = GripperObs(pose=Pose(x=x, y=y, z=CARRY_Z), holding=None)
    return state.model_copy(update={"objects": objs, "gripper": grip, "step": state.step + 1})


def _prioritised(s: WorldState, base: list[InRegion], clearance: float = 0.0) -> list[InRegion]:
    goal_obj = {g.obj for g in base}

    def key(ig):
        i, g = ig
        if g.holds(s):
            return (3, i)
        r = s.objects[g.region].pose
        if placement_conflict(s, g.obj, r.x, r.y) is None and \
                policies._crowding(s, r.x, r.y, clearance, exclude=(g.obj,)) is None:
            return (0, i)
        b = policies._blocker(s, g, clearance)
        return (1 if b is not None and b not in goal_obj else 2, i)
    return [g for _, g in sorted(enumerate(base), key=key)]


def plan(goals, state: WorldState, bounds: Bounds = WORKSPACE, max_physical: int | None = None,
         clearance: float = 0.0, staging_margin: float = PLACE_TOL) -> Plan | PlanningFailure:
    """Plan for unordered InRegion goals from an authoritative WorldState (not modified).

    If every goal already holds, the plan is just finish, even while an unrelated cube is held.
    clearance (default 0, mock: disabled) is the backend's physical place clearance. With clearance > 0 a goal
    destination is blocked while another cube's centre is within `clearance` of the region centre (that cube is
    cleared first; a held cube whose goal is crowded is staged), and staging keeps clearance + staging_margin
    from every other cube and region centre. No qualifying staging point: PlanningFailure NO_STAGING."""
    if max_physical is not None and max_physical < 0:
        raise ValueError("max_physical must be >= 0")
    if clearance < 0 or staging_margin < 0:
        raise ValueError("clearance and staging_margin must be >= 0")
    held = state.gripper.holding
    if held is not None and (held not in state.objects or state.objects[held].kind != "cube"):
        return _fail(PlanningFailureCode.INVALID_STATE, f"gripper holds {held!r}, not a known cube", "HELD_OBJECT")
    canonical, failure = validate(goals, state)
    if failure:
        return failure
    W = WORKSPACE
    if not (W[0] <= bounds[0] <= bounds[1] <= W[1] and W[2] <= bounds[2] <= bounds[3] <= W[3]):
        return _fail(PlanningFailureCode.OUT_OF_BOUNDS, f"bounds {bounds} not inside WORKSPACE {W}",
                     "BOUNDS_EXCEED_WORKSPACE")
    if any(o.kind not in ("cube", "target_region") for o in state.objects.values()):
        return _fail(PlanningFailureCode.UNSUPPORTED_GEOMETRY, "only cubes and target regions are supported",
                     "OTHER_OBJECT_KINDS")
    regions = sorted(k for k, o in state.objects.items() if o.kind == "target_region")
    for a, b in itertools.combinations(regions, 2):
        if _overlap(state, a, b):
            return _fail(PlanningFailureCode.UNSUPPORTED_GEOMETRY, f"regions {a} and {b} overlap", "OVERLAPPING_REGIONS")
    for _, r in canonical:
        p = state.objects[r].pose
        if not _inside(bounds, p.x, p.y):
            return _fail(PlanningFailureCode.OUT_OF_BOUNDS, f"goal region {r} centre outside bounds", "GOAL_REGION")
    base = [InRegion(obj=o, region=r, half_x=REGION_HALF, half_y=REGION_HALF) for o, r in canonical]
    goal_region = dict(canonical)
    n_cubes = sum(o.kind == "cube" for o in state.objects.values())
    bound = 4 * n_cubes if max_physical is None else max_physical
    s = state.model_copy(deep=True)
    steps: list[PlanStep] = []
    while True:
        turn = policies.decide(s, _prioritised(s, base, clearance), aware=True, bounds=bounds, clearance=clearance,
                               staging_margin=staging_margin)
        if not turn.tool_calls:
            code = turn.meta.get("code")  # structured stop from the policy
            if code in POLICY_STOPS:
                return _fail(POLICY_STOPS[code], turn.text, steps=steps)
            return _fail(PlanningFailureCode.PLANNER_INTERNAL, turn.text, f"POLICY_STOP:{code}", steps=steps)
        tc, mode = turn.tool_calls[0], turn.meta["mode"]
        if tc.name == "finish":
            steps.append(PlanStep(op="finish", purpose="finish"))
            phys = [x for x in steps if x.op != "finish"]
            return Plan(goals=canonical, steps=steps, physical_actions=len(phys),
                        staging_moves=sum(x.purpose == "stage" for x in phys))
        if len(steps) >= bound:
            return _fail(PlanningFailureCode.BOUND_EXCEEDED, f"more than {bound} physical actions", steps=steps)
        if tc.name == "pick":
            o = tc.args["object_id"]
            p = s.objects[o].pose
            if not _inside(bounds, p.x, p.y):
                return _fail(PlanningFailureCode.OUT_OF_BOUNDS, f"{o} at ({p.x}, {p.y}) outside bounds", "PICK",
                             steps=steps)
            step = PlanStep(op="pick", purpose="clear_blocker" if mode == "clear_blocker" else "goal",
                            object_id=o, x=p.x, y=p.y)
        else:
            held_now = s.gripper.holding
            x, y = tc.args["x"], tc.args["y"]
            if placement_conflict(s, held_now, x, y) is not None:  # defensive: decide() never proposes one
                return _fail(PlanningFailureCode.PLANNER_INTERNAL, f"policy proposed a conflicting place for {held_now}",
                             "POLICY_CONFLICT", steps=steps)
            if mode == "stage":
                step = PlanStep(op="place", purpose="stage", object_id=held_now, x=x, y=y)
            else:
                step = PlanStep(op="place", purpose="goal", object_id=held_now, region=goal_region[held_now], x=x, y=y)
        steps.append(step)
        try:
            s = simulate(s, step)
        except InvalidStep as e:  # defensive: decide() only proposes executable steps
            return _fail(PlanningFailureCode.PLANNER_INTERNAL, str(e), "SIMULATE_REJECTED", steps=steps[:-1])
