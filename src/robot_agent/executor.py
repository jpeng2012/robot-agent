"""Closed-loop symbolic plan execution (F012c, mock backend).

PlanExecutor is an LLMClient: AgentLoop drives it exactly as it drives a model, so every action still passes
validate_call, F010 grounding, verify and the loop's goal_check. It has no environment handle; it
reads only the tool results in `messages`.

Per turn it emits at most one tool call:
  1. No pending action and no delivered state -> observe.
  2. Otherwise it first reconciles the pending action with its own tool result (which must be the newest
     tool message, carry a world_state that is not older than the state the action was grounded on, and,
     for an ok physical action, show that a skill actually executed). Missing or stale feedback is never
     treated as success: it costs one replan and an observe.
  3. A failed result (non-transient, i.e. after the loop's own retries; GOAL_NOT_MET included) or a
     delivered state that differs from planner.simulate's prediction triggers a replan from that state.
  4. Before emitting a step it re-checks it on the newest delivered state (pick: the object is within
     GRASP_TOL of its planned pose; place: holding the planned cube and the destination is free; finish:
     every goal holds). Coordinates are always taken from the delivered state (pick: the object's pose;
     goal place: the region centre; staging place: the planned point, re-validated).
Replans are bounded by max_replans. A second re-plan from a physically identical state is a livelock (the
deterministic planner would repeat a plan that already failed from there), so a non-transient failure that
leaves the scene unchanged gets exactly one re-attempt. Planning failures and executor failures end in a repeated text-only stop (AgentLoop
ends the episode as llm_stopped) whose meta says which. Success is decided only by AgentLoop.
"""

import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from robot_agent import planner
from robot_agent.env import GRASP_TOL
from robot_agent.goals import InRegion
from robot_agent.llm import AssistantTurn, ToolCall, call, latest_state
from robot_agent.planner import Bounds, InvalidStep, Plan, PlanningFailure, PlanStep
from robot_agent.skills import PLACE_TOL
from robot_agent.world import WORKSPACE, FailureCode, WorldState

EXECUTOR = "plan"


class ExecutorFailureCode(StrEnum):
    REPLAN_BUDGET_EXHAUSTED = "REPLAN_BUDGET_EXHAUSTED"
    LIVELOCK_DETECTED = "LIVELOCK_DETECTED"


class Trigger(StrEnum):
    EXECUTION_FAILURE = "EXECUTION_FAILURE"  # failed tool result (after the loop's transient retries)
    GOAL_NOT_MET = "GOAL_NOT_MET"            # finish rejected by the loop's goal_check
    STATE_DEVIATION = "STATE_DEVIATION"      # ok result, but the delivered state differs from the prediction
    PRECHECK_FAILED = "PRECHECK_FAILED"      # the next step no longer holds on the delivered state
    MISSING_FEEDBACK = "MISSING_FEEDBACK"    # no tool result (or no world_state) for the pending call
    STALE_FEEDBACK = "STALE_FEEDBACK"        # result state older than the action's grounding, or no execution


@dataclass(frozen=True)
class _Pending:
    call_id: str
    name: str
    step_index: int | None
    before: WorldState | None        # the delivered state the call was grounded on
    expected: WorldState | None      # planner.simulate prediction (pick/place only)
    reason: dict | None = None       # carried replan reason for a recovery observe


@dataclass(frozen=True)
class Tolerances:
    """Reconciliation tolerances. Object identities, kinds and the held object are always compared exactly."""
    xy: float = PLACE_TOL       # every non-held object (resting cubes, regions)
    z: float = PLACE_TOL
    held_pose: bool = True      # also compare the held object's pose with the nominal (mock) carry pose


MOCK_TOLERANCES = Tolerances()
# Physical backends (F015 robosuite): resting xy as tight as the place postcondition (PLACE_TOL), resting z
# as the InRegion / worker resting tolerance; the carried pose is not predicted (grasp offset and lift
# height vary) and is already checked by the pick postcondition (CARRY_XY_TOL, MIN_CARRY_Z).
PHYSICAL_TOLERANCES = Tolerances(xy=PLACE_TOL, z=0.025, held_pose=False)


def deviation(expected: WorldState, actual: WorldState, tol: Tolerances = MOCK_TOLERANCES) -> str | None:
    """Why `actual` differs from `expected` beyond `tol` (held object, object set, kinds, poses)."""
    if expected.gripper.holding != actual.gripper.holding:
        return f"holding {actual.gripper.holding!r}, expected {expected.gripper.holding!r}"
    if set(expected.objects) != set(actual.objects):
        return f"objects {sorted(actual.objects)}, expected {sorted(expected.objects)}"
    for oid in sorted(expected.objects):
        e, a = expected.objects[oid], actual.objects[oid]
        if e.kind != a.kind:
            return f"{oid} is a {a.kind}, expected {e.kind}"
        if not tol.held_pose and oid == actual.gripper.holding:
            continue
        d, dz = a.pose.dist_xy(e.pose.x, e.pose.y), abs(a.pose.z - e.pose.z)
        if d > tol.xy or dz > tol.z:
            return f"{oid} at ({a.pose.x:.3f}, {a.pose.y:.3f}, {a.pose.z:.3f}), expected " \
                   f"({e.pose.x:.3f}, {e.pose.y:.3f}, {e.pose.z:.3f})"
    return None


def _signature(s: WorldState) -> str:
    """Physical content of a state (no t/step/observed_at), for livelock detection."""
    return json.dumps([s.gripper.holding, sorted((k, o.kind, round(o.pose.x, 6), round(o.pose.y, 6),
                                                  round(o.pose.z, 6)) for k, o in s.objects.items())])


class PlanExecutor:
    """Deterministic plan-following LLMClient for trusted InRegion goals (see module docstring)."""

    def __init__(self, goals: list[InRegion], bounds: Bounds = WORKSPACE, max_replans: int = 3,
                 tolerances: Tolerances = MOCK_TOLERANCES):
        if max_replans < 0:
            raise ValueError("max_replans must be >= 0")
        self.goals, self.bounds, self.max_replans = list(goals), bounds, max_replans
        self.tolerances = tolerances
        self._reset()

    def _reset(self) -> None:
        """Per-episode state; called at the start of every episode, so an instance can be reused."""
        self.plan: Plan | None = None
        self.plan_id = -1
        self.index = 0  # next step of self.plan
        self.replans = 0
        self._pending: _Pending | None = None
        self._replanned_from: set[str] = set()  # signatures of states already re-planned from
        self._announce: dict | None = None  # plan/reason metadata for the next emitted turn
        self._stop: AssistantTurn | None = None

    # ------------------------------------------------------------------ LLMClient
    def step(self, system: str, messages: list[dict], tools: list[dict]) -> AssistantTurn:
        if not any(m["role"] == "assistant" for m in messages):  # a new episode
            self._reset()
        if self._stop is not None:  # terminal: repeat the stop so the loop ends as llm_stopped
            return self._stop.model_copy(deep=True)
        pending, self._pending = self._pending, None
        if pending is None:
            s = latest_state(messages)
            if s is None:
                return self._observe(None, None)
            return self._new_plan(s, None)
        s, reason = self._reconcile(pending, messages)
        if s is None:  # missing or stale feedback: nothing can be trusted; observe, then re-plan
            if (stop := self._consume_replan(reason)) is not None:
                return stop
            return self._observe(pending.before, reason)
        if pending.name == "observe" and reason is None:
            if pending.reason is not None and (stop := self._livelock(s, pending.reason)) is not None:
                return stop
            return self._new_plan(s, pending.reason)
        if reason is not None:
            return self._replan(s, reason)
        self.index += 1
        return self._advance(s)

    # ------------------------------------------------------------------ reconciliation
    def _reconcile(self, p: _Pending, messages: list[dict]) -> tuple[WorldState | None, dict | None]:
        base = {"step_index": p.step_index, "plan_id": self.plan_id, "call": p.name}
        result = next((m for m in reversed(messages) if m["role"] == "tool"), None)
        if result is None or result.get("tool_call_id") != p.call_id:
            got = None if result is None else result.get("tool_call_id")
            return None, {"trigger": Trigger.MISSING_FEEDBACK, "code": None,
                          "detail": f"newest tool result is for {got!r}, not {p.call_id!r}", **base}
        content = result["content"]
        if content.get("world_state") is None:
            return None, {"trigger": Trigger.MISSING_FEEDBACK, "code": None,
                          "detail": f"result for {p.call_id} has no world_state", **base}
        s = WorldState.model_validate(content["world_state"])
        b = p.before
        if b is not None and (s.step < b.step or s.t < b.t):
            return None, {"trigger": Trigger.STALE_FEEDBACK, "code": None,
                          "detail": f"result state (step {s.step}, t {s.t}) is older than its grounding "
                                    f"(step {b.step}, t {b.t})", **base}
        ok = content.get("ok") is True
        if ok and p.name in ("pick", "place") and s.step <= b.step:
            return None, {"trigger": Trigger.STALE_FEEDBACK, "code": None,
                          "detail": f"{p.name} reported ok but no skill executed (step {s.step})", **base}
        if not ok:
            code = (content.get("error") or {}).get("code")
            trig = Trigger.GOAL_NOT_MET if code == FailureCode.GOAL_NOT_MET else Trigger.EXECUTION_FAILURE
            return s, {"trigger": trig, "code": code, "detail": (content.get("error") or {}).get("detail"), **base}
        if p.expected is not None and (why := deviation(p.expected, s, self.tolerances)) is not None:
            return s, {"trigger": Trigger.STATE_DEVIATION, "code": None, "detail": why, **base}
        return s, None

    # ------------------------------------------------------------------ planning
    def _consume_replan(self, reason: dict) -> AssistantTurn | None:
        if self.replans >= self.max_replans:
            return self._terminal({"executor_failure": {
                "code": ExecutorFailureCode.REPLAN_BUDGET_EXHAUSTED,
                "detail": f"replan needed after {self.replans} of max_replans={self.max_replans}"}}, reason)
        self.replans += 1
        return None

    def _livelock(self, s: WorldState, reason: dict) -> AssistantTurn | None:
        """A second re-plan from a physically identical state would repeat the same plan (the planner is
        deterministic) after it already failed once from there: stop. Every non-transient failure that
        leaves the scene unchanged therefore gets exactly one re-attempt, wherever it occurs in a plan."""
        sig = _signature(s)
        if sig in self._replanned_from:
            return self._terminal({"executor_failure": {
                "code": ExecutorFailureCode.LIVELOCK_DETECTED,
                "detail": "already re-planned once from this physical state"}}, reason)
        self._replanned_from.add(sig)
        return None

    def _replan(self, s: WorldState, reason: dict) -> AssistantTurn:
        if (stop := self._livelock(s, reason) or self._consume_replan(reason)) is not None:
            return stop
        return self._new_plan(s, reason)

    def _new_plan(self, s: WorldState, reason: dict | None) -> AssistantTurn:
        p = planner.plan(self.goals, s, bounds=self.bounds)
        if isinstance(p, PlanningFailure):
            return self._terminal({"planning_failure": {"code": p.code, "subcode": p.subcode, "detail": p.detail}},
                                  reason)
        self.plan, self.index, self.plan_id = p, 0, self.plan_id + 1
        self._announce = {"plan": p.model_dump(mode="json"), "source_step": s.step,
                          **({"replan_reason": reason} if reason else {})}
        return self._advance(s)

    # ------------------------------------------------------------------ acting
    def _ground(self, s: WorldState, st: PlanStep) -> tuple[ToolCall, WorldState | None] | str:
        """The grounded call for plan step `st` on delivered state `s` and its predicted result, or why not."""
        if st.op == "finish":
            unmet = [str(g) for g in self.goals if not g.holds(s)]
            if unmet:
                return f"goals not met in the delivered state: {unmet}"
            return call("finish", summary="all requested goals hold in the latest world state"), None
        if st.op == "pick":
            if st.object_id not in s.objects:
                return f"{st.object_id} is not in the delivered state"
            p = s.objects[st.object_id].pose
            if (d := p.dist_xy(st.x, st.y)) > GRASP_TOL:
                return f"{st.object_id} is {d:.3f} m from its planned pose"
            x0, x1, y0, y1 = self.bounds
            if not (x0 <= p.x <= x1 and y0 <= p.y <= y1):
                return f"{st.object_id} at ({p.x}, {p.y}) is outside the executor bounds"
            issued = st.model_copy(update={"x": p.x, "y": p.y})
        else:
            if s.gripper.holding != st.object_id:
                return f"holding {s.gripper.holding!r}, plan places {st.object_id}"
            if st.purpose == "goal":
                if st.region not in s.objects:
                    return f"region {st.region} is not in the delivered state"
                r = s.objects[st.region].pose
                issued = st.model_copy(update={"x": r.x, "y": r.y})
            else:
                issued = st
        try:
            expected = planner.simulate(s, issued)  # also rejects occupied destinations (placement_conflict)
        except InvalidStep as e:
            return str(e)
        args = {"object_id": issued.object_id} if st.op == "pick" else {}
        return call(st.op, **args, x=issued.x, y=issued.y), expected

    def _advance(self, s: WorldState) -> AssistantTurn:
        st = self.plan.steps[self.index]
        g = self._ground(s, st)
        if isinstance(g, str):
            return self._replan(s, {"trigger": Trigger.PRECHECK_FAILED, "code": None, "detail": g,
                                    "step_index": self.index, "plan_id": self.plan_id, "call": st.op})
        tc, expected = g
        self._pending = _Pending(tc.id, tc.name, self.index, s, expected)
        return self._turn([tc], mode=st.op, step_index=self.index, purpose=st.purpose)

    def _observe(self, before: WorldState | None, reason: dict | None) -> AssistantTurn:
        tc = call("observe")
        self._pending = _Pending(tc.id, "observe", None, before, None, reason)
        extra = {"replan_reason": reason} if reason else {}
        return self._turn([tc], mode="observe", step_index=None, purpose=None, **extra)

    def _turn(self, tool_calls: list[ToolCall], **meta: Any) -> AssistantTurn:
        m = {"executor": EXECUTOR, **meta, "plan_id": self.plan_id if self.plan else None, "replans": self.replans}
        if self._announce is not None:
            m.update(self._announce)
            self._announce = None
        return AssistantTurn(tool_calls=tool_calls, meta=_jsonable(m))

    def _terminal(self, outcome: dict, reason: dict | None) -> AssistantTurn:
        (kind, info), = [(k, v) for k, v in outcome.items() if k in ("planning_failure", "executor_failure")]
        text = f"{kind.replace('_', ' ')}: {info['code']}: {info['detail']}"
        meta = {"executor": EXECUTOR, "mode": "stop", **outcome, "plan_id": self.plan_id if self.plan else None,
                "replans": self.replans, **({"replan_reason": reason} if reason else {})}
        self._announce = None
        self._stop = AssistantTurn(text=text, meta=_jsonable(meta))
        return self._stop.model_copy(deep=True)


def _jsonable(d: dict) -> dict:
    return json.loads(json.dumps(d, default=str))
