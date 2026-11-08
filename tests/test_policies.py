"""F011b-1: naive vs dependency-aware deterministic policies on the mock clear_target task."""

import copy
import inspect
import math

import pytest

from robot_agent.env import MockTabletopEnv
from robot_agent.goals import InRegion, On, TaskSpec, check_goal, clear_target_task, region_pick_place_task
from robot_agent.llm import AssistantTurn, PickPlacePolicyLLM, call, latest_state
from robot_agent.loop import SYSTEM_PROMPT, AgentLoop
from robot_agent.policies import DependencyAwarePolicy, NaivePolicy, _blocker
from robot_agent.skills import tool_schemas
from robot_agent.world import CUBE_HALF, REGION_HALF, WORKSPACE, WorldState, placement_conflict

# Independent physical oracle constants (pinned against the production values below).
CUBE = 0.02
REGION = 0.045
TARGET, PARKING, RED0 = (0.05, 0.30), (-0.25, 0.30), (0.10, -0.20)


def test_oracle_constants_match_production():
    assert (CUBE, REGION) == (CUBE_HALF, REGION_HALF)


# ------------------------------------------------------------------ independent physical oracle

def physical_violations(state: dict) -> list[str]:
    """Overlapping resting cubes, or a target_region holding two cubes. Not based on placement_conflict."""
    held = state["gripper"]["holding"]
    objs = state["objects"]
    cubes = sorted(k for k, o in objs.items() if o["kind"] == "cube" and k != held)
    out = []
    for i, a in enumerate(cubes):
        for b in cubes[i + 1:]:
            pa, pb = objs[a]["pose"], objs[b]["pose"]
            if abs(pa["x"] - pb["x"]) < 2 * CUBE and abs(pa["y"] - pb["y"]) < 2 * CUBE:
                out.append(f"overlap {a} {b}")
    ws = WorldState.model_validate(state)
    for rid, r in objs.items():
        if r["kind"] == "target_region":
            n = sum(InRegion(obj=c, region=rid, half_x=REGION, half_y=REGION).holds(ws) for c in cubes)
            if n > 1:
                out.append(f"{rid} holds {n} cubes")
    return out


def is_valid_staging(state: dict, held: str, x: float, y: float) -> bool:
    """Independent check: inside WORKSPACE, footprint outside every region, no cube overlap."""
    x0, x1, y0, y1 = WORKSPACE
    if not (x0 < x < x1 and y0 < y < y1):
        return False
    for rid, r in state["objects"].items():
        p = r["pose"]
        if r["kind"] == "target_region" and abs(x - p["x"]) < REGION + CUBE and abs(y - p["y"]) < REGION + CUBE:
            return False
        if r["kind"] == "cube" and rid != held and abs(x - p["x"]) < 2 * CUBE and abs(y - p["y"]) < 2 * CUBE:
            return False
    return True


# ------------------------------------------------------------------ harness

class Recording:
    """Wraps a policy: records what it was given and returned; checks it does not mutate messages."""

    def __init__(self, policy):
        self.policy, self.calls = policy, []

    def step(self, system, messages, tools):
        before = copy.deepcopy(messages)
        turn = self.policy.step(system, messages, tools)
        assert messages == before, "policy mutated the message list"
        seen = latest_state(messages)
        self.calls.append((turn, seen.model_dump(mode="json") if seen else None))
        return turn


class PrefixThen:
    """Scripted prefix (to force a rejected plan), then the policy."""

    def __init__(self, prefix, policy):
        self.prefix, self.policy = list(prefix), policy

    def step(self, system, messages, tools):
        if self.prefix:
            return AssistantTurn(tool_calls=self.prefix.pop(0))
        return self.policy.step(system, messages, tools)


def run(tmp_path, llm, task=None, max_turns=20, max_failures=8):
    env = MockTabletopEnv()
    rec = Recording(llm)
    loop = AgentLoop(env, rec, trace_dir=tmp_path, max_turns=max_turns, max_failures=max_failures)
    result = loop.run(task or clear_target_task(), seed=0)
    return result, env, loop.tracer.events, rec


def of(events, kind):
    return [e for e in events if e["event"] == kind]


def calls(events):
    return [c for e in of(events, "llm_turn") for c in e["tool_calls"]]


def physical(events):
    return [(c["name"], c["args"]) for c in calls(events) if c["name"] in ("pick", "place")]


def task_with(layout, goal):
    return TaskSpec(name="clear_target_variant", instruction="test layout", layout=layout, goal=goal)


def region_goal(obj, region):
    return InRegion(obj=obj, region=region, half_x=REGION_HALF, half_y=REGION_HALF)


def _goal_region_at(task, x, y):
    """The InRegion goal whose region contains (x, y), or None (layout poses; regions never move here)."""
    for g in task.goal:
        if isinstance(g, InRegion):
            _, rx, ry = task.layout[g.region]
            if abs(x - rx) < REGION and abs(y - ry) < REGION:
                return g
    return None


def proactive_clearances_from_trace(events, task):
    """Outcome-based, independent of the policy's meta: successful picks of a cube resting in another
    object's unsatisfied goal region, before any DESTINATION_OCCUPIED rejection aimed at that region."""
    results = of(events, "tool_result")
    picks = {c["id"]: c for c in calls(events) if c["name"] == "pick"}
    rejected_regions, count = set(), 0
    for r in results:
        if r["call_id"] in picks and r["ok"]:
            prev = [x for x in results if x["seq"] < r["seq"]][-1]["world_state"]
            ws = WorldState.model_validate(prev)
            oid = picks[r["call_id"]]["args"]["object_id"]
            p = prev["objects"][oid]["pose"]
            g = _goal_region_at(task, p["x"], p["y"])
            if g is not None and g.obj != oid and not g.holds(ws) and g.region not in rejected_regions:
                count += 1
        if not r["ok"] and r["error"]["code"] == "DESTINATION_OCCUPIED":
            args = next(c["args"] for c in calls(events) if c["id"] == r["call_id"])
            g = _goal_region_at(task, args["x"], args["y"])
            if g is not None:
                rejected_regions.add(g.region)
    return count


def metrics(events, task, optimal):
    """Evaluation metrics from trace events only."""
    end = of(events, "episode_end")[-1]
    results = of(events, "tool_result")
    turns = of(events, "llm_turn")
    traced = [e.get(k) for e in events for k in ("state", "state_after", "world_state") if e.get(k)]
    success = (end["result"]["success"] and end["result"]["reason"] == "goal_met"
               and check_goal(task, WorldState.model_validate(end["state"])) == [])
    # Rejected plans: a rejection continues the previous plan only if the immediately preceding
    # tool_result in the event sequence was the same rejection (same held object and place point).
    plans, prev_key = [], None
    for r in results:
        key = None
        if not r["ok"] and r["error"]["code"] == "DESTINATION_OCCUPIED":
            args = next(c["args"] for c in calls(events) if c["id"] == r["call_id"])
            key = (r["world_state"]["gripper"]["holding"], args["x"], args["y"])
            if key != prev_key:
                plans.append((key, r["seq"]))
        prev_key = key
    recovered = unattributed = 0
    for (obj, x, y), seq in plans:
        g = _goal_region_at(task, x, y)
        if g is None:
            unattributed += 1  # rejected place outside every known goal region
            continue
        later = [WorldState.model_validate(r["world_state"]) for r in results if r["seq"] > seq]
        if success and any(region_goal(obj, g.region).holds(s) for s in later):
            recovered += 1
    skills_ok = sum(e["ok"] for e in of(events, "verify"))
    return {
        "success": success,
        "violations_attempted": sum(e["failure"]["code"] == "DESTINATION_OCCUPIED" for e in of(events, "validation_error")),
        "violations_env_rejected": sum(not e["outcome"]["ok"] and e["outcome"]["failure"]["code"] == "DESTINATION_OCCUPIED"
                                       for e in of(events, "env_outcome")),
        "invalid_state_observations": sum(len(physical_violations(s)) for s in traced),
        "turns": end["result"]["llm_turns"],
        "physical_executions": len(of(events, "env_outcome")),
        "successful_skills": skills_ok,
        "unnecessary_actions": skills_ok - optimal if success else None,
        "proactive_resolutions_meta": sum((t.get("meta") or {}).get("mode") == "clear_blocker" for t in turns),
        "proactive_clearances_trace": proactive_clearances_from_trace(events, task),
        "rejected_plans": len(plans),
        "rejected_plans_unattributed": unattributed,
        "reactive_recoveries": recovered if plans else "n/a",
    }


def assert_grounded_and_valid(events, rec, task):
    """Every call was chosen from the previous tool_result's state; args follow from that state only."""
    results = of(events, "tool_result")
    turns = of(events, "llm_turn")
    assert len(turns) == len(rec.calls)
    for turn_ev, (turn, seen) in zip(turns, rec.calls):
        assert len(turn.tool_calls) <= 1  # never chains calls
        prev = [r for r in results if r["seq"] < turn_ev["seq"]]
        assert seen == (prev[-1]["world_state"] if prev else None)
        assert turn_ev["grounding_source_seq"] == (prev[-1]["seq"] if prev else None)
        for tc in turn.tool_calls:
            if tc.name == "pick":
                p = seen["objects"][tc.args["object_id"]]["pose"]
                assert (tc.args["x"], tc.args["y"]) == (p["x"], p["y"])
            if tc.name == "place":
                held = seen["gripper"]["holding"]
                own = [g.region for g in task.goal if isinstance(g, InRegion) and g.obj == held]
                own_xy = {(seen["objects"][r]["pose"]["x"], seen["objects"][r]["pose"]["y"]) for r in own}
                xy = (tc.args["x"], tc.args["y"])
                assert xy in own_xy or is_valid_staging(seen, held, *xy), (xy, held)
    assert not [r for r in results if not r["ok"] and r["error"]["code"] == "UNGROUNDED_ACTION"]
    for e in events:
        for k in ("state", "state_after", "world_state"):
            if e.get(k):
                assert physical_violations(e[k]) == []


def livelock_keys(events):
    """(delivered state without time/step, call) pairs; a repeat means a stateless policy is looping."""
    results = of(events, "tool_result")
    keys = []
    for t in of(events, "llm_turn"):
        prev = [r for r in results if r["seq"] < t["seq"]]
        s = copy.deepcopy(prev[-1]["world_state"]) if prev else None
        if s:
            s.pop("t"), s.pop("step")
            for o in s["objects"].values():
                o.pop("observed_at")
        for c in t["tool_calls"]:
            keys.append(repr((s, c["name"], sorted(c["args"].items()))))
    return keys


T = clear_target_task()
CORRECT, REVERSED = T.goal, T.goal[::-1]
PB, PR = ("pick", {"object_id": "cube_blue", "x": 0.05, "y": 0.3}), ("pick", {"object_id": "cube_red", "x": 0.1, "y": -0.2})
TO_P, TO_T = ("place", {"x": -0.25, "y": 0.3}), ("place", {"x": 0.05, "y": 0.3})
OPTIMAL = [PB, TO_P, PR, TO_T]


# ------------------------------------------------------------------ four core comparisons

# proactive_trace: blocker removed before any attempt to fill its region (outcome, from the trace);
# proactive_meta: the policy's own decision label. They differ when the order already clears the blocker.
@pytest.mark.parametrize("policy, order, seq, attempted, executes, turns, failures, reason, trace_pro, meta_pro", [
    (NaivePolicy, CORRECT, OPTIMAL, 0, 4, 6, 0, "goal_met", 1, 0),
    (NaivePolicy, REVERSED, [PR] + [TO_T] * 8, 8, 1, 10, 8, "failure_budget", 0, 0),
    (DependencyAwarePolicy, CORRECT, OPTIMAL, 0, 4, 6, 0, "goal_met", 1, 0),
    (DependencyAwarePolicy, REVERSED, OPTIMAL, 0, 4, 6, 0, "goal_met", 1, 1),
])
def test_core_comparison(tmp_path, policy, order, seq, attempted, executes, turns, failures, reason,
                         trace_pro, meta_pro):
    result, env, events, rec = run(tmp_path, policy(order))
    assert physical(events) == seq
    assert result.reason == reason and result.llm_turns == turns and result.failures == failures
    assert env.execute_calls == executes
    m = metrics(events, T, optimal=4)
    assert m["violations_attempted"] == attempted and m["violations_env_rejected"] == 0
    assert m["invalid_state_observations"] == 0
    assert (m["proactive_clearances_trace"], m["proactive_resolutions_meta"]) == (trace_pro, meta_pro)
    assert_grounded_and_valid(events, rec, T)


def test_behavioural_difference_metrics(tmp_path):
    naive = metrics(run(tmp_path / "n", NaivePolicy(REVERSED))[2], T, optimal=4)
    aware = metrics(run(tmp_path / "a", DependencyAwarePolicy(REVERSED))[2], T, optimal=4)
    assert naive == {"success": False, "violations_attempted": 8, "violations_env_rejected": 0,
                     "invalid_state_observations": 0, "turns": 10, "physical_executions": 1, "successful_skills": 1,
                     "unnecessary_actions": None, "proactive_resolutions_meta": 0, "proactive_clearances_trace": 0,
                     "rejected_plans": 1, "rejected_plans_unattributed": 0, "reactive_recoveries": 0}
    assert aware == {"success": True, "violations_attempted": 0, "violations_env_rejected": 0,
                     "invalid_state_observations": 0, "turns": 6, "physical_executions": 4, "successful_skills": 4,
                     "unnecessary_actions": 0, "proactive_resolutions_meta": 1, "proactive_clearances_trace": 1,
                     "rejected_plans": 0, "rejected_plans_unattributed": 0, "reactive_recoveries": "n/a"}


# ------------------------------------------------------------------ extra layouts

SWAP = task_with({"target": ("target_region", *TARGET), "cube_blue": ("cube", *TARGET),
                  "parking": ("target_region", *PARKING), "cube_red": ("cube", *PARKING)},
                 [region_goal("cube_blue", "parking"), region_goal("cube_red", "target")])
NO_SUBGOAL = task_with({"target": ("target_region", *TARGET), "cube_blue": ("cube", *TARGET),
                        "cube_red": ("cube", *RED0)}, [region_goal("cube_red", "target")])
FREE_TARGET = task_with({"target": ("target_region", *TARGET), "cube_blue": ("cube", 0.20, 0.0),
                         "parking": ("target_region", *PARKING), "cube_red": ("cube", *RED0)},
                        [region_goal("cube_blue", "parking"), region_goal("cube_red", "target")])


def test_swap_layout_needs_staging(tmp_path):
    result, env, events, rec = run(tmp_path, DependencyAwarePolicy(SWAP.goal), task=SWAP)
    seq = physical(events)
    assert result.reason == "goal_met" and env.execute_calls == 6 and result.llm_turns == 8
    assert [n for n, _ in seq] == ["pick", "place", "pick", "place", "pick", "place"]
    assert seq[0] == ("pick", {"object_id": "cube_red", "x": PARKING[0], "y": PARKING[1]})  # the blocker
    stage = seq[1][1]
    # Pinned: nearest valid grid point to red's pre-pick spot (-0.25, 0.30); four points tie at 0.1 m,
    # (-0.35, 0.3), (-0.15, 0.3), (-0.25, 0.2), (-0.25, 0.4), and the (distance, x, y) key picks the first.
    assert stage == {"x": -0.35, "y": 0.3}
    assert seq[2:] == [("pick", {"object_id": "cube_blue", "x": 0.05, "y": 0.3}), TO_P,
                       ("pick", {"object_id": "cube_red", **stage}), TO_T]
    m = metrics(events, SWAP, optimal=6)
    assert m["violations_attempted"] == 0 and m["unnecessary_actions"] == 0
    # Outcome: both cubes were cleared out of the other's goal region before any attempt (red, then blue);
    # intent: only the first was a clear_blocker decision (blue's own goal was free by then).
    assert (m["proactive_clearances_trace"], m["proactive_resolutions_meta"]) == (2, 1)
    assert_grounded_and_valid(events, rec, SWAP)
    naive = run(tmp_path / "naive", NaivePolicy(SWAP.goal), task=SWAP)
    assert naive[0].reason == "failure_budget" and metrics(naive[2], SWAP, 6)["violations_attempted"] == 8


def test_blocker_without_subgoal_is_staged(tmp_path):
    result, env, events, rec = run(tmp_path, DependencyAwarePolicy(NO_SUBGOAL.goal), task=NO_SUBGOAL)
    seq = physical(events)
    assert result.reason == "goal_met" and env.execute_calls == 4 and result.llm_turns == 6
    assert seq[0] == PB and seq[2] == PR and seq[3] == TO_T
    stage = seq[1][1]
    # Pinned: nearest valid grid point to blue's spot (0.05, 0.30); ties at 0.1 m broken by x.
    assert stage == {"x": -0.05, "y": 0.3}
    assert is_valid_staging(of(events, "tool_result")[1]["world_state"], "cube_blue", stage["x"], stage["y"])
    m = metrics(events, NO_SUBGOAL, optimal=4)
    assert m["violations_attempted"] == 0
    assert (m["proactive_clearances_trace"], m["proactive_resolutions_meta"]) == (1, 1)
    assert_grounded_and_valid(events, rec, NO_SUBGOAL)
    naive = run(tmp_path / "naive", NaivePolicy(NO_SUBGOAL.goal), task=NO_SUBGOAL)
    assert naive[0].reason == "failure_budget" and physical(naive[2])[0] == PR


@pytest.mark.parametrize("policy", [NaivePolicy, DependencyAwarePolicy])
def test_free_target_follows_supplied_order(tmp_path, policy):
    result, env, events, rec = run(tmp_path, policy(FREE_TARGET.goal[::-1]), task=FREE_TARGET)
    assert physical(events) == [PR, TO_T, ("pick", {"object_id": "cube_blue", "x": 0.2, "y": 0.0}), TO_P]
    assert result.reason == "goal_met" and env.execute_calls == 4
    m = metrics(events, FREE_TARGET, optimal=4)
    assert (m["proactive_clearances_trace"], m["proactive_resolutions_meta"]) == (0, 0)  # no false dependency
    assert_grounded_and_valid(events, rec, FREE_TARGET)


def test_recovery_after_rejected_plan_is_reactive(tmp_path):
    prefix = [[call("observe")], [call("pick", object_id="cube_red", x=RED0[0], y=RED0[1])],
              [call("place", x=TARGET[0], y=TARGET[1])]]
    result, env, events, rec = run(tmp_path, PrefixThen(prefix, DependencyAwarePolicy(CORRECT)))
    assert physical(events) == [PR, TO_T, ("place", {"x": 0.1, "y": -0.2}), PB, TO_P, PR, TO_T]
    assert result.reason == "goal_met" and env.execute_calls == 6 and result.llm_turns == 9
    m = metrics(events, T, optimal=4)
    assert m["violations_attempted"] == 1 and m["rejected_plans"] == 1 and m["reactive_recoveries"] == 1
    # Blue was cleared from the target only after the rejected attempt: reactive, by both measures.
    assert (m["proactive_clearances_trace"], m["proactive_resolutions_meta"]) == (0, 0)
    assert m["unnecessary_actions"] == 2  # put red back, then re-pick it
    modes = [(t.get("meta") or {}).get("mode") for t in of(events, "llm_turn")[3:]]
    assert modes[0] == "stage"  # first policy decision: red's target is occupied, so stage it
    assert_grounded_and_valid(events, rec, T)  # scripted prefix and policy turns alike


# ------------------------------------------------------------------ delivered-state-only, statelessness

def test_constructors_take_no_environment():
    for cls in (NaivePolicy, DependencyAwarePolicy):
        assert list(inspect.signature(cls.__init__).parameters) == ["self", "subgoals"]


def test_policies_act_on_synthetic_states_no_env_needed():
    s = MockTabletopEnv().reset(T)
    s.objects["cube_blue"].pose.x, s.objects["cube_blue"].pose.y = 0.07, 0.31  # no env ever produced this
    msgs = [{"role": "user", "content": "Task"},
            {"role": "tool", "tool_call_id": "c0", "name": "observe",
             "content": {"ok": True, "world_state": s.model_dump(mode="json")}}]
    for cls in (NaivePolicy, DependencyAwarePolicy):
        turn = cls(CORRECT).step(SYSTEM_PROMPT, msgs, tool_schemas())
        [tc] = turn.tool_calls
        assert (tc.name, tc.args) == ("pick", {"object_id": "cube_blue", "x": 0.07, "y": 0.31})
    turn = NaivePolicy(CORRECT).step(SYSTEM_PROMPT, [{"role": "user", "content": "Task"}], tool_schemas())
    assert [tc.name for tc in turn.tool_calls] == ["observe"]


def test_layout_shift_changes_arguments_not_behaviour(tmp_path):
    shifted = task_with({k: (kind, x, y + 0.05) for k, (kind, x, y) in T.layout.items()}, T.goal)
    for cls in (NaivePolicy, DependencyAwarePolicy):
        order = CORRECT if cls is NaivePolicy else REVERSED
        result, _, events, rec = run(tmp_path / cls.__name__, cls(order), task=shifted)
        assert result.reason == "goal_met"
        L = shifted.layout  # arguments follow the delivered (shifted) poses exactly
        assert [a["y"] for _, a in physical(events)] == [L["cube_blue"][2], L["parking"][2],
                                                          L["cube_red"][2], L["target"][2]]
        assert L["cube_red"][2] != RED0[1]  # really shifted
        assert_grounded_and_valid(events, rec, shifted)


def test_no_hidden_state_across_episodes(tmp_path):
    policy = DependencyAwarePolicy(REVERSED)
    env = MockTabletopEnv()
    loop = AgentLoop(env, policy, trace_dir=tmp_path, max_turns=20, max_failures=8)
    runs = []
    for _ in range(2):
        r = loop.run(T, seed=0)
        runs.append((physical(loop.tracer.events), r.reason, r.llm_turns, r.failures))
    fresh = run(tmp_path / "fresh", DependencyAwarePolicy(REVERSED))
    runs.append((physical(fresh[2]), fresh[0].reason, fresh[0].llm_turns, fresh[0].failures))
    assert runs[0] == runs[1] == runs[2]
    ids = [c["id"] for c in calls(fresh[2])]
    assert len(ids) == len(set(ids))


# ------------------------------------------------------------------ termination and livelock

def test_dependency_aware_never_repeats_a_state_action_pair(tmp_path):
    for task, goal in ((T, REVERSED), (T, CORRECT), (SWAP, SWAP.goal), (NO_SUBGOAL, NO_SUBGOAL.goal)):
        result, _, events, _ = run(tmp_path / str(id(goal)), DependencyAwarePolicy(goal), task=task)
        keys = livelock_keys(events)
        assert len(keys) == len(set(keys)) and result.reason == "goal_met"


def test_livelock_detector_flags_the_naive_deadlock(tmp_path):
    keys = livelock_keys(run(tmp_path, NaivePolicy(REVERSED))[2])
    assert len(keys) > len(set(keys))


def test_naive_deadlock_is_bounded_by_the_loop_budget(tmp_path):
    result, env, events, _ = run(tmp_path, NaivePolicy(REVERSED), max_turns=6, max_failures=8)
    assert result.reason == "max_steps" and result.llm_turns == 6 and env.execute_calls == 1


def test_blocker_detection_agrees_with_occupancy_on_traced_states(tmp_path):
    for task in (T, SWAP, NO_SUBGOAL, FREE_TARGET):
        _, _, events, _ = run(tmp_path / task.name / str(len(task.goal)), DependencyAwarePolicy(task.goal), task=task)
        for r in of(events, "tool_result"):
            s = WorldState.model_validate(r["world_state"])
            for g in task.goal:
                if g.holds(s) or s.gripper.holding not in (None, g.obj):
                    continue
                d = (s.objects[g.region].pose.x, s.objects[g.region].pose.y)
                assert (_blocker(s, g) is None) == (placement_conflict(s, g.obj, *d) is None)


# ------------------------------------------------------------------ compatibility

def test_single_object_task_matches_existing_policy(tmp_path):
    task = region_pick_place_task()
    new = run(tmp_path / "new", DependencyAwarePolicy(task.goal), task=task)
    old = run(tmp_path / "old", PickPlacePolicyLLM("cube_red", "target"), task=task)
    assert physical(new[2]) == physical(old[2]) and new[0].reason == old[0].reason == "goal_met"
    assert new[1].execute_calls == 2


def test_constructor_rejects_unsupported_subgoals():
    with pytest.raises(ValueError):
        DependencyAwarePolicy([On(obj="cube_red", support="plate")])
    with pytest.raises(ValueError):
        NaivePolicy([region_goal("cube_red", "target"), region_goal("cube_blue", "target")])


def test_staging_respects_workspace_margin():
    from robot_agent.policies import _staging_points
    pts = _staging_points()
    assert all(WORKSPACE[0] < x < WORKSPACE[1] and WORKSPACE[2] < y < WORKSPACE[3] for x, y in pts)
    assert len(pts) == 19 * 19 and math.isclose(min(x for x, _ in pts), -0.45)


# ------------------------------------------------------------------ metric logic on hand-built traces

def _ev(seq, event, **kw):
    return {"seq": seq, "event": event, **kw}


def test_rejection_grouping_requires_adjacency_and_tolerates_unknown_regions():
    s0 = MockTabletopEnv().reset(T).model_dump(mode="json")
    held = copy.deepcopy(s0)
    held["gripper"]["holding"] = "cube_red"
    rej = {"code": "DESTINATION_OCCUPIED", "detail": "x"}
    ok_obs = _ev(0, "tool_result", call_id="o", ok=True, error=None, world_state=s0)
    events = [
        _ev(1, "llm_turn", tool_calls=[{"id": "a", "name": "place", "args": {"x": 0.05, "y": 0.3}}]),
        _ev(2, "tool_result", call_id="a", ok=False, error=rej, world_state=held),
        _ev(3, "llm_turn", tool_calls=[{"id": "b", "name": "observe", "args": {}}]),
        _ev(4, "tool_result", call_id="b", ok=True, error=None, world_state=held),
        _ev(5, "llm_turn", tool_calls=[{"id": "c", "name": "place", "args": {"x": 0.05, "y": 0.3}}]),
        _ev(6, "tool_result", call_id="c", ok=False, error=rej, world_state=held),
        _ev(7, "llm_turn", tool_calls=[{"id": "d", "name": "place", "args": {"x": 0.05, "y": 0.3}}]),
        _ev(8, "tool_result", call_id="d", ok=False, error=rej, world_state=held),
        _ev(9, "llm_turn", tool_calls=[{"id": "e", "name": "place", "args": {"x": 0.40, "y": 0.40}}]),
        _ev(10, "tool_result", call_id="e", ok=False, error=rej, world_state=held),
        _ev(11, "episode_end", result={"success": False, "reason": "failure_budget", "llm_turns": 5},
            unmet=[], state=held),
    ]
    m = metrics([ok_obs] + events, T, optimal=4)
    # a and c are identical but separated by an observe: two plans; c, d adjacent: one; e: outside regions.
    assert m["rejected_plans"] == 3 and m["rejected_plans_unattributed"] == 1
    assert m["reactive_recoveries"] == 0 and m["unnecessary_actions"] is None
