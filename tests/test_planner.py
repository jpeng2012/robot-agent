"""F012b-2: symbolic dependency planner (mock backend). Plans are replayed through AgentLoop + MockTabletopEnv."""

import itertools

import pytest

from robot_agent.env import MockTabletopEnv
from robot_agent.goals import InRegion, On, TaskSpec, check_goal
from robot_agent.llm import AssistantTurn, ScriptedLLM, call
from robot_agent.loop import AgentLoop
from robot_agent.planner import Plan, PlanningFailure, PlanningFailureCode as C, plan
from robot_agent import policies
from robot_agent.policies import _staging, _staging_points, decide
from robot_agent.world import REGION_HALF, WORKSPACE, WorldState

TARGET, PARKING, RED0 = (0.05, 0.30), (-0.25, 0.30), (0.10, -0.20)


def G(o, r):
    return InRegion(obj=o, region=r, half_x=REGION_HALF, half_y=REGION_HALF)


STD = [G("cube_red", "target"), G("cube_blue", "parking")]
CLEAR = {"target": ("target_region", *TARGET), "cube_blue": ("cube", *TARGET),
         "parking": ("target_region", *PARKING), "cube_red": ("cube", *RED0)}
SWAP = {**CLEAR, "cube_red": ("cube", *PARKING)}
FREE = {**CLEAR, "cube_blue": ("cube", 0.20, 0.0)}
NO_SUBGOAL = {"target": ("target_region", *TARGET), "cube_blue": ("cube", *TARGET), "cube_red": ("cube", *RED0)}
BLUE_PARKED = {**CLEAR, "cube_blue": ("cube", *PARKING)}
ALL_SAT = {**CLEAR, "cube_blue": ("cube", *PARKING), "cube_red": ("cube", *TARGET)}
R3 = {"r0": ("target_region", -0.3, 0.3), "r1": ("target_region", 0.0, 0.3), "r2": ("target_region", 0.3, 0.3)}
CHAIN3 = {**R3, "c0": ("cube", 0.3, -0.2), "c1": ("cube", 0.0, 0.3), "c2": ("cube", -0.3, 0.3)}  # F012b-1 RAND03
CHAIN3_GOALS = [G("c0", "r0"), G("c1", "r2"), G("c2", "r1")]
CYCLE3 = {**R3, "c0": ("cube", -0.3, 0.3), "c1": ("cube", 0.0, 0.3), "c2": ("cube", 0.3, 0.3)}
CYCLE3_GOALS = [G("c0", "r1"), G("c1", "r2"), G("c2", "r0")]


def task(layout, goals):
    return TaskSpec(name="planner_test", instruction="", layout=layout, goal=list(goals))


def state(layout, goals=()):
    return MockTabletopEnv().reset(task(layout, goals))


def to_turns(p: Plan):
    turns = []
    for st in p.steps:
        if st.op == "pick":
            turns.append([call("pick", object_id=st.object_id, x=st.x, y=st.y)])
        elif st.op == "place":
            turns.append([call("place", x=st.x, y=st.y)])
        else:
            turns.append([call("finish", summary="plan complete")])
    return turns


def execute(tmp_path, layout, goals, p, prefix=()):
    """Replay the plan through the production loop; returns (result, env, events)."""
    env = MockTabletopEnv()
    loop = AgentLoop(env, ScriptedLLM([[call("observe")], *prefix, *to_turns(p)]), trace_dir=tmp_path,
                     max_turns=40, max_failures=8)
    r = loop.run(task(layout, goals), seed=0)
    return r, env, loop.tracer.events


def failures(events):
    return [e for e in events if e["event"] == "tool_result" and not e["ok"]]


# ------------------------------------------------------------------ feasible plans

@pytest.mark.parametrize("name, layout, goals, n, staging", [
    ("clear", CLEAR, STD, 4, 0), ("free_target", FREE, STD, 4, 0), ("swap", SWAP, STD, 6, 1),
    ("no_subgoal", NO_SUBGOAL, [G("cube_red", "target")], 4, 1), ("blue_parked", BLUE_PARKED, STD, 2, 0),
    ("all_satisfied", ALL_SAT, STD, 0, 0), ("chain3", CHAIN3, CHAIN3_GOALS, 6, 0), ("cycle3", CYCLE3, CYCLE3_GOALS, 8, 1),
])
def test_plan_executes_without_rejection_and_meets_goals(tmp_path, name, layout, goals, n, staging):
    p = plan(goals, state(layout, goals))
    assert isinstance(p, Plan) and (p.physical_actions, p.staging_moves) == (n, staging)
    assert p.steps[-1].op == "finish"
    r, env, ev = execute(tmp_path, layout, goals, p)
    assert r.success and r.reason == "goal_met" and failures(ev) == []
    assert env.execute_calls == n


@pytest.mark.parametrize("layout, goals", [(CLEAR, STD), (SWAP, STD), (CHAIN3, CHAIN3_GOALS), (CYCLE3, CYCLE3_GOALS)])
def test_plan_is_invariant_to_goal_presentation_order(layout, goals):
    s = state(layout, goals)
    plans = {plan(list(perm), s).model_dump_json() for perm in itertools.permutations(goals)}
    assert len(plans) == 1


def test_red_first_and_blue_first_give_the_blocker_first_plan():
    s = state(CLEAR, STD)
    for goals in (STD, STD[::-1]):
        p = plan(goals, s)
        assert [(x.op, x.object_id, x.region) for x in p.steps[:4]] == [
            ("pick", "cube_blue", None), ("place", "cube_blue", "parking"),
            ("pick", "cube_red", None), ("place", "cube_red", "target")]


def test_chain_resolved_from_its_free_end_without_staging():
    # c0 -> r0 is blocked by c2, whose goal r1 is blocked by c1, whose goal r2 is free: c1, c2, c0
    p = plan(CHAIN3_GOALS, state(CHAIN3, CHAIN3_GOALS))
    assert [x.object_id for x in p.steps if x.op == "pick"] == ["c1", "c2", "c0"] and p.staging_moves == 0


def test_already_satisfied_goal_is_never_disturbed(tmp_path):
    p = plan(STD, state(BLUE_PARKED, STD))
    assert all(x.object_id != "cube_blue" for x in p.steps)
    r, _, ev = execute(tmp_path, BLUE_PARKED, STD, p)
    blue = [e["world_state"]["objects"]["cube_blue"]["pose"] for e in ev if e["event"] == "tool_result"]
    assert r.success and all((b["x"], b["y"]) == PARKING for b in blue)


def test_plan_from_holding_state(tmp_path):
    held = state(CLEAR, STD).model_copy(deep=True)  # the state after picking red at its pose (mock kinematics)
    held.gripper.holding = "cube_red"
    held.objects["cube_red"].pose.z = 0.1
    p = plan(STD, held)
    assert p.physical_actions == 5 and p.steps[0].op == "place" and p.steps[0].purpose == "stage"
    r, _, ev = execute(tmp_path, CLEAR, STD, p, prefix=[[call("pick", object_id="cube_red", x=RED0[0], y=RED0[1])]])
    assert r.success and failures(ev) == []


def test_goals_met_while_holding_unrelated_cube_needs_no_action():
    layout = {**ALL_SAT, "cube_green": ("cube", 0.3, -0.3)}
    s = state(layout, STD)
    s.gripper.holding = "cube_green"
    p = plan(STD, s)
    assert isinstance(p, Plan) and p.physical_actions == 0


def test_planner_never_mutates_state_or_env():
    env = MockTabletopEnv()
    s = env.reset(task(SWAP, STD))
    before, env_before = s.model_dump_json(), env.observe().model_dump_json()
    plan(STD, s)
    plan(STD, s, bounds=(-0.10, 0.10, 0.25, 0.35))
    assert s.model_dump_json() == before and env.observe().model_dump_json() == env_before
    assert env.execute_calls == 0


# ------------------------------------------------------------------ structured failures (supported input paths)

TIGHT = {"target": ("target_region", 0.05, 0.30), "parking": ("target_region", -0.05, 0.30),
         "cube_blue": ("cube", 0.05, 0.30), "cube_red": ("cube", -0.05, 0.30)}


@pytest.mark.parametrize("goals, layout, kwargs, code, sub", [
    ([G("cube_green", "target")], CLEAR, {}, C.INVALID_GOALS, "UNKNOWN_OBJECT"),
    ([G("target", "parking")], CLEAR, {}, C.INVALID_GOALS, "NOT_A_CUBE"),
    ([G("cube_red", "storage")], CLEAR, {}, C.INVALID_GOALS, "UNKNOWN_REGION"),
    ([G("cube_red", "cube_blue")], CLEAR, {}, C.INVALID_GOALS, "NOT_A_REGION"),
    ([G("cube_red", "target"), G("cube_red", "target")], CLEAR, {}, C.INVALID_GOALS, "DUPLICATE"),
    ([], CLEAR, {}, C.INVALID_GOALS, "EMPTY"),
    ([On(obj="cube_red", support="cube_blue")], CLEAR, {}, C.INVALID_GOALS, "UNSUPPORTED_PREDICATE"),
    ([InRegion(obj="cube_red", region="target", half_x=0.1, half_y=0.1)], CLEAR, {}, C.INVALID_GOALS,
     "UNSUPPORTED_REGION_SIZE"),
    ([G("cube_red", "target"), G("cube_blue", "target")], CLEAR, {}, C.INFEASIBLE_GOALS, "REGION_CONFLICT"),
    ([G("cube_red", "target"), G("cube_red", "parking")], CLEAR, {}, C.INFEASIBLE_GOALS, "OBJECT_CONFLICT"),
    (STD, CLEAR, {"bounds": (-0.6, 0.6, -0.5, 0.5)}, C.OUT_OF_BOUNDS, "BOUNDS_EXCEED_WORKSPACE"),
    (STD, CLEAR, {"bounds": (-0.10, 0.45, -0.45, 0.45)}, C.OUT_OF_BOUNDS, "GOAL_REGION"),
    (STD, CLEAR, {"bounds": (-0.30, 0.45, 0.0, 0.45)}, C.OUT_OF_BOUNDS, "PICK"),
    (STD, {**CLEAR, "parking": ("target_region", 0.10, 0.30)}, {}, C.UNSUPPORTED_GEOMETRY, "OVERLAPPING_REGIONS"),
    (STD, {**CLEAR, "plate": ("plate", 0.3, 0.0)}, {}, C.UNSUPPORTED_GEOMETRY, "OTHER_OBJECT_KINDS"),
    (STD, TIGHT, {"bounds": (-0.10, 0.10, 0.25, 0.35)}, C.NO_STAGING, None),
    (STD, CLEAR, {"max_physical": 3}, C.BOUND_EXCEEDED, None),
])
def test_structured_failures(goals, layout, kwargs, code, sub):
    f = plan(goals, state(layout, [g for g in goals if isinstance(g, InRegion)]), **kwargs)
    assert isinstance(f, PlanningFailure) and f.code == code and f.subcode == sub


def test_no_staging_is_a_bounds_effect_not_infeasibility():
    assert isinstance(plan(STD, state(TIGHT, STD)), Plan)  # same layout plans with the full workspace


def test_policy_stop_codes_are_structured():
    s = state(CLEAR, STD)
    t = decide(s, [G("cube_blue", "target"), G("cube_red", "target")], aware=True)  # blue satisfied, blocks red
    assert t.tool_calls == [] and t.meta["code"] == "CONFLICTING_GOALS" and t.text.startswith("conflicting goals")
    held = s.model_copy(deep=True)
    held.gripper.holding = "cube_red"
    assert decide(held, [G("cube_blue", "parking")], aware=False).meta["code"] == "NO_SUBGOAL_FOR_HELD"
    t = decide(state(TIGHT, STD).model_copy(update={"gripper": held.gripper}), STD, aware=True, bounds=(-0.10, 0.10, 0.25, 0.35))
    assert t.meta["code"] == "NO_STAGING"


def test_blocker_not_found_maps_to_structured_failure(monkeypatch):
    # unreachable in cube-only layouts; checks the mapping from the policy's structured stop only
    monkeypatch.setattr(policies, "_blocker", lambda s, g: None)
    f = plan([G("cube_red", "target")], state(CLEAR))
    assert isinstance(f, PlanningFailure) and f.code == C.BLOCKER_NOT_FOUND


# ------------------------------------------------------------------ shared staging grid

def test_staging_defaults_equal_workspace_and_bounds_restrict():
    s = state(SWAP, STD)
    held = s.model_copy(deep=True)
    held.gripper.holding = "cube_red"
    assert _staging_points() == _staging_points(WORKSPACE)
    assert _staging(held, "cube_red") == _staging(held, "cube_red", WORKSPACE)
    b = (-0.45, 0.0, 0.0, 0.45)
    xy = _staging(held, "cube_red", b)
    assert xy is not None and b[0] <= xy[0] <= b[1] and b[2] <= xy[1] <= b[3]
    assert _staging_points((0.0, 0.05, 0.0, 0.05)) == []


# ------------------------------------------------------------------ review follow-ups: edges, tie-breaks, internal faults

def real_held_state(tmp_path, layout, goals, cube):
    """The state AgentLoop delivers after a real pick (env kinematics, not hand-built)."""
    env = MockTabletopEnv()
    p = env.reset(task(layout, goals)).objects[cube].pose
    loop = AgentLoop(env, ScriptedLLM([[call("observe")], [call("pick", object_id=cube, x=p.x, y=p.y)]]),
                     trace_dir=tmp_path, max_turns=2)
    loop.run(task(layout, goals), seed=0)
    last = [e for e in loop.tracer.events if e["event"] == "tool_result"][-1]
    return WorldState.model_validate(last["world_state"])


def test_plan_from_real_held_state(tmp_path):
    s = real_held_state(tmp_path / "pick", CLEAR, STD, "cube_red")
    assert s.gripper.holding == "cube_red" and s.objects["cube_red"].pose.z > 0
    p = plan(STD, s)
    assert p.physical_actions == 5 and (p.steps[0].op, p.steps[0].purpose) == ("place", "stage")
    r, _, ev = execute(tmp_path / "run", CLEAR, STD, p, prefix=[[call("pick", object_id="cube_red", x=RED0[0], y=RED0[1])]])
    assert r.success and failures(ev) == []


@pytest.mark.parametrize("layout, goals, k", [(CLEAR, STD, 4), (ALL_SAT, STD, 0), (CYCLE3, CYCLE3_GOALS, 8)])
def test_bound_allows_exactly_the_needed_actions(layout, goals, k):
    s = state(layout, goals)
    assert isinstance(plan(goals, s, max_physical=k), Plan)  # finish is checked before the bound
    if k:
        f = plan(goals, s, max_physical=k - 1)
        assert f.code == C.BOUND_EXCEEDED and len(f.partial_steps) == k - 1


def test_default_bound_is_four_actions_per_cube(monkeypatch):
    s = state(CLEAR, STD)
    # a never-finishing policy, capped so a missing bound fails the test instead of hanging it
    loop = iter(itertools.islice(itertools.cycle([
        AssistantTurn(tool_calls=[call("pick", object_id="cube_red", x=RED0[0], y=RED0[1])]),
        AssistantTurn(tool_calls=[call("place", x=RED0[0], y=RED0[1])])]), 100))
    monkeypatch.setattr(policies, "decide", lambda *a, **k: next(loop).model_copy(update={"meta": {"mode": "goal"}}))
    f = plan(STD, s)
    assert f.code == C.BOUND_EXCEEDED and len(f.partial_steps) == 4 * 2


def test_tie_break_is_canonical_order():
    # both destinations free: the canonical order (cube_blue < cube_red) decides
    p = plan(STD[::-1], state(FREE, STD))
    assert [x.object_id for x in p.steps if x.op == "pick"] == ["cube_blue", "cube_red"]


def test_blocker_without_goal_ranks_before_cycles():
    # c3 -> rA is blocked by goal-less x; c1/c2 form a 2-cycle over rB/rC. Rule 2 (x) must precede rule 3 (cycle).
    layout = {"rA": ("target_region", -0.3, 0.3), "rB": ("target_region", 0.0, 0.3), "rC": ("target_region", 0.3, 0.3),
              "x": ("cube", -0.3, 0.3), "c3": ("cube", -0.3, -0.2), "c1": ("cube", 0.3, 0.3), "c2": ("cube", 0.0, 0.3)}
    goals = [G("c1", "rB"), G("c2", "rC"), G("c3", "rA")]
    p = plan(goals, state(layout, goals))
    assert isinstance(p, Plan) and p.steps[0].object_id == "x" and p.physical_actions == 10 and p.staging_moves == 2


@pytest.mark.parametrize("dx, ok", [(0.09, True), (0.089, False)])
def test_region_overlap_boundary(dx, ok):
    layout = {**CLEAR, "parking": ("target_region", TARGET[0] + dx, TARGET[1])}
    r = plan(STD, state(layout, STD))
    assert isinstance(r, Plan) if ok else r.subcode == "OVERLAPPING_REGIONS"


def test_goal_region_centre_on_bounds_edge_is_inside():
    assert isinstance(plan(STD, state(CLEAR, STD), bounds=(PARKING[0], 0.45, -0.45, 0.45)), Plan)


def test_pick_out_of_bounds_reports_partial_steps():
    f = plan(STD, state(CLEAR, STD), bounds=(-0.30, 0.45, 0.0, 0.45))
    assert f.subcode == "PICK" and [(x.op, x.object_id) for x in f.partial_steps] == [
        ("pick", "cube_blue"), ("place", "cube_blue")]


def test_unknown_policy_stop_is_internal_not_a_crash(monkeypatch):
    monkeypatch.setattr(policies, "decide", lambda *a, **k: AssistantTurn(text="x", meta={"mode": "stop",
                                                                                        "code": "NO_SUBGOAL_FOR_HELD"}))
    f = plan(STD, state(CLEAR, STD))
    assert f.code == C.PLANNER_INTERNAL and f.subcode == "POLICY_STOP:NO_SUBGOAL_FOR_HELD"


def test_conflicting_place_from_policy_is_internal(tmp_path, monkeypatch):
    s = real_held_state(tmp_path, CLEAR, STD, "cube_red")
    monkeypatch.setattr(policies, "decide", lambda *a, **k: AssistantTurn(
        tool_calls=[call("place", x=TARGET[0], y=TARGET[1])], meta={"mode": "goal"}))
    f = plan(STD, s)
    assert f.code == C.PLANNER_INTERNAL and f.subcode == "POLICY_CONFLICT"


def test_inconsistent_held_object_is_invalid_state():
    s = state(CLEAR, STD)
    s.gripper.holding = "target"
    assert plan(STD, s).code == C.INVALID_STATE
    with pytest.raises(ValueError):
        plan(STD, state(CLEAR, STD), max_physical=-1)
