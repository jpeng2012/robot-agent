"""F015: two-cube robosuite scene driven by the symbolic planner + PlanExecutor through AgentLoop.

Simulator tests need ROBOT_AGENT_ROBOSUITE_PYTHON; the tolerance-profile tests always run.
"""

import os

import pytest

from robot_agent.executor import MOCK_TOLERANCES, PHYSICAL_TOLERANCES, PlanExecutor, deviation
from robot_agent.goals import InRegion, TaskSpec, check_goal
from robot_agent.loop import AgentLoop
from robot_agent.skills import PickArgs, PlaceArgs
from robot_agent.world import REGION_HALF, GripperObs, ObjectObs, Pose, WorldState

TARGET, PARKING, RED0 = (0.07, 0.20), (-0.08, 0.20), (0.05, -0.05)
MOVED = (-0.05, -0.08)  # valid, free, reachable


def G(o, r):
    return InRegion(obj=o, region=r, half_x=REGION_HALF, half_y=REGION_HALF)


TASK = TaskSpec(name="f015_two_cube_clear", instruction="", layout={
    "target": ("target_region", *TARGET), "cube_blue": ("cube", *TARGET),
    "parking": ("target_region", *PARKING), "cube_red": ("cube", *RED0)},
    goal=[G("cube_red", "target"), G("cube_blue", "parking")])


# ------------------------------------------------------------------ physical tolerance profile (no simulator)

def ws(red=(0.05, -0.05, 0.0), blue=(0.07, 0.2, 0.0), holding=None):
    obj = lambda k, kind, p: ObjectObs(id=k, kind=kind, pose=Pose(x=p[0], y=p[1], z=p[2]))  # noqa: E731
    return WorldState(t=0, step=0, gripper=GripperObs(pose=Pose(x=0, y=0, z=0.3), holding=holding), objects={
        "cube_red": obj("cube_red", "cube", red), "cube_blue": obj("cube_blue", "cube", blue),
        "target": obj("target", "target_region", (*TARGET, 0.0)), "parking": obj("parking", "target_region", (*PARKING, 0.0))})


def test_default_tolerances_are_the_mock_profile():
    assert PlanExecutor([G("cube_red", "target")]).tolerances == MOCK_TOLERANCES
    assert (MOCK_TOLERANCES.xy, MOCK_TOLERANCES.z, MOCK_TOLERANCES.held_pose) == (0.02, 0.02, True)


def test_physical_profile_still_compares_other_objects_while_holding():
    expected = ws(red=(0.05, -0.05, 0.10), holding="cube_red")
    moved_blue = ws(red=(0.05, -0.05, 0.10), blue=(0.07 + 0.03, 0.2, 0.0), holding="cube_red")
    assert "cube_blue" in deviation(expected, moved_blue, PHYSICAL_TOLERANCES)


def test_physical_profile_ignores_carried_pose_but_not_identity():
    expected = ws(red=(0.05, -0.05, 0.10), holding="cube_red")  # simulate(): nominal carry pose
    actual = ws(red=(0.058, -0.044, 0.128), holding="cube_red")  # grasp offset, higher lift
    assert deviation(expected, actual, PHYSICAL_TOLERANCES) is None
    assert deviation(expected, actual, MOCK_TOLERANCES) is not None  # mock behaviour unchanged
    assert "holding" in deviation(expected, ws(holding="cube_blue"), PHYSICAL_TOLERANCES)
    assert "holding" in deviation(expected, ws(holding=None), PHYSICAL_TOLERANCES)


@pytest.mark.parametrize("blue, ok", [
    ((0.07 + 0.019, 0.2, 0.0), True), ((0.07 + 0.021, 0.2, 0.0), False),   # resting xy: PLACE_TOL
    ((0.07, 0.2, 0.024), True), ((0.07, 0.2, 0.026), False),               # resting z: 0.025
])
def test_physical_profile_resting_tolerances(blue, ok):
    assert (deviation(ws(), ws(blue=blue), PHYSICAL_TOLERANCES) is None) is ok


# ------------------------------------------------------------------ F016 staging clearance (mock state, no simulator)

REACH = (-0.15, 0.15, -0.12, 0.36)
SWAP_LAYOUT = {**TASK.layout, "cube_red": ("cube", *PARKING)}


def mock_state(layout):
    from robot_agent.env import MockTabletopEnv
    return MockTabletopEnv().reset(TaskSpec(name="m", instruction="", layout=layout, goal=TASK.goal))


def test_staging_clearance_is_threaded_through_the_planner():
    from robot_agent.planner import Plan, plan
    s = mock_state(SWAP_LAYOUT)
    stage = lambda p: next((x.x, x.y) for x in p.steps if x.purpose == "stage")  # noqa: E731
    assert stage(plan(TASK.goal, s, bounds=REACH)) == stage(plan(TASK.goal, s, bounds=REACH, clearance=0.0)) == (-0.1, 0.13)
    p = plan(TASK.goal, s, bounds=REACH, clearance=0.09)  # physical clearance; staging keeps 0.09 + 0.02
    assert isinstance(p, Plan) and p.physical_actions == 6 and stage(p) == (-0.1, 0.08)
    assert all(((stage(p)[0] - x) ** 2 + (stage(p)[1] - y) ** 2) ** 0.5 >= 0.11 for x, y in (TARGET, PARKING))
    f = plan(TASK.goal, s, bounds=(-0.13, 0.13, 0.12, 0.28), clearance=0.09)
    assert f.code == "NO_STAGING"


def test_executor_prechecks_place_clearance():
    from robot_agent.planner import PlanStep, simulate
    s = mock_state(TASK.layout)
    held = simulate(s, PlanStep(op="pick", purpose="goal", object_id="cube_red", x=RED0[0], y=RED0[1]))
    near_blue = PlanStep(op="place", purpose="stage", object_id="cube_red", x=TARGET[0] - 0.085, y=TARGET[1])
    assert "within 0.090 m of cube_blue" in PlanExecutor(TASK.goal, clearance=0.09)._ground(held, near_blue)
    tc, _ = PlanExecutor(TASK.goal)._ground(held, near_blue)  # mock default: no physical clearance
    assert tc.name == "place"
    just_clear = PlanStep(op="place", purpose="stage", object_id="cube_red", x=TARGET[0] - 0.095, y=TARGET[1])
    tc, _ = PlanExecutor(TASK.goal, clearance=0.09, staging_margin=0.06)._ground(held, just_clear)
    assert tc.name == "place"  # the precheck uses the physical clearance, never clearance + staging margin



# ------------------------------------------------------------------ F018 clearance-aware goals (mock state, no simulator)

NEAR_PARKING = (-0.08, 0.13)  # 0.07 m from the parking centre, outside the region: a physical blocker


def held_c3_state():
    """F017 C3 after the disturbance: blue held (picked from target), staged red pushed next to parking."""
    from robot_agent.planner import PlanStep, simulate
    s = mock_state({**SWAP_LAYOUT, "cube_red": ("cube", *NEAR_PARKING)})
    return simulate(s, PlanStep(op="pick", purpose="goal", object_id="cube_blue", x=TARGET[0], y=TARGET[1]))


def steps(p):
    return [(x.op, x.object_id, x.purpose) for x in p.steps]


def test_held_cube_with_crowded_goal_is_staged_then_blocker_cleared():
    from robot_agent.planner import Plan, plan
    s = held_c3_state()
    old = plan(TASK.goal, s, bounds=REACH)  # clearance disabled: the crowded goal place is planned directly
    assert steps(old)[0] == ("place", "cube_blue", "goal")
    p = plan(TASK.goal, s, bounds=REACH, clearance=0.09)
    assert isinstance(p, Plan) and p.physical_actions == 5 and steps(p) == [
        ("place", "cube_blue", "stage"), ("pick", "cube_red", "goal"), ("place", "cube_red", "goal"),
        ("pick", "cube_blue", "goal"), ("place", "cube_blue", "goal"), ("finish", None, "finish")]
    st = p.steps[0]
    for x, y in (TARGET, PARKING, NEAR_PARKING):
        assert ((st.x - x) ** 2 + (st.y - y) ** 2) ** 0.5 >= 0.09 + 0.02
    f = plan(TASK.goal, s, bounds=(-0.13, 0.13, 0.12, 0.28), clearance=0.09)  # nowhere safe to put blue down
    assert f.code == "NO_STAGING"


def test_crowding_cube_without_goal_is_cleared_before_the_goal_place():
    from robot_agent.planner import plan
    lay = {"target": ("target_region", *TARGET), "parking": ("target_region", *PARKING),
           "cube_blue": ("cube", 0.05, -0.05), "cube_red": ("cube", *NEAR_PARKING)}
    goals = [TASK.goal[1]]  # blue -> parking only; red has no goal but crowds parking
    s = mock_state(lay)
    assert plan(goals, s, bounds=REACH).physical_actions == 2
    p = plan(goals, s, bounds=REACH, clearance=0.09)
    assert steps(p)[:2] == [("pick", "cube_red", "clear_blocker"), ("place", "cube_red", "stage")]
    assert p.physical_actions == 4


def test_satisfied_cube_crowding_another_goal_is_a_conflict():
    from robot_agent.planner import plan
    lay = {"a": ("target_region", 0.0, 0.2), "b": ("target_region", 0.10, 0.2),
           "cube_red": ("cube", 0.04, 0.2), "cube_blue": ("cube", 0.0, -0.05)}  # red inside a, 0.06 from b's centre
    goals = [G("cube_red", "a"), G("cube_blue", "b")]
    s = mock_state(lay)
    assert plan(goals, s, bounds=REACH).physical_actions == 2
    assert plan(goals, s, bounds=REACH, clearance=0.09).code == "CONFLICTING_GOALS"


def test_staging_margin_is_separate_from_the_goal_clearance():
    from robot_agent.planner import plan
    a5 = {"target": ("target_region", 0.05, 0.2), "parking": ("target_region", -0.07, 0.2),
          "cube_blue": ("cube", 0.05, 0.2), "cube_red": ("cube", -0.07, 0.2)}  # region centres 0.12 apart
    s = mock_state(a5)
    p = plan(TASK.goal, s, bounds=REACH, clearance=0.09, staging_margin=0.06)  # wide staging, goal check stays 0.09
    st = next(x for x in p.steps if x.purpose == "stage")
    assert p.physical_actions == 6 and min(((st.x - x) ** 2 + (st.y - y) ** 2) ** 0.5 for x, y in ((0.05, 0.2), (-0.07, 0.2))) >= 0.15


sim = pytest.mark.skipif(not os.environ.get("ROBOT_AGENT_ROBOSUITE_PYTHON"), reason="ROBOT_AGENT_ROBOSUITE_PYTHON not set")


@pytest.fixture(scope="module")
def env():
    from robot_agent.robosuite_env import RobosuiteEnv

    with RobosuiteEnv(scene="two_cube") as e:
        yield e


def run(env, tmp_path, perturb=None, seed=0, task=TASK, clearance=0.0, bounds=None, transit_budget=None,
        staging_margin=0.02):
    env.test_perturb_at, env.test_transit_budget = dict(perturb or {}), transit_budget
    try:
        env.reset(task, seed=seed)  # backend_info (reach) is per reset
        ex = PlanExecutor(task.goal, bounds=bounds or tuple(env.backend_info["reach"]), tolerances=PHYSICAL_TOLERANCES,
                          clearance=clearance, staging_margin=staging_margin)
        loop = AgentLoop(env, ex, trace_dir=tmp_path, max_turns=20)
        r = loop.run(task, seed=seed)
    finally:
        env.test_perturb_at, env.test_transit_budget = {}, None
    return r, loop.tracer.events, ex


def of(ev, name):
    return [e for e in ev if e["event"] == name]


def outcomes(ev):
    calls = {e["call_id"]: e for e in of(ev, "tool_call")}
    return [(calls[e["call_id"]]["name"], calls[e["call_id"]]["args"], e) for e in of(ev, "env_outcome")]


def own_goal_check(state):
    """Independent of InRegion: box test on the observed simulator poses."""
    o, held = state["objects"], state["gripper"]["holding"]
    return [f"{c}->{r}" for c, r in (("cube_red", "target"), ("cube_blue", "parking"))
            if not (held != c and abs(o[c]["pose"]["x"] - o[r]["pose"]["x"]) < REGION_HALF
                    and abs(o[c]["pose"]["y"] - o[r]["pose"]["y"]) < REGION_HALF and abs(o[c]["pose"]["z"]) <= 0.025)]


def inbox(state, c, r):
    o = state["objects"]
    return (state["gripper"]["holding"] != c and abs(o[c]["pose"]["x"] - o[r]["pose"]["x"]) < REGION_HALF
            and abs(o[c]["pose"]["y"] - o[r]["pose"]["y"]) < REGION_HALF and abs(o[c]["pose"]["z"]) <= 0.025)


def delivered_before(ev, seq):
    return [x for x in of(ev, "tool_result") if x["seq"] < seq][-1]["world_state"]


def common_checks(r, ev, env, blocker_first=True):
    assert r.success and r.reason == "goal_met", r
    end = of(ev, "episode_end")[-1]
    assert own_goal_check(end["state"]) == [] and end["unmet"] == []
    last_place = [e for name, _, e in outcomes(ev) if name == "place"][-1]
    assert last_place["outcome"]["info"]["in_region"] == {"cube_red": "target", "cube_blue": "parking"}  # worker's own check
    assert all(g["ok"] for g in of(ev, "grounding_check"))
    assert not any((e["error"] or {}).get("code") == "UNGROUNDED_ACTION" for e in of(ev, "tool_result"))
    # Blocker first, from the delivered simulator states: when red first rests in the target, blue is in parking.
    states = [x["world_state"] for x in of(ev, "tool_result")]
    j = next(i for i, s in enumerate(states) if inbox(s, "cube_red", "target"))
    assert inbox(states[0], "cube_blue", "target") and (inbox(states[j], "cube_blue", "parking") or not blocker_first)
    # The cube not being handled did not move (state delivered before the call vs simulator state after it).
    for name, args, e in outcomes(ev):
        before = delivered_before(ev, e["seq"])
        handled = args["object_id"] if name == "pick" else before["gripper"]["holding"]
        moved_by_test = (e["outcome"]["info"].get("test_perturbation") or {}).get("object_id")
        for c in ("cube_red", "cube_blue"):
            if c not in (handled, moved_by_test):
                a, b = before["objects"][c]["pose"], e["state_after"]["objects"][c]["pose"]
                assert max(abs(a[k] - b[k]) for k in "xyz") < 0.005, (name, c, a, b)
    # Every attempted pick started its approach above the commanded point at retreat height.
    z_min = env.backend_info["cubes"]["cube_red"]["z_ref"] + 0.12 - 0.01  # gripper ref + RETREAT_Z - margin
    for name, args, e in outcomes(ev):
        if name == "pick" and e["outcome"]["info"].get("control_steps", 0) > 0:
            info = e["outcome"]["info"]
            assert info["transit_converged"] is True
            ax, ay, az = info["approach_start_eef_world"]
            assert ((ax - args["x"]) ** 2 + (ay - args["y"]) ** 2) ** 0.5 < 0.011 and az >= z_min, info


@sim
def test_reset_places_two_identified_cubes_and_regions(env):
    s = env.reset(TASK, seed=0)
    for k, (kind, x, y) in TASK.layout.items():
        p = s.objects[k].pose
        assert s.objects[k].kind == kind and abs(p.x - x) < 1e-6 and abs(p.y - y) < 1e-6 and abs(p.z) < 0.005
    info = env.backend_info
    assert info["scene"] == "two_cube" and info["init"]["cube_ids"] == ["cube_red", "cube_blue"]
    assert info["reach"] == [-0.15, 0.15, -0.12, 0.36] and info["init"]["barrier_pos"][1] > 0.4
    assert s.gripper.holding is None


@sim
def test_two_cube_scene_rejects_single_cube_layout(env):
    from robot_agent.goals import region_pick_place_task
    with pytest.raises(RuntimeError, match="exactly two cubes"):
        env.reset(region_pick_place_task(), seed=0)


@sim
def test_nominal_clear_task_moves_blocker_first_and_meets_goals(env, tmp_path):
    r, ev, ex = run(env, tmp_path)
    common_checks(r, ev, env)
    assert r.failures == 0 and r.skills_executed == 4 and ex.replans == 0 and ex.plan_id == 0
    assert [(n, a.get("object_id")) for n, a, _ in outcomes(ev)] == [
        ("pick", "cube_blue"), ("place", None), ("pick", "cube_red"), ("place", None)]


@sim
def test_same_seed_reproduces_the_episode(env, tmp_path):
    runs = [run(env, tmp_path / str(i)) for i in range(2)]
    assert all(r.success for r, _, _ in runs)
    seqs = [[(n, a, e["outcome"]["ok"], e["outcome"]["info"]["cube_poses_world"], e["outcome"]["info"]["eef_pose_world"])
             for n, a, e in outcomes(ev)] for _, ev, _ in runs]
    finals = [of(ev, "episode_end")[-1]["state"] for _, ev, _ in runs]
    assert len(seqs[0]) == 4 and seqs[0] == seqs[1] and finals[0] == finals[1]  # measured: exactly equal on this build
    # the seed matters: robot init-joint noise differs for another seed (same process, same worker)
    q = [(env.reset(TASK, seed=s), env.backend_info["init"]["robot_init_qpos"])[1] for s in (0, 1, 0)]
    assert q[0] != q[1] and q[0] == q[2]


@sim
def test_perturbed_pick_fails_physically_then_replans_and_succeeds(env, tmp_path):
    r, ev, ex = run(env, tmp_path, perturb={2: ("cube_red", *MOVED)})
    common_checks(r, ev, env)
    out = outcomes(ev)
    name, args, failed = out[2]
    assert (name, args) == ("pick", {"object_id": "cube_red", "x": RED0[0], "y": RED0[1]})  # last delivered pose
    assert failed["outcome"]["info"]["test_perturbation"] == {"execute_index": 2, "object_id": "cube_red", "to": list(MOVED)}
    assert failed["outcome"]["failure"]["code"] == "OBJECT_NOT_AT_POSE"
    assert failed["outcome"]["info"]["termination"] == "no_grasp_contact"  # the gripper closed at the stale pose
    gc = next(g for g in of(ev, "grounding_check") if g["call_id"] == failed["call_id"])
    assert gc["ok"]  # the move happened after validation and grounding, inside the worker
    res = next(e for e in of(ev, "tool_result") if e["call_id"] == failed["call_id"])
    p = res["world_state"]["objects"]["cube_red"]["pose"]
    assert abs(p["x"] - MOVED[0]) < 0.005 and abs(p["y"] - MOVED[1]) < 0.005  # new authoritative state delivered
    [rr] = [t["meta"]["replan_reason"] for t in of(ev, "llm_turn") if "replan_reason" in t["meta"]]
    assert (rr["trigger"], rr["code"], rr["step_index"]) == ("EXECUTION_FAILURE", "OBJECT_NOT_AT_POSE", 2)
    assert out[3][0] == "pick" and (out[3][1]["x"], out[3][1]["y"]) == (p["x"], p["y"])  # re-grounded pick
    assert ex.replans == 1 and r.failures == 1 and r.skills_executed == 5


@sim
@pytest.mark.parametrize("schedule, first, why", [
    ({0: ("cube_red", TARGET[0] + 0.03, TARGET[1])}, None, "overlaps cube_blue"),
    ({0: ("cube_red", TARGET[0], TARGET[1] + 0.044)}, None, "already contains cube_blue"),  # no overlap, region full
    ({0: ("cube_red", 0.2, 0.0)}, None, "outside REACH"),
    ({0: ("cube_green", 0.0, 0.0)}, None, "unknown cube"),
    ({1: ("cube_red", 0.0, 0.0)}, "pick_red", "is held"),
])
def test_invalid_perturbation_is_refused_without_changing_the_scene(env, schedule, first, why):
    env.test_perturb_at = schedule
    try:
        before = env.reset(TASK, seed=0)
        if first == "pick_red":
            assert env.execute("pick", PickArgs(object_id="cube_red", x=RED0[0], y=RED0[1])).ok
            before = env.observe()
        with pytest.raises(RuntimeError, match=why):
            if first == "pick_red":
                env.execute("place", PlaceArgs(x=-0.025, y=0.0))
            else:
                env.execute("pick", PickArgs(object_id="cube_blue", x=TARGET[0], y=TARGET[1]))
        assert env.observe() == before  # same poses, holding, step and time: nothing applied
    finally:
        env.test_perturb_at = {}


@sim
def test_place_clearance_threshold(env):
    env.reset(TASK, seed=0)
    assert env.execute("pick", PickArgs(object_id="cube_red", x=RED0[0], y=RED0[1])).ok
    before = env.observe()
    near_blue = (TARGET[0] - 0.07, TARGET[1] - 0.05)  # 0.086 m from blue: no overlap, outside both regions
    out = env.execute("place", PlaceArgs(x=near_blue[0], y=near_blue[1]))
    assert not out.ok and out.failure.code == "DESTINATION_OCCUPIED" and "finger clearance" in out.failure.detail
    after = env.observe()
    assert (after.objects, after.gripper, after.t) == (before.objects, before.gripper, before.t)  # no motion
    ok_point = (TARGET[0] - 0.095, TARGET[1])  # 0.095 m from blue, outside both regions: accepted
    out = env.execute("place", PlaceArgs(x=ok_point[0], y=ok_point[1]))
    assert out.ok, out
    p = env.observe().objects["cube_red"].pose
    assert ((p.x - ok_point[0]) ** 2 + (p.y - ok_point[1]) ** 2) ** 0.5 <= 0.02 and env.observe().gripper.holding is None


@sim
@pytest.mark.parametrize("layout_change, why", [
    ({"parking": ("target_region", TARGET[0] - 0.10, TARGET[1])}, "region centres closer"),
    ({"cube_red": ("cube", TARGET[0], TARGET[1] - 0.10)}, "finger clearance"),
])
def test_reset_refuses_layouts_without_finger_clearance(env, layout_change, why):
    task = TASK.model_copy(update={"layout": {**TASK.layout, **layout_change}})
    with pytest.raises(RuntimeError, match=why):
        env.reset(task, seed=0)


# ------------------------------------------------------------------ F016: SWAP with physical staging

SWAP = TASK.model_copy(update={"name": "f016_two_cube_swap",
                               "layout": {**TASK.layout, "cube_red": ("cube", *PARKING)}})


def clearance(env):
    return env.backend_info["place_clearance"]  # worker finger clearance; staging adds staging_margin (0.02)


def places(ev):
    return [(a, e) for n, a, e in outcomes(ev) if n == "place"]


@sim
def test_swap_stages_with_physical_clearance_and_meets_goals(env, tmp_path):
    env.reset(SWAP, seed=0)
    c = clearance(env)
    r, ev, ex = run(env, tmp_path, task=SWAP, clearance=c)
    common_checks(r, ev, env)
    assert r.failures == 0 and r.skills_executed == 6 and ex.replans == 0 and ex.plan_id == 0
    assert [(n, a.get("object_id")) for n, a, _ in outcomes(ev)] == [
        ("pick", "cube_red"), ("place", None), ("pick", "cube_blue"), ("place", None), ("pick", "cube_red"), ("place", None)]
    assert all(e["outcome"]["ok"] for _, e in places(ev))  # zero refused (unsafe) place attempts
    plan = next(t["meta"]["plan"] for t in of(ev, "llm_turn") if "plan" in t["meta"])
    [stage] = [s for s in plan["steps"] if s["purpose"] == "stage"]
    regions = [TARGET, PARKING]
    assert all(((stage["x"] - x) ** 2 + (stage["y"] - y) ** 2) ** 0.5 >= c + 0.02 for x, y in regions)  # + staging margin
    lo = env.backend_info["reach"]
    assert lo[0] <= stage["x"] <= lo[1] and lo[2] <= stage["y"] <= lo[3]
    staged = of(ev, "tool_result")[2]["world_state"]["objects"]["cube_red"]["pose"]  # after the staging place
    assert ((staged["x"] - stage["x"]) ** 2 + (staged["y"] - stage["y"]) ** 2) ** 0.5 <= 0.02


@sim
def test_swap_without_clearance_is_refused_by_the_worker_and_bounded(env, tmp_path):
    # Control: mock staging (clearance 0) puts red 0.073 m from the parking centre; the worker refuses blue's
    # goal place there, and the executor stops on livelock instead of retrying indefinitely.
    r, ev, ex = run(env, tmp_path, task=SWAP, clearance=0.0)
    assert not r.success and r.reason == "llm_stopped"
    refused = [e for _, e in places(ev) if not e["outcome"]["ok"]]
    assert refused and all(e["outcome"]["failure"]["code"] == "DESTINATION_OCCUPIED" for e in refused)
    assert all(e["outcome"]["info"]["control_steps"] == 0 for e in refused) and len(refused) <= 2
    stop = [t["meta"] for t in of(ev, "llm_turn") if t["meta"].get("mode") == "stop"][-1]
    assert stop["executor_failure"]["code"] == "LIVELOCK_DETECTED"


@sim
def test_swap_same_seed_reproduces(env, tmp_path):
    env.reset(SWAP, seed=0)
    c = clearance(env)
    runs = [run(env, tmp_path / str(i), task=SWAP, clearance=c) for i in range(2)]
    assert all(r.success for r, _, _ in runs)
    seqs = [[(n, a, e["outcome"]["ok"], e["outcome"]["info"]["cube_poses_world"]) for n, a, e in outcomes(ev)]
            for _, ev, _ in runs]
    finals = [of(ev, "episode_end")[-1]["state"] for _, ev, _ in runs]
    assert len(seqs[0]) == 6 and seqs[0] == seqs[1] and finals[0] == finals[1]


@sim
def test_no_safe_staging_is_a_planning_failure_without_physical_commands(env, tmp_path):
    env.reset(SWAP, seed=0)
    tight = (-0.13, 0.13, 0.12, 0.28)  # every staging grid point is within the clearance of a region centre
    r, ev, ex = run(env, tmp_path, task=SWAP, clearance=clearance(env), bounds=tight)
    assert not r.success and r.reason == "llm_stopped" and of(ev, "env_outcome") == []
    stop = [t["meta"] for t in of(ev, "llm_turn") if t["meta"].get("mode") == "stop"][-1]
    assert stop["planning_failure"]["code"] == "NO_STAGING"


@sim
def test_disturbance_after_staging_is_detected_and_replanned(env, tmp_path):
    env.reset(SWAP, seed=0)
    moved = (0.05, -0.08)  # staged red is moved to another free, reachable, clear point during blue's pick
    r, ev, ex = run(env, tmp_path, task=SWAP, clearance=clearance(env), perturb={2: ("cube_red", *moved)})
    common_checks(r, ev, env)
    out = outcomes(ev)
    assert out[2][2]["outcome"]["ok"] and out[2][2]["outcome"]["info"]["test_perturbation"]["object_id"] == "cube_red"
    [rr] = [t["meta"]["replan_reason"] for t in of(ev, "llm_turn") if "replan_reason" in t["meta"]]
    assert (rr["trigger"], rr["step_index"]) == ("STATE_DEVIATION", 2) and "cube_red" in rr["detail"]
    picks = [a for n, a, _ in out if n == "pick" and a["object_id"] == "cube_red"]
    assert (picks[-1]["x"], picks[-1]["y"]) != (picks[0]["x"], picks[0]["y"])  # re-grounded at the moved pose
    assert abs(picks[-1]["x"] - moved[0]) < 0.005 and abs(picks[-1]["y"] - moved[1]) < 0.005
    assert ex.replans == 1 and r.failures == 0 and r.skills_executed == 6
    assert all(e["outcome"]["ok"] for _, e in places(ev))


@sim
def test_transit_failure_is_bounded_and_fail_closed(env, tmp_path):
    env.test_transit_budget = 0
    try:
        before = env.reset(TASK, seed=0)
        out = env.execute("pick", PickArgs(object_id="cube_blue", x=TARGET[0], y=TARGET[1]))
    finally:
        env.test_transit_budget = None
    info = out.info
    assert not out.ok and out.failure.code == "POSTCONDITION_FAILED" and "pick not attempted" in out.failure.detail
    assert (info["termination"], info["fsm_state"], info["transit_converged"]) == ("transit_budget", "TRANSIT", False)
    after = env.observe()
    assert after.gripper.holding is None
    for c in ("cube_red", "cube_blue"):
        a, b = before.objects[c].pose, after.objects[c].pose
        assert max(abs(a.x - b.x), abs(a.y - b.y), abs(a.z - b.z)) < 1e-3  # nothing was touched
    # Through AgentLoop the same refusal is bounded: one re-attempt, then an explicit executor stop.
    r, ev, ex = run(env, tmp_path, transit_budget=0)
    assert not r.success and r.reason == "llm_stopped" and len(of(ev, "env_outcome")) == 2
    stop = [t["meta"] for t in of(ev, "llm_turn") if t["meta"].get("mode") == "stop"][-1]
    assert stop["executor_failure"]["code"] == "LIVELOCK_DETECTED"


# ------------------------------------------------------------------ F018 replays of the F017 failures

@sim
def test_f017_c3_replay_recovers_from_the_held_blue_state(env, tmp_path):
    env.reset(SWAP, seed=0)
    r, ev, ex = run(env, tmp_path, task=SWAP, clearance=clearance(env), perturb={2: ("cube_red", *NEAR_PARKING)})
    common_checks(r, ev, env, blocker_first=False)  # recovery order: blue staged, red placed, then blue parked
    [rr] = [t["meta"]["replan_reason"] for t in of(ev, "llm_turn") if "replan_reason" in t["meta"]]
    assert (rr["trigger"], rr["step_index"]) == ("STATE_DEVIATION", 2)
    replanned = next(t["meta"]["plan"] for t in of(ev, "llm_turn") if t["meta"].get("plan_id") == 1 and "plan" in t["meta"])
    assert [(s["op"], s["object_id"], s["purpose"]) for s in replanned["steps"][:1]] == [("place", "cube_blue", "stage")]
    assert ex.replans == 1 and r.failures == 0 and r.skills_executed == 3 + 5
    assert all(e["outcome"]["ok"] for _, e in places(ev))  # no refused (unsafe) place


@sim
def test_f017_b015_replay_wide_staging_margin_keeps_goal_places_legal(env, tmp_path):
    a5 = SWAP.model_copy(update={"name": "f018_a5", "layout": {
        "target": ("target_region", 0.05, 0.20), "cube_blue": ("cube", 0.05, 0.20),
        "parking": ("target_region", -0.07, 0.20), "cube_red": ("cube", -0.07, 0.20)}})
    env.reset(a5, seed=5)
    r, ev, ex = run(env, tmp_path, task=a5, seed=5, clearance=clearance(env), staging_margin=0.06)
    assert r.success and r.failures == 0 and r.skills_executed == 6 and ex.replans == 0
    assert all(e["outcome"]["ok"] for _, e in places(ev))
