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


# ------------------------------------------------------------------ simulator

sim = pytest.mark.skipif(not os.environ.get("ROBOT_AGENT_ROBOSUITE_PYTHON"), reason="ROBOT_AGENT_ROBOSUITE_PYTHON not set")


@pytest.fixture(scope="module")
def env():
    from robot_agent.robosuite_env import RobosuiteEnv

    with RobosuiteEnv(scene="two_cube") as e:
        yield e


def run(env, tmp_path, perturb=None, seed=0):
    env.test_perturb_at = dict(perturb or {})
    try:
        env.reset(TASK, seed=seed)  # backend_info (reach) is per reset
        ex = PlanExecutor(TASK.goal, bounds=tuple(env.backend_info["reach"]), tolerances=PHYSICAL_TOLERANCES)
        loop = AgentLoop(env, ex, trace_dir=tmp_path, max_turns=20)
        r = loop.run(TASK, seed=seed)
    finally:
        env.test_perturb_at = {}
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


def common_checks(r, ev, env):
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
    assert inbox(states[j], "cube_blue", "parking") and inbox(states[0], "cube_blue", "target")
    # The cube not being handled did not move (state delivered before the call vs simulator state after it).
    for name, args, e in outcomes(ev):
        before = delivered_before(ev, e["seq"])
        handled = args["object_id"] if name == "pick" else before["gripper"]["holding"]
        for c in ("cube_red", "cube_blue"):
            if c != handled:
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
