"""F011a: clear an occupied target region before filling it (mock backend)."""

import pytest

from robot_agent.env import CARRY_Z, Faults, MockTabletopEnv
from robot_agent.goals import (InRegion, check_goal, clear_target_task, pick_place_task,
                               region_pick_place_task)
from robot_agent.llm import PickPlacePolicyLLM, ScriptedLLM, call
from robot_agent.loop import AgentLoop
from robot_agent.world import (CUBE_HALF, REGION_HALF, GripperObs, ObjectObs, Pose, WorldState,
                               placement_conflict)

TARGET, PARKING, RED0 = (0.05, 0.30), (-0.25, 0.30), (0.10, -0.20)
OBSERVE = [call("observe")]


def pick(oid, xy):
    return [call("pick", object_id=oid, x=xy[0], y=xy[1])]


def place(xy):
    return [call("place", x=xy[0], y=xy[1])]


FINISH = [call("finish", summary="done")]


def episode(tmp_path, turns, faults=None, task=None):
    env = MockTabletopEnv(faults)
    loop = AgentLoop(env, ScriptedLLM(turns), trace_dir=tmp_path)
    result = loop.run(task or clear_target_task(), seed=0)
    events = loop.tracer.events
    for e in events:  # no traced state may ever be physically impossible
        for key in ("state", "state_after", "world_state"):
            if e.get(key):
                assert_no_overlap(e[key])
    return result, env, events


def of(events, kind):
    return [e for e in events if e["event"] == kind]


def error_codes(events):
    return [e["error"]["code"] for e in of(events, "tool_result") if not e["ok"]]


def assert_no_overlap(s: dict):
    held = s["gripper"]["holding"]
    cubes = {k: o["pose"] for k, o in s["objects"].items() if o["kind"] == "cube" and k != held}
    names = sorted(cubes)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            pa, pb = cubes[a], cubes[b]
            assert not (abs(pa["x"] - pb["x"]) < 2 * CUBE_HALF and abs(pa["y"] - pb["y"]) < 2 * CUBE_HALF), (a, b)
    for rid, r in s["objects"].items():
        if r["kind"] == "target_region":
            inside = [c for c, p in cubes.items()
                      if abs(p["x"] - r["pose"]["x"]) < REGION_HALF and abs(p["y"] - r["pose"]["y"]) < REGION_HALF]
            assert len(inside) <= 1, (rid, inside)


# ---------------------------------------------------------------- task and predicates

def test_task_layout_and_goal():
    t = clear_target_task()
    assert t.layout["cube_blue"][1:] == t.layout["target"][1:]  # blue starts in the target
    assert all(isinstance(g, InRegion) and (g.half_x, g.half_y) == (REGION_HALF, REGION_HALF) for g in t.goal)
    s = MockTabletopEnv().reset(t)
    assert check_goal(t, s) == ["in_region(cube_blue, parking)", "in_region(cube_red, target)"]


def test_region_half_matches_existing_region_task():
    [g] = region_pick_place_task().goal
    assert (g.half_x, g.half_y) == (REGION_HALF, REGION_HALF)


# ---------------------------------------------------------------- occupancy boundaries (pure)

def _state(cubes: dict, regions: dict, holding=None, plates=None):
    objs = {k: ObjectObs(id=k, kind="cube", pose=Pose(x=x, y=y)) for k, (x, y) in cubes.items()}
    objs |= {k: ObjectObs(id=k, kind="target_region", pose=Pose(x=x, y=y)) for k, (x, y) in regions.items()}
    objs |= {k: ObjectObs(id=k, kind="plate", pose=Pose(x=x, y=y)) for k, (x, y) in (plates or {}).items()}
    return WorldState(t=0, step=0, objects=objs, gripper=GripperObs(pose=Pose(x=0, y=0, z=0.3), holding=holding))


@pytest.mark.parametrize("dx, conflict", [(0.039, True), (0.041, False)])
def test_cube_footprint_overlap_boundary(dx, conflict):
    s = _state({"a": (0.0, 0.0), "b": (0.3, 0.3)}, {}, holding="b")
    why = placement_conflict(s, "b", dx, 0.0)
    assert (why is not None) is conflict and (not conflict or why.startswith("cube overlap"))


@pytest.mark.parametrize("blue_dx, red_dx, conflict, reason", [
    (0.0, 0.042, True, "region occupied"),     # no physical overlap, but the region is full
    (0.0, 0.046, False, None),                 # red outside the region, no overlap
    (0.044, -0.03, True, "region occupied"),   # blue off-centre but still inside
    (0.046, -0.03, False, None),               # blue's centre just outside the region: region free
])
def test_region_capacity_boundary(blue_dx, red_dx, conflict, reason):
    s = _state({"cube_blue": (0.05 + blue_dx, 0.30), "cube_red": (0.10, -0.20)}, {"target": TARGET},
               holding="cube_red")
    why = placement_conflict(s, "cube_red", 0.05 + red_dx, 0.30)
    assert (why is not None) is conflict and (reason is None or why.startswith(reason))


def test_parking_holds_one_cube():
    s = _state({"cube_red": PARKING, "cube_blue": (0.3, 0.0)}, {"parking": PARKING}, holding="cube_blue")
    # 0.042 m from the occupant: no footprint overlap (R1), so only region capacity (R2) can refuse it
    assert placement_conflict(s, "cube_blue", PARKING[0] + 0.042, PARKING[1]).startswith("region occupied")


def test_held_cube_and_non_cube_objects_never_conflict():
    s = _state({"cube_red": RED0}, {"target": TARGET}, holding="cube_red", plates={"plate": (-0.2, 0.15)})
    assert placement_conflict(s, "cube_red", *RED0) is None       # put back where it came from
    assert placement_conflict(s, "cube_red", -0.2, 0.15) is None  # onto a plate
    assert placement_conflict(s, "cube_red", *TARGET) is None     # into an empty region


# ---------------------------------------------------------------- AgentLoop episodes

def test_correct_order_succeeds(tmp_path):
    turns = [OBSERVE, pick("cube_blue", TARGET), place(PARKING), pick("cube_red", RED0), place(TARGET), FINISH]
    result, env, events = episode(tmp_path, turns)
    assert result.success and result.reason == "goal_met" and error_codes(events) == []
    assert env.execute_calls == 4 and result.skills_executed == 4


def test_invalid_order_rejected_before_execution_and_recovers(tmp_path):
    turns = [OBSERVE, pick("cube_red", RED0), place(TARGET),          # red onto the occupied target
             place(RED0),                                              # put red back where it was
             pick("cube_blue", TARGET), place(PARKING), pick("cube_red", RED0), place(TARGET), FINISH]
    result, env, events = episode(tmp_path, turns)
    rejected = next(e for e in of(events, "validation_error") if e["failure"]["code"] == "DESTINATION_OCCUPIED")
    assert rejected["failure"]["detail"] == "region occupied: target already contains cube_blue"
    results = of(events, "tool_result")
    after_pick, after_reject = results[1]["world_state"], results[2]["world_state"]
    assert after_reject == after_pick  # no execute: nothing moved, not even time or step
    assert after_reject["gripper"]["holding"] == "cube_red"
    assert after_reject["objects"]["cube_red"]["pose"] == {"x": RED0[0], "y": RED0[1], "z": CARRY_Z}
    assert after_reject["objects"]["cube_blue"]["pose"] == {"x": TARGET[0], "y": TARGET[1], "z": 0.0}
    assert len(of(events, "env_outcome")) == env.execute_calls == 6  # the rejected place never executed
    put_back = results[3]
    assert put_back["ok"] and put_back["world_state"]["objects"]["cube_red"]["pose"] == {
        "x": RED0[0], "y": RED0[1], "z": 0.0}
    assert result.success and error_codes(events) == ["DESTINATION_OCCUPIED"] and result.retries == 0


def test_not_holding_keeps_precedence_over_occupancy(tmp_path):
    result, env, events = episode(tmp_path, [OBSERVE, place(TARGET)])
    assert error_codes(events) == ["NOT_HOLDING"] and env.execute_calls == 0


@pytest.mark.parametrize("prefix, unmet", [
    ([], ["in_region(cube_blue, parking)", "in_region(cube_red, target)"]),
    ([pick("cube_blue", TARGET), place(PARKING)], ["in_region(cube_red, target)"]),
])
def test_premature_finish_reports_remaining_subgoals(tmp_path, prefix, unmet):
    result, env, events = episode(tmp_path, [OBSERVE, *prefix, FINISH])
    [gc] = of(events, "goal_check")
    assert gc["passed"] is False and gc["unmet"] == unmet
    assert error_codes(events) == ["GOAL_NOT_MET"] and not result.success


def test_env_rejects_conflict_created_after_validation(tmp_path):
    # Blue is parked, then moved back into the target inside execute() of red's place (index 3):
    # validation saw an empty target, so only the env check can refuse it.
    turns = [OBSERVE, pick("cube_blue", TARGET), place(PARKING), pick("cube_red", RED0), place(TARGET),
             place(RED0), pick("cube_blue", TARGET), place(PARKING), pick("cube_red", RED0), place(TARGET), FINISH]
    result, env, events = episode(tmp_path, turns, faults=Faults(perturb_at={3: ("cube_blue", *TARGET)}))
    outcomes = of(events, "env_outcome")
    rejected = outcomes[3]
    assert rejected["outcome"]["ok"] is False
    assert rejected["outcome"]["failure"]["code"] == "DESTINATION_OCCUPIED"
    sa = rejected["state_after"]
    assert sa["gripper"]["holding"] == "cube_red"
    assert sa["objects"]["cube_red"]["pose"]["z"] == CARRY_Z          # still in the gripper
    assert sa["objects"]["cube_blue"]["pose"] == {"x": TARGET[0], "y": TARGET[1], "z": 0.0}
    before = outcomes[2]["state_after"]  # red just picked; only the injected blue move differs
    assert sa["gripper"] == before["gripper"]
    assert sa["objects"]["cube_red"]["pose"] == before["objects"]["cube_red"]["pose"]
    assert not of(events, "validation_error")  # the precondition passed: blue was parked when validated
    assert of(events, "retry") == []
    assert result.success and error_codes(events) == ["DESTINATION_OCCUPIED"]


# ---------------------------------------------------------------- preservation of existing behaviour

@pytest.mark.parametrize("task, oid", [(pick_place_task(), "plate"), (region_pick_place_task(), "target")])
def test_mock_refuses_to_pick_non_cube_objects(tmp_path, task, oid):
    _, x, y = task.layout[oid]
    result, env, events = episode(tmp_path, [OBSERVE, pick(oid, (x, y))], task=task)
    [out] = of(events, "env_outcome")  # passed validation and grounding; refused by the env
    assert not of(events, "validation_error")
    assert out["outcome"]["failure"] == {"code": "UNKNOWN_OBJECT", "detail": f"{oid!r} is not graspable"}
    assert out["state_after"]["gripper"]["holding"] is None
    assert out["state_after"]["objects"][oid]["pose"] == {"x": x, "y": y, "z": 0.0}


@pytest.mark.parametrize("task, support", [(pick_place_task(), "plate"), (region_pick_place_task(), "target")])
def test_existing_single_cube_tasks_still_succeed(tmp_path, task, support):
    env = MockTabletopEnv()
    loop = AgentLoop(env, PickPlacePolicyLLM("cube_red", support), trace_dir=tmp_path)
    result = loop.run(task, seed=0)
    assert result.success and result.reason == "goal_met" and result.failures == 0
    assert env.execute_calls == 2


def test_destination_occupied_is_never_retried_blindly():
    from robot_agent.world import TRANSIENT, FailureCode
    assert FailureCode.DESTINATION_OCCUPIED not in TRANSIENT
