"""Robosuite backend. Simulator tests need ROBOT_AGENT_ROBOSUITE_PYTHON; InRegion tests always run."""

import os

import pytest

from robot_agent.goals import InRegion, check_goal, region_pick_place_task
from robot_agent.llm import PickPlacePolicyLLM
from robot_agent.loop import AgentLoop
from robot_agent.skills import PickArgs, PlaceArgs
from robot_agent.world import GripperObs, ObjectObs, Pose, WorldState


def state(cube: Pose, holding: str | None = None) -> WorldState:
    return WorldState(t=0, step=0, gripper=GripperObs(pose=Pose(x=0, y=0, z=0.3), holding=holding), objects={
        "cube_red": ObjectObs(id="cube_red", kind="cube", pose=cube),
        "target": ObjectObs(id="target", kind="target_region", pose=Pose(x=0.05, y=0.30)),
    })


@pytest.mark.parametrize("cube, holding, ok", [
    (Pose(x=0.05, y=0.30), None, True),
    (Pose(x=0.09, y=0.34), None, True),  # box corner: inside the square, outside a 0.05 circle
    (Pose(x=0.10, y=0.30), None, False),  # outside in x
    (Pose(x=0.05, y=0.30, z=0.10), None, False),  # above the region, not resting on the table
    (Pose(x=0.05, y=0.30), "cube_red", False),  # still held
])
def test_in_region(cube, holding, ok):
    assert (check_goal(region_pick_place_task(), state(cube, holding)) == []) is ok


def test_in_region_is_a_table_region_not_a_support():
    g = InRegion(obj="cube_red", region="target", half_x=0.045, half_y=0.045)
    assert str(g) == "in_region(cube_red, target)"


sim = pytest.mark.skipif(not os.environ.get("ROBOT_AGENT_ROBOSUITE_PYTHON"),
                         reason="ROBOT_AGENT_ROBOSUITE_PYTHON not set")


@pytest.fixture(scope="module")
def env():
    from robot_agent.robosuite_env import RobosuiteEnv

    with RobosuiteEnv() as e:
        yield e


@sim
def test_reset_matches_layout_and_rests_on_table(env):
    s = env.reset(region_pick_place_task(), seed=0)
    cube, tgt = s.objects["cube_red"].pose, s.objects["target"].pose
    assert abs(cube.x - 0.0) < 0.01 and abs(cube.y + 0.02) < 0.01 and abs(cube.z) < 0.005
    assert (tgt.x, tgt.y, tgt.z) == pytest.approx((0.05, 0.30, 0.0))
    assert s.gripper.holding is None and s.step == 0
    assert env.backend_info["controller"] == "CompositeController" and env.backend_info["control_freq"] == 10


@sim
def test_direct_pick_then_place_meets_goal(env):
    task = region_pick_place_task()
    s = env.reset(task, seed=0)
    cube = s.objects["cube_red"].pose
    out = env.execute("pick", PickArgs(object_id="cube_red", x=cube.x, y=cube.y))
    assert out.ok, out
    assert out.info["termination"] == "fsm_lifted" and out.info["grasped"] and out.info["cube_pose"]["z"] >= 0.08
    s = env.observe()
    assert s.gripper.holding == "cube_red" and s.objects["cube_red"].pose.z >= 0.08

    out = env.execute("place", PlaceArgs(x=0.05, y=0.30))
    assert out.ok, out
    assert out.info["termination"] == "fsm_released" and not out.info["grasped"] and out.info["in_target_region"]
    s = env.observe()
    assert s.gripper.holding is None and abs(s.objects["cube_red"].pose.z) < 0.01
    assert check_goal(task, s) == []


@sim
def test_pick_at_wrong_point_is_not_a_slip(env):
    env.reset(region_pick_place_task(), seed=0)
    out = env.execute("pick", PickArgs(object_id="cube_red", x=0.0, y=-0.10))
    assert not out.ok and out.failure.code == "OBJECT_NOT_AT_POSE", out
    assert not out.info["grasped"]
    assert env.observe().gripper.holding is None


@sim
def test_rule_based_agent_loop_succeeds(env, tmp_path):
    task = region_pick_place_task()
    result = AgentLoop(env, PickPlacePolicyLLM("cube_red", "target", tol=0.02), trace_dir=tmp_path).run(task, seed=0)
    assert result.success and result.reason == "goal_met", result
    assert result.failures == 0 and result.skills_executed == 2
