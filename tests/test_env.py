from robot_agent.env import Faults, MockTabletopEnv
from robot_agent.goals import check_goal, pick_place_task
from robot_agent.skills import PickArgs, PlaceArgs


def test_reset_populates_observation_fields():
    s = MockTabletopEnv().reset(pick_place_task())
    cube = s.objects["cube_red"]
    assert (cube.id, cube.kind, cube.pose.x, cube.pose.y) == ("cube_red", "cube", 0.10, -0.20)
    assert cube.visible and cube.confidence == 1.0 and cube.observed_at == 0.0
    assert s.gripper.holding is None and s.step == 0


def test_pick_then_place_achieves_goal_and_advances_time():
    env = MockTabletopEnv()
    task = pick_place_task()
    env.reset(task)
    assert env.execute("pick", PickArgs(object_id="cube_red", x=0.10, y=-0.20)).ok
    assert env.observe().gripper.holding == "cube_red"
    assert env.execute("place", PlaceArgs(x=-0.20, y=0.15)).ok
    s = env.observe()
    assert check_goal(task, s) == []
    assert s.step == 2 and s.t > 0 and s.objects["cube_red"].observed_at == s.t


def test_perturbation_applies_before_scheduled_call():
    env = MockTabletopEnv(Faults(perturb_at={0: ("cube_red", 0.3, 0.3)}))
    env.reset(pick_place_task())
    out = env.execute("pick", PickArgs(object_id="cube_red", x=0.10, y=-0.20))
    assert not out.ok and out.failure.code == "OBJECT_NOT_AT_POSE"
    assert env.observe().objects["cube_red"].pose.x == 0.3
