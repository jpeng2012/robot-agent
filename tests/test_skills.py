import json

from robot_agent.env import MockTabletopEnv
from robot_agent.goals import pick_place_task
from robot_agent.skills import PlaceArgs, tool_schemas, validate_call, verify
from robot_agent.world import SkillOutcome


def test_tool_schemas_are_json_schema():
    schemas = {t["name"]: t for t in tool_schemas()}
    assert set(schemas) == {"observe", "pick", "place", "finish"}
    pick = schemas["pick"]["input_schema"]
    assert pick["type"] == "object" and set(pick["required"]) == {"object_id", "x", "y"}
    assert pick["additionalProperties"] is False
    json.dumps(schemas)


def test_validate_accepts_grounded_pick():
    s = MockTabletopEnv().reset(pick_place_task())
    args = validate_call("pick", {"object_id": "cube_red", "x": 0.1, "y": -0.2}, s)
    assert args.object_id == "cube_red"


def test_verify_catches_env_reporting_false_success():
    env = MockTabletopEnv()
    before = env.reset(pick_place_task())
    before.gripper.holding = "cube_red"  # pretend we were holding it
    after = env.observe()  # cube never moved
    failure = verify("place", PlaceArgs(x=-0.2, y=0.15), before, SkillOutcome(ok=True), after)
    assert failure is not None and failure.code == "POSTCONDITION_FAILED"


# --- F010 grounding_failure (pure) ---

from robot_agent.skills import PickArgs, grounding_failure, requires_grounding  # noqa: E402
from robot_agent.world import Pose  # noqa: E402


def _state():
    return MockTabletopEnv().reset(pick_place_task())


def test_requires_grounding_only_for_pose_dependent_tools():
    assert {n for n in ("observe", "pick", "place", "finish", "nope") if requires_grounding(n)} == {"pick", "place"}


def test_grounding_failure_reasons():
    pick = PickArgs(object_id="cube_red", x=0.1, y=-0.2)
    live = _state()
    assert grounding_failure("pick", pick, None, live).code == "UNGROUNDED_ACTION"
    assert grounding_failure("pick", pick, _state(), live) is None
    missing = _state(); del missing.objects["cube_red"]
    assert "not in the last world state" in grounding_failure("pick", pick, missing, live).detail
    moved = _state(); moved.objects["cube_red"].pose = Pose(x=0.3, y=-0.2)
    assert "moved" in grounding_failure("pick", pick, moved, live).detail
    nudged = _state(); nudged.objects["cube_red"].pose = Pose(x=0.11, y=-0.2)  # within GRASP_TOL
    assert grounding_failure("pick", pick, nudged, live) is None
    place = PlaceArgs(x=-0.2, y=0.15)
    holding = _state(); holding.gripper.holding = "cube_red"
    assert "holding changed" in grounding_failure("place", place, _state(), holding).detail
    assert grounding_failure("place", place, holding, holding) is None
    assert grounding_failure("observe", None, None, live) is None
