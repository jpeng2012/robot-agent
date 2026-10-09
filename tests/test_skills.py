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
