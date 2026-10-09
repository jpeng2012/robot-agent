"""Acceptance scenarios for the closed loop."""

import json

import pytest

from robot_agent.env import Faults, MockTabletopEnv
from robot_agent.goals import pick_place_task
from robot_agent.llm import PickPlacePolicyLLM, ScriptedLLM, call
from robot_agent.loop import AgentLoop
from robot_agent.world import FailureCode, Pose


def run(tmp_path, llm=None, faults=None, env=None):
    env = env or MockTabletopEnv(faults)
    loop = AgentLoop(env, llm or PickPlacePolicyLLM("cube_red", "plate"), trace_dir=tmp_path)
    result = loop.run(pick_place_task(), seed=0)
    return result, env, loop.tracer.events


def of(events, kind):
    return [e for e in events if e["event"] == kind]


def error_codes(events):
    return [e["error"]["code"] for e in of(events, "tool_result") if not e["ok"]]


def test_pick_and_place_succeeds(tmp_path):
    result, env, events = run(tmp_path)
    assert result.success and result.reason == "goal_met"
    assert result.skills_executed == 2 and result.retries == 0
    assert of(events, "goal_check")[-1]["passed"] is True
    assert error_codes(events) == []


def test_failed_pick_recovers_by_bounded_retry(tmp_path):
    result, env, events = run(tmp_path, faults=Faults(fail_once={"pick": FailureCode.GRASP_SLIP}))
    assert result.success and result.retries == 1
    [retry] = of(events, "retry")
    assert retry["reason"] == "GRASP_SLIP"
    # Recovery happened inside the loop: the LLM issued only one pick and saw no error.
    assert len([e for e in of(events, "tool_call") if e["name"] == "pick"]) == 1
    assert error_codes(events) == []
    assert env.execute_calls == 3  # slip, retried pick, place


class SlipAndNudgeEnv(MockTabletopEnv):
    """Slip that also knocks the object away, so retrying the same grasp would be unsafe."""

    def _pick(self, oid, x, y):
        out = super()._pick(oid, x, y)
        if not out.ok and out.failure.code == FailureCode.GRASP_SLIP:
            self.poses[oid] = Pose(x=x + 0.1, y=y)
        return out


def test_transient_failure_not_retried_when_grounding_is_stale(tmp_path):
    env = SlipAndNudgeEnv(Faults(fail_once={"pick": FailureCode.GRASP_SLIP}))
    result, env, events = run(tmp_path, env=env)
    assert of(events, "retry") == []
    assert error_codes(events) == ["GRASP_SLIP"]  # handed back to the LLM instead
    assert result.success  # policy re-grounds on the refreshed state


def test_object_moved_after_planning_requires_regrounding(tmp_path):
    result, env, events = run(tmp_path, faults=Faults(perturb_at={0: ("cube_red", 0.25, 0.05)}))
    assert result.success
    assert error_codes(events) == ["OBJECT_NOT_AT_POSE"]
    assert of(events, "retry") == []  # not transient: no blind retry
    picks = [e["args"] for e in of(events, "tool_call") if e["name"] == "pick"]
    assert picks[0] == {"object_id": "cube_red", "x": 0.10, "y": -0.20}  # stale grounding
    assert picks[1] == {"object_id": "cube_red", "x": 0.25, "y": 0.05}  # re-grounded
    # The failed attempt's refreshed state already exposed the new pose.
    failed = of(events, "env_outcome")[0]
    assert failed["state_after"]["objects"]["cube_red"]["pose"]["x"] == 0.25


@pytest.mark.parametrize(
    "bad_call, code",
    [
        (call("pick", object_id="cube_red", x=0.1), FailureCode.INVALID_ARGS),  # missing field
        (call("pick", object_id="cube_red", x="left", y=-0.2), FailureCode.INVALID_ARGS),  # wrong type
        (call("pick", object_id="cube_red", x="0.1", y=-0.2), FailureCode.INVALID_ARGS),  # no coercion
        (call("pick", object_id="cube_red", x=0.1, y=-0.2, force=99), FailureCode.INVALID_ARGS),  # extra
        (call("pick", object_id="cube_blue", x=0.1, y=-0.2), FailureCode.UNKNOWN_OBJECT),
        (call("pick", object_id="cube_red", x=3.0, y=-0.2), FailureCode.OUT_OF_WORKSPACE),
        (call("pick", object_id="cube_red", x=0.3, y=0.3), FailureCode.OBJECT_NOT_AT_POSE),
        (call("place", x=-0.2, y=0.15), FailureCode.NOT_HOLDING),
        (call("teleport", object_id="cube_red"), FailureCode.UNKNOWN_TOOL),
    ],
)
def test_invalid_tool_calls_rejected_before_env(tmp_path, bad_call, code):
    env = MockTabletopEnv()
    initial = env.reset(pick_place_task())
    result, env, events = run(tmp_path, llm=ScriptedLLM([[bad_call]]), env=env)
    assert not result.success and result.reason == "llm_stopped"
    assert env.execute_calls == 0
    assert env.observe().model_dump(exclude={"t"}) == initial.model_dump(exclude={"t"})
    [err] = of(events, "validation_error")
    assert err["failure"]["code"] == code


def test_malformed_args_payload_rejected(tmp_path):
    bad = call("pick")
    bad.args = "not a json object"
    result, env, events = run(tmp_path, llm=ScriptedLLM([[bad]]))
    assert error_codes(events) == ["INVALID_ARGS"] and env.execute_calls == 0


def test_loop_continues_after_invalid_call(tmp_path):
    llm = ScriptedLLM([
        [call("pick", object_id="cube_red", x="oops", y=0)],
        [call("pick", object_id="cube_red", x=0.10, y=-0.20)],
        [call("place", x=-0.20, y=0.15)],
        [call("finish", summary="done")],
    ])
    result, _, events = run(tmp_path, llm=llm)
    assert result.success and error_codes(events) == ["INVALID_ARGS"]


def test_premature_finish_rejected_by_goal_check(tmp_path):
    llm = ScriptedLLM([[call("observe")], [call("finish", summary="looks done to me")]])
    result, _, events = run(tmp_path, llm=llm)
    assert not result.success
    assert error_codes(events) == ["GOAL_NOT_MET"]
    [check] = of(events, "goal_check")
    assert check["passed"] is False and check["unmet"] == ["on(cube_red, plate)"]


def test_silent_failure_cannot_produce_false_success(tmp_path):
    naive = ScriptedLLM([
        [call("pick", object_id="cube_red", x=0.10, y=-0.20)],
        [call("place", x=-0.20, y=0.15)],
        [call("finish", summary="placed it")],
    ])
    result, _, events = run(tmp_path, llm=naive, faults=Faults(silent_drop_on_place=(-0.05, 0.30)))
    assert not result.success
    # Env said ok, postcondition said otherwise; then the goal gate refused finish.
    assert of(events, "env_outcome")[1]["outcome"]["ok"] is True
    assert error_codes(events) == ["POSTCONDITION_FAILED", "GOAL_NOT_MET"]


def test_policy_recovers_from_silent_failure(tmp_path):
    result, _, events = run(tmp_path, faults=Faults(silent_drop_on_place=(-0.05, 0.30)))
    assert result.success and error_codes(events) == ["POSTCONDITION_FAILED"]
    assert result.skills_executed == 4  # pick, place (dropped), pick, place


def test_trace_is_valid_jsonl(tmp_path):
    result, _, events = run(tmp_path, faults=Faults(fail_once={"pick": FailureCode.GRASP_SLIP}))
    lines = result.trace_path.read_text().splitlines()
    records = [json.loads(line) for line in lines]
    assert records == events
    assert [r["seq"] for r in records] == list(range(len(records)))
    assert records[0]["event"] == "episode_start" and records[-1]["event"] == "episode_end"
    assert {"llm_turn", "tool_call", "env_outcome", "verify", "retry", "tool_result",
            "goal_check"} <= {r["event"] for r in records}
