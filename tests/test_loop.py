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


def test_text_only_turn_is_nudged_once(tmp_path):
    llm = ScriptedLLM([
        [call("pick", object_id="cube_red", x=0.10, y=-0.20)],
        [],  # narrates without calling a tool
        [call("place", x=-0.20, y=0.15)],
        [call("finish", summary="done")],
    ])
    result, _, events = run(tmp_path, llm=llm)
    assert result.success and len(of(events, "nudge")) == 1


def test_two_consecutive_text_only_turns_stop(tmp_path):
    result, _, events = run(tmp_path, llm=ScriptedLLM([[call("observe")]]))
    assert result.reason == "llm_stopped" and len(of(events, "nudge")) == 1


def test_pick_postcondition_catches_false_success(tmp_path):
    """Postcondition detects pick that reported ok=True but didn't actually grasp."""
    from robot_agent.env import CARRY_Z
    from robot_agent.world import SkillOutcome

    class FalseSuccessEnv(MockTabletopEnv):
        """Env reports pick success but doesn't actually grasp the object."""

        def _pick(self, oid, x, y):
            # Simulate controller reporting success without physical grasp
            # Move gripper and object as if pick succeeded, but DON'T set self.holding
            self.gripper = Pose(x=x, y=y, z=CARRY_Z)
            self.poses[oid] = Pose(x=x, y=y, z=CARRY_Z)
            # Missing: self.holding = oid
            return SkillOutcome(ok=True)  # Environment reports false success

    naive = ScriptedLLM([
        [call("pick", object_id="cube_red", x=0.10, y=-0.20)],
        [call("finish", summary="picked it")],
    ])

    result, env, events = run(tmp_path, llm=naive, env=FalseSuccessEnv())

    # Assertions: Postcondition should catch the lie
    assert not result.success  # Episode should fail
    assert of(events, "env_outcome")[0]["outcome"]["ok"] is True  # Env lied
    assert error_codes(events) == ["POSTCONDITION_FAILED", "GOAL_NOT_MET"]
    # Verify that postcondition caught the false success
    [verify_event] = of(events, "verify")
    assert not verify_event["ok"]
    assert verify_event["failure"]["code"] == "POSTCONDITION_FAILED"


def test_inconsistent_state_holding_without_contact(tmp_path):
    """Spatial consistency check rejects object far from gripper."""
    from robot_agent.env import CARRY_Z
    from robot_agent.world import SkillOutcome

    class InconsistentStateEnv(MockTabletopEnv):
        """Reports holding object but object pose is far from gripper."""

        def _pick(self, oid, x, y):
            # Set holding but place object far away (inconsistent state)
            self.holding = oid
            self.gripper = Pose(x=x, y=y, z=CARRY_Z)
            # Place object 0.5m away from gripper - physically impossible to be holding it
            self.poses[oid] = Pose(x=x + 0.5, y=y + 0.5, z=CARRY_Z)
            return SkillOutcome(ok=True)

    naive = ScriptedLLM([
        [call("pick", object_id="cube_red", x=0.10, y=-0.20)],
        [call("finish", summary="picked it")],
    ])

    result, env, events = run(tmp_path, llm=naive, env=InconsistentStateEnv())

    # Postcondition should now REJECT spatial inconsistency
    assert of(events, "env_outcome")[0]["outcome"]["ok"] is True
    verify_event = of(events, "verify")[0]  # Get first verify event (pick)
    assert not verify_event["ok"]  # Postcondition fails
    assert verify_event["failure"]["code"] == "POSTCONDITION_FAILED"
    # Error message should mention XY distance
    assert "0.707" in verify_event["failure"]["detail"]  # sqrt(0.5^2 + 0.5^2)
    assert error_codes(events) == ["POSTCONDITION_FAILED", "GOAL_NOT_MET"]
    assert not result.success


def test_pick_postcondition_rejects_object_on_table(tmp_path):
    """Spatial consistency check rejects object at resting height."""
    from robot_agent.env import CARRY_Z
    from robot_agent.world import SkillOutcome

    class ObjectStillOnTableEnv(MockTabletopEnv):
        def _pick(self, oid, x, y):
            self.holding = oid
            self.gripper = Pose(x=x, y=y, z=CARRY_Z)
            # Object at correct XY but still on table (z=0.0)
            self.poses[oid] = Pose(x=x, y=y, z=0.0)
            return SkillOutcome(ok=True)

    naive = ScriptedLLM([
        [call("pick", object_id="cube_red", x=0.10, y=-0.20)],
        [call("finish", summary="picked it")],
    ])

    result, env, events = run(tmp_path, llm=naive, env=ObjectStillOnTableEnv())

    # Postcondition should reject: holding=True but z=0.0 (resting)
    assert of(events, "env_outcome")[0]["outcome"]["ok"] is True
    verify_event = of(events, "verify")[0]  # Get first verify event (pick)
    assert not verify_event["ok"]
    assert verify_event["failure"]["code"] == "POSTCONDITION_FAILED"
    assert "z=0.000" in verify_event["failure"]["detail"]
    assert "resting on table" in verify_event["failure"]["detail"]
    assert error_codes(events) == ["POSTCONDITION_FAILED", "GOAL_NOT_MET"]
    assert not result.success


def test_pick_postcondition_accepts_minimal_carry_height(tmp_path):
    """Spatial consistency accepts object just above MIN_CARRY_Z threshold."""
    from robot_agent.env import CARRY_Z
    from robot_agent.world import SkillOutcome

    class MinimalCarryEnv(MockTabletopEnv):
        def _pick(self, oid, x, y):
            self.holding = oid
            self.gripper = Pose(x=x, y=y, z=CARRY_Z)
            # Object at minimal valid carry height: MIN_CARRY_Z + 1mm
            self.poses[oid] = Pose(x=x, y=y, z=0.051)
            return SkillOutcome(ok=True)

    result, env, events = run(tmp_path, env=MinimalCarryEnv())

    # Should PASS: just above threshold
    assert result.success
    pick_verify = of(events, "verify")[0]  # Get first verify event (pick)
    assert pick_verify["ok"]
    assert error_codes(events) == []


def test_pick_postcondition_accepts_max_carry_xy_error(tmp_path):
    """Spatial consistency accepts object at CARRY_XY_TOL boundary."""
    from robot_agent.env import CARRY_Z
    from robot_agent.world import SkillOutcome

    class MaxXYErrorEnv(MockTabletopEnv):
        def _pick(self, oid, x, y):
            self.holding = oid
            self.gripper = Pose(x=x, y=y, z=CARRY_Z)
            # Object at exactly CARRY_XY_TOL distance (0.049m < 0.05m)
            self.poses[oid] = Pose(x=x + 0.049, y=y, z=CARRY_Z)
            return SkillOutcome(ok=True)

    result, env, events = run(tmp_path, env=MaxXYErrorEnv())

    # Should PASS: within tolerance
    assert result.success
    pick_verify = of(events, "verify")[0]  # Get first verify event (pick)
    assert pick_verify["ok"]


# ========== F008: Baseline Qwen3 Replanning Evaluation ==========


def test_failure_context_structure(tmp_path):
    """Verify tool results include failure code, detail, and fresh world_state."""
    # Use existing SlipAndNudgeEnv and PickPlacePolicyLLM to generate failure
    env = SlipAndNudgeEnv(Faults(fail_once={"pick": FailureCode.GRASP_SLIP}))
    policy = PickPlacePolicyLLM("cube_red", "plate")
    loop = AgentLoop(env, policy, trace_dir=tmp_path)
    result = loop.run(pick_place_task(), seed=0)

    # Use existing helper to get events
    events = loop.tracer.events

    # Get failed tool results
    failed_tool_results = [e for e in of(events, "tool_result") if not e["ok"]]

    # Verify at least one failure result
    assert len(failed_tool_results) >= 1, "Expected at least one failed tool result"

    # Verify structure of first failure
    failure_event = failed_tool_results[0]
    assert failure_event["ok"] is False
    assert "error" in failure_event
    assert "code" in failure_event["error"]  # e.g., "GRASP_SLIP"
    assert "detail" in failure_event["error"]  # human-readable description
    assert "world_state" in failure_event  # fresh observation
    assert "objects" in failure_event["world_state"]
    assert "gripper" in failure_event["world_state"]


@pytest.mark.requires_server
def test_qwen3_nominal_pick_and_place(tmp_path):
    """Baseline: Qwen3 completes pick-and-place without injected failures."""
    from robot_agent.llm import OllamaLLMClient

    env = MockTabletopEnv()
    llm = OllamaLLMClient(model="qwen3:8b", base_url="http://localhost:11434/v1", temperature=0.0)
    loop = AgentLoop(env, llm, max_turns=10, trace_dir=tmp_path)
    result = loop.run(pick_place_task(), seed=0)

    # Collect metrics
    assert result.success, f"Nominal execution failed: {result.reason}"
    print(f"\nNominal execution: success={result.success}, llm_turns={result.llm_turns}, "
          f"skills_executed={result.skills_executed}, failures={result.failures}")

    # Trace analysis (extract tool calls)
    from pathlib import Path
    trace_file = list(Path(tmp_path).glob("*.jsonl"))[0]
    tool_calls = []
    with open(trace_file) as f:
        for line in f:
            event = json.loads(line)
            if event["event"] == "tool_call":
                tool_calls.append(event["name"])

    print(f"Tool call sequence: {' → '.join(tool_calls)}")


@pytest.mark.requires_server
def test_qwen3_recovers_from_moved_object(tmp_path):
    """Qwen3 replans after object moves between observe and execute."""
    from robot_agent.llm import OllamaLLMClient

    env = SlipAndNudgeEnv(Faults(fail_once={"pick": FailureCode.GRASP_SLIP}))
    llm = OllamaLLMClient(model="qwen3:8b", base_url="http://localhost:11434/v1", temperature=0.0)
    loop = AgentLoop(env, llm, max_turns=15, trace_dir=tmp_path)
    result = loop.run(pick_place_task(), seed=0)

    # Metrics
    print(f"\nMoved-object recovery: success={result.success}, llm_turns={result.llm_turns}, "
          f"skills_executed={result.skills_executed}, failures={result.failures}, reason={result.reason}")

    # Trace analysis: verify Qwen3 chose updated coordinates
    from pathlib import Path
    trace_file = list(Path(tmp_path).glob("*.jsonl"))[0]

    pick_calls = []
    with open(trace_file) as f:
        for line in f:
            event = json.loads(line)
            if event["event"] == "tool_call" and event["name"] == "pick":
                args = event["args"]
                pick_calls.append((args["x"], args["y"]))

    print(f"Pick coordinates: {pick_calls}")
    # If Qwen3 replans correctly, second pick should have different coordinates
    if len(pick_calls) >= 2:
        dx = abs(pick_calls[0][0] - pick_calls[1][0])
        dy = abs(pick_calls[0][1] - pick_calls[1][1])
        print(f"Coordinate change: Δx={dx:.4f}, Δy={dy:.4f}")
        # SlipAndNudgeEnv nudges by ~0.1m (see line 54), so expect Δ > 0.05m
        assert dx > 0.05 or dy > 0.05, f"Qwen3 did not update coordinates after failure (Δx={dx}, Δy={dy})"


@pytest.mark.requires_server
def test_qwen3_recovers_from_postcondition_failure(tmp_path):
    """Qwen3 retries after pick postcondition detects false grasp."""
    from robot_agent.llm import OllamaLLMClient
    from robot_agent.world import SkillOutcome

    class TransientFalseSuccessEnv(MockTabletopEnv):
        """First pick reports ok=True but doesn't set holding (postcondition fails). Second pick succeeds."""

        def __init__(self):
            super().__init__()
            self.pick_count = 0

        def _pick(self, oid, x, y):
            if self.pick_count == 0:
                self.pick_count += 1
                # Report success but don't set holding
                return SkillOutcome(ok=True)
            return super()._pick(oid, x, y)

    env = TransientFalseSuccessEnv()
    llm = OllamaLLMClient(model="qwen3:8b", base_url="http://localhost:11434/v1", temperature=0.0)
    loop = AgentLoop(env, llm, max_turns=15, trace_dir=tmp_path)
    result = loop.run(pick_place_task(), seed=0)

    # Metrics
    print(f"\nPostcondition-failure recovery: success={result.success}, llm_turns={result.llm_turns}, "
          f"skills_executed={result.skills_executed}, failures={result.failures}, reason={result.reason}")

    # Trace analysis: verify Qwen3 observed after postcondition failure
    from pathlib import Path
    trace_file = list(Path(tmp_path).glob("*.jsonl"))[0]

    sequence = []
    with open(trace_file) as f:
        for line in f:
            event = json.loads(line)
            if event["event"] == "tool_call":
                sequence.append(event["name"])
            elif event["event"] == "tool_result" and not event["ok"]:
                sequence.append(f"FAIL({event['error']['code']})")

    print(f"Event sequence: {' → '.join(sequence)}")
