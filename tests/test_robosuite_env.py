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


# --- Seeded resets (Milestone 1 Phase B1) ---

def _init(env, seed):
    env.reset(region_pick_place_task(), seed=seed)
    return env.backend_info["init"]


@sim
def test_same_seed_gives_identical_initial_conditions(env):
    a, b = _init(env, 0), _init(env, 0)
    assert a["rng"]["installed_at_reset"] == b["rng"]["installed_at_reset"] - 1
    ra, rb = a.pop("rng"), b.pop("rng")  # reset counter differs by design; everything else must match
    ra.pop("installed_at_reset"); rb.pop("installed_at_reset")
    assert ra == rb and ra["source"] == "seed"
    assert a == b and a["seeded"] is True and a["seed"] == 0


@sim
def test_different_seeds_vary_robot_joints_and_cube_size(env):
    a, b = _init(env, 0), _init(env, 1)
    assert a["robot_init_qpos"] != b["robot_init_qpos"]
    assert a["cube_half_size"] != b["cube_half_size"]


@sim
def test_unseeded_reset_after_seeded_reset_installs_fresh_unseeded_generators(env):
    # Checks what the worker installed, not that two random samples happen to differ.
    seeded = _init(env, 0)
    u1, u2 = _init(env, None), _init(env, None)
    for u in (u1, u2):
        assert u["seeded"] is False and u["seed"] is None and u["rng"]["source"] == "os_entropy"
        for g in ("env_seed_seq", "sampler_seed_seq"):
            assert u["rng"][g]["entropy"] != seeded["rng"][g]["entropy"] and u["rng"][g]["spawn_key"] == []
    assert u2["rng"]["installed_at_reset"] == u1["rng"]["installed_at_reset"] + 1
    assert u1["rng"]["env_seed_seq"] != u2["rng"]["env_seed_seq"]  # new generators, not the previous ones
    assert u1["rng"]["sampler_seed_seq"] != u2["rng"]["sampler_seed_seq"]
    assert seeded["rng"]["source"] == "seed" and seeded["rng"]["env_seed_seq"]["entropy"] == "0"
    assert seeded["rng"]["env_seed_seq"]["spawn_key"] != seeded["rng"]["sampler_seed_seq"]["spawn_key"]


@sim
def test_backend_info_records_versions_and_source_hashes(env):
    env.reset(region_pick_place_task(), seed=0)
    v = env.backend_info["versions"]
    assert {"python", "robosuite", "mujoco", "numpy", "files"} <= set(v)
    assert set(v["files"]) == {"worker", "env", "expert"}
    assert all(len(f["sha256"]) == 64 for f in v["files"].values())
    assert {"robot_init_qpos", "cube_half_size", "z_ref", "cube_pose_world", "eef_pose_world"} \
        <= set(env.backend_info["init"])


# --- Phase B2: deterministic failure paths through AgentLoop on robosuite ---
#
# Three different counters are kept apart in these tests:
#   execute_calls  - RobotEnv.execute() calls made by the loop (CountingEnv, client side)
#   WorldState.step - the backend's step counter (the worker increments it even when it rejects)
#   control_steps  - simulator control steps actually run (env_outcome.info, absent on rejection)

from robot_agent.llm import AssistantTurn, call, latest_state  # noqa: E402

TARGET = (0.05, 0.30)


class CountingEnv:
    """Records RobotEnv.execute() calls made by AgentLoop; everything else passes through.

    Each call is attributed, at call time, to the tool call the loop is handling (the call_id of the
    loop's latest trace event), so per-call counts do not rely on the loop's own env_outcome events.
    """

    def __init__(self, env):
        self.env, self.loop, self.calls = env, None, []  # calls: (skill name, call_id)

    @property
    def execute_calls(self):
        return [name for name, _ in self.calls]

    def executes_for(self, call_id):
        return sum(1 for _, cid in self.calls if cid == call_id)

    def reset(self, task, seed=None):
        self.calls = []
        return self.env.reset(task, seed)

    def observe(self):
        return self.env.observe()

    def execute(self, name, args):
        cid = next(e["call_id"] for e in reversed(self.loop.tracer.events) if e.get("call_id"))
        self.calls.append((name, cid))
        return self.env.execute(name, args)

    @property
    def backend_info(self):
        return self.env.backend_info


class HybridLLM:
    """Scripted failure-inducing prefix, then PickPlacePolicyLLM for recovery.

    The policy sees only the message list (latest delivered world_state), never the env. Each policy
    decision is recorded together with latest_state(messages) recomputed by the harness from the same
    messages, so tests can check what was delivered and that the arguments match it.
    """

    def __init__(self, prefix):
        self.prefix = list(prefix)
        self.policy = PickPlacePolicyLLM("cube_red", "target", tol=0.02)
        self.recovery = []  # (tool_call, delivered WorldState the policy read)

    def step(self, system, messages, tools):
        if self.prefix:
            return AssistantTurn(tool_calls=self.prefix.pop(0))
        turn = self.policy.step(system, messages, tools)
        self.recovery += [(tc, latest_state(messages)) for tc in turn.tool_calls]
        return turn


def of(events, kind, call_id=None):
    return [e for e in events if e["event"] == kind and (call_id is None or e.get("call_id") == call_id)]


def failure_layer(events, call_id):
    """Classify a failed call by its event chain (codes are shared between harness and worker)."""
    evs = [e for e in events if e.get("call_id") == call_id]
    kinds = [e["event"] for e in evs]
    [res] = [e for e in evs if e["event"] == "tool_result"]
    if res["ok"]:
        return None
    if any(e["event"] == "tool_call" and e.get("skipped") for e in evs):
        return "skipped"
    outs = [e["outcome"] for e in evs if e["event"] == "env_outcome"]
    if "validation_error" in kinds:
        assert not outs
        gc = [e for e in evs if e["event"] == "grounding_check"]
        return "grounding" if gc and not gc[-1]["ok"] else "validate_call"
    if "goal_check" in kinds:
        return "goal_check"
    if not outs[-1]["ok"]:
        return "worker_physical" if outs[-1]["info"].get("control_steps", 0) > 0 else "worker_rejected"
    return "postcondition"


def tool_result(events, call_id):
    [r] = of(events, "tool_result", call_id)
    return r


def previous_tool_result(events, seq):
    return [e for e in events if e["event"] == "tool_result" and e["seq"] < seq][-1]


def assert_failure(events, cenv, tc, layer, code, executes):
    assert failure_layer(events, tc.id) == layer
    assert tool_result(events, tc.id)["error"]["code"] == code
    assert len(of(events, "env_outcome", tc.id)) == executes  # loop's record
    assert cenv.executes_for(tc.id) == executes  # independent client-side record


def assert_rejected_without_motion(events, tc):
    """Worker rejection: execute was called and the step advanced, but nothing physical happened."""
    [out] = of(events, "env_outcome", tc.id)
    assert out["outcome"]["ok"] is False and out["outcome"]["info"] == {}  # no control_steps reported
    before = previous_tool_result(events, out["seq"])["world_state"]
    after = out["state_after"]
    assert after["t"] == before["t"]  # simulator time did not advance
    assert after["objects"] == before["objects"]  # poses, visibility, observed_at unchanged
    assert after["gripper"] == before["gripper"]  # eef pose and contact-based holding unchanged
    assert after["step"] == before["step"] + 1  # backend step counter still advanced
    assert of(events, "retry", tc.id) == []


def assert_recovery_grounded(events, llm):
    """Every recovery action: grounded by the tool_result delivered just before its turn, and its
    arguments are exactly that delivered state's poses (no hard-coded or env-read coordinates)."""
    assert llm.recovery
    for tc, seen in llm.recovery:
        [turn] = [e for e in of(events, "llm_turn") if any(c["id"] == tc.id for c in e["tool_calls"])]
        src = previous_tool_result(events, turn["seq"])
        assert turn["grounding_source_seq"] == src["seq"]
        assert seen.model_dump(mode="json") == src["world_state"]  # policy read exactly that state
        if tc.name in ("pick", "place"):
            [chk] = of(events, "grounding_check", tc.id)
            assert chk["ok"] is True
            assert (chk["source_seq"], chk["source_call_id"], chk["source_step"]) == \
                (src["seq"], src["call_id"], src["world_state"]["step"])
        if tc.name == "pick":
            p = src["world_state"]["objects"]["cube_red"]["pose"]
            assert (tc.args["x"], tc.args["y"]) == (p["x"], p["y"])
        if tc.name == "place":
            p = src["world_state"]["objects"]["target"]["pose"]
            assert (tc.args["x"], tc.args["y"]) == (p["x"], p["y"])


def control_steps(events):
    return sum(e["outcome"]["info"].get("control_steps", 0) for e in of(events, "env_outcome"))


def run_hybrid(env, tmp_path, prefix, seed=0):
    cenv, llm = CountingEnv(env), HybridLLM(prefix)
    loop = AgentLoop(cenv, llm, trace_dir=tmp_path)
    cenv.loop = loop
    result = loop.run(region_pick_place_task(), seed=seed)
    return result, loop.tracer.events, cenv, llm


def assert_recovered(result, events, cenv, *, failures, skills_executed, execute_calls):
    assert result.success and result.reason == "goal_met", result
    assert result.failures == failures and result.retries == 0
    assert result.skills_executed == skills_executed  # backend steps, incl. worker rejections
    assert cenv.execute_calls == execute_calls  # client-side RobotEnv.execute() calls
    assert len(cenv.calls) == len(of(events, "env_outcome"))  # every execute produced one env_outcome
    for name, cid in cenv.calls:  # attributed to the same calls as the loop's env_outcome events
        assert len(of(events, "env_outcome", cid)) == sum(1 for _, c in cenv.calls if c == cid)
    assert of(events, "goal_check")[-1]["passed"] is True


@sim
def test_b2_blind_correct_pick_rejected_without_execution_then_recovers(env, tmp_path):
    blind = call("pick", object_id="cube_red", x=0.0, y=-0.02)
    result, events, cenv, llm = run_hybrid(env, tmp_path, [[blind]])
    assert_failure(events, cenv, blind, "grounding", "UNGROUNDED_ACTION", executes=0)
    [chk] = of(events, "grounding_check", blind.id)
    assert chk["source_seq"] is None and chk["source_call_id"] is None and chk["live_step"] == 0
    rej = tool_result(events, blind.id)["world_state"]
    assert rej["step"] == 0 and rej["t"] == 0.0
    first_recovery, _ = llm.recovery[0]
    assert first_recovery.name == "pick"  # the rejection result itself grounded the next action
    assert of(events, "grounding_check", first_recovery.id)[0]["source_call_id"] == blind.id
    assert_recovery_grounded(events, llm)
    assert_recovered(result, events, cenv, failures=1, skills_executed=2, execute_calls=["pick", "place"])
    assert control_steps(events) > 0


@sim
def test_b2_wrong_pick_after_observe_rejected_by_validation(env, tmp_path):
    wrong = call("pick", object_id="cube_red", x=0.08, y=-0.02)
    result, events, cenv, llm = run_hybrid(env, tmp_path, [[call("observe")], [wrong]])
    assert_failure(events, cenv, wrong, "validate_call", "OBJECT_NOT_AT_POSE", executes=0)
    assert of(events, "grounding_check", wrong.id) == []  # validation precedes grounding
    obs_state = of(events, "tool_result")[0]["world_state"]
    rej = tool_result(events, wrong.id)["world_state"]
    assert rej == obs_state  # no execute, no step, no time
    assert of(events, "grounding_check", llm.recovery[0][0].id)[0]["source_call_id"] == wrong.id
    assert_recovery_grounded(events, llm)
    assert_recovered(result, events, cenv, failures=1, skills_executed=2, execute_calls=["pick", "place"])


@sim
def test_b2_premature_finish_rejected_by_goal_check_then_recovers(env, tmp_path):
    early = call("finish", summary="done")
    result, events, cenv, llm = run_hybrid(env, tmp_path, [[call("observe")], [early]])
    assert_failure(events, cenv, early, "goal_check", "GOAL_NOT_MET", executes=0)
    [gc] = of(events, "goal_check", early.id)
    assert gc["passed"] is False and gc["unmet"] == ["in_region(cube_red, target)"]
    assert of(events, "validation_error", early.id) == [] and of(events, "grounding_check", early.id) == []
    assert tool_result(events, early.id)["world_state"]["step"] == 0
    assert of(events, "grounding_check", llm.recovery[0][0].id)[0]["source_call_id"] == early.id
    assert_recovery_grounded(events, llm)
    assert_recovered(result, events, cenv, failures=1, skills_executed=2, execute_calls=["pick", "place"])


@sim
@pytest.mark.parametrize("bad_xy, seed, detail", [
    ((0.30, 0.30), 0, "outside robosuite reach"),  # inside WORKSPACE (±0.5), outside REACH
    ((0.0, 0.14), 7, "is on the barrier"),
])
def test_b2_place_outside_robot_reach_rejected_by_worker_without_motion(env, tmp_path, bad_xy, seed, detail):
    pick = call("pick", object_id="cube_red", x=0.0, y=-0.02)
    bad = call("place", x=bad_xy[0], y=bad_xy[1])
    result, events, cenv, llm = run_hybrid(env, tmp_path, [[call("observe")], [pick], [bad]], seed=seed)
    assert not of(events, "validation_error", bad.id)  # harness accepted it (inside WORKSPACE)
    [chk] = of(events, "grounding_check", bad.id)
    assert chk["ok"] and chk["source_call_id"] == pick.id and chk["source_step"] == chk["live_step"] == 1
    assert failure_layer(events, bad.id) == "worker_rejected"
    err = tool_result(events, bad.id)["error"]
    assert err["code"] == "OUT_OF_WORKSPACE" and detail in err["detail"]
    assert_rejected_without_motion(events, bad)
    after = tool_result(events, bad.id)["world_state"]
    assert after["gripper"]["holding"] == "cube_red" and after["step"] == 2  # grasp survives rejection
    [place_rec, _] = llm.recovery[0]
    assert place_rec.name == "place"
    rec_chk = of(events, "grounding_check", place_rec.id)[0]
    assert rec_chk["source_call_id"] == bad.id and rec_chk["source_step"] == rec_chk["live_step"] == 2
    assert_recovery_grounded(events, llm)
    assert_recovered(result, events, cenv, failures=1, skills_executed=3,
                     execute_calls=["pick", "place", "place"])
    phys = [e["outcome"]["info"].get("control_steps", 0) for e in of(events, "env_outcome")]
    assert phys[0] > 0 and phys[1] == 0 and phys[2] > 0  # pick moved, rejected place did not, recovery did


@sim
def test_b2_unknown_object_reaches_worker_and_is_rejected_without_motion(env, tmp_path):
    bad = call("pick", object_id="target", x=TARGET[0], y=TARGET[1])
    result, events, cenv, llm = run_hybrid(env, tmp_path, [[call("observe")], [bad]])
    # The harness accepts it (target is a known object, at the commanded point, unmoved) ...
    assert not of(events, "validation_error", bad.id)
    assert of(events, "grounding_check", bad.id)[0]["ok"] is True
    # ... so it must reach the worker, which rejects it before any motion.
    assert cenv.executes_for(bad.id) == 1 and len(of(events, "env_outcome", bad.id)) == 1
    assert failure_layer(events, bad.id) == "worker_rejected"
    err = tool_result(events, bad.id)["error"]
    assert err["code"] == "UNKNOWN_OBJECT" and "not graspable" in err["detail"]  # worker's wording, not the harness's
    assert_rejected_without_motion(events, bad)
    assert of(events, "grounding_check", llm.recovery[0][0].id)[0]["source_call_id"] == bad.id
    assert_recovery_grounded(events, llm)
    assert_recovered(result, events, cenv, failures=1, skills_executed=3,
                     execute_calls=["pick", "pick", "place"])


@sim
def test_b2_place_chained_after_physical_pick_in_one_turn_is_ungrounded(env, tmp_path):
    pick = call("pick", object_id="cube_red", x=0.0, y=-0.02)
    chained = call("place", x=TARGET[0], y=TARGET[1])
    result, events, cenv, llm = run_hybrid(env, tmp_path, [[call("observe")], [pick, chained]])
    assert of(events, "env_outcome", pick.id)[0]["outcome"]["ok"] is True  # pick executed physically
    assert_failure(events, cenv, chained, "grounding", "UNGROUNDED_ACTION", executes=0)
    [chk] = of(events, "grounding_check", chained.id)
    assert "holding changed" in chk["reason"] and "(None -> 'cube_red')" in chk["reason"]
    assert chk["source_step"] == 0 and chk["live_step"] == 1  # same-turn snapshot predates the pick
    after = tool_result(events, chained.id)["world_state"]
    assert after["gripper"]["holding"] == "cube_red" and after["step"] == 1
    rec, _ = llm.recovery[0]
    assert rec.name == "place" and of(events, "grounding_check", rec.id)[0]["source_call_id"] == chained.id
    assert_recovery_grounded(events, llm)
    assert_recovered(result, events, cenv, failures=1, skills_executed=2, execute_calls=["pick", "place"])


@sim
def test_b2_reset_after_episode_ending_while_holding_restores_seeded_state(env, tmp_path):
    from robot_agent.llm import ScriptedLLM
    env.reset(region_pick_place_task(), seed=0)
    fresh = dict(env.backend_info["init"])
    loop = AgentLoop(env, ScriptedLLM([[call("observe")], [call("pick", object_id="cube_red", x=0.0, y=-0.02)]]),
                     trace_dir=tmp_path)
    result = loop.run(region_pick_place_task(), seed=0)  # ends llm_stopped while holding the cube
    assert result.reason == "llm_stopped" and env.observe().gripper.holding == "cube_red"
    s = env.reset(region_pick_place_task(), seed=0)
    again = dict(env.backend_info["init"])
    fr, ar = dict(fresh.pop("rng")), dict(again.pop("rng"))
    fr.pop("installed_at_reset"); ar.pop("installed_at_reset")
    assert again == fresh and ar == fr
    assert s.gripper.holding is None and s.step == 0 and s.t == 0.0
