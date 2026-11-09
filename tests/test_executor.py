"""F012c: closed-loop symbolic plan execution through AgentLoop (mock backend), plus B006 fault guards."""

import inspect
from pathlib import Path

import pytest

from robot_agent import executor as executor_module
from robot_agent.env import CARRY_Z, Faults, InvalidFault, MockTabletopEnv
from robot_agent.executor import PlanExecutor, deviation
from robot_agent.goals import InRegion, TaskSpec
from robot_agent.llm import AssistantTurn, call
from robot_agent.loop import AgentLoop
from robot_agent.planner import InvalidStep, Plan, PlanStep, plan, simulate
from robot_agent.skills import PickArgs, PlaceArgs
from robot_agent.world import REGION_HALF, FailureCode as F, WorldState

TARGET, PARKING, RED0 = (0.05, 0.30), (-0.25, 0.30), (0.10, -0.20)


def G(o, r):
    return InRegion(obj=o, region=r, half_x=REGION_HALF, half_y=REGION_HALF)


STD = [G("cube_red", "target"), G("cube_blue", "parking")]
CLEAR = {"target": ("target_region", *TARGET), "cube_blue": ("cube", *TARGET),
         "parking": ("target_region", *PARKING), "cube_red": ("cube", *RED0)}
SWAP = {**CLEAR, "cube_red": ("cube", *PARKING)}
FREE = {**CLEAR, "cube_blue": ("cube", 0.20, 0.0)}
R3 = {"r0": ("target_region", -0.3, 0.3), "r1": ("target_region", 0.0, 0.3), "r2": ("target_region", 0.3, 0.3)}
CHAIN3 = {**R3, "c0": ("cube", 0.3, -0.2), "c1": ("cube", 0.0, 0.3), "c2": ("cube", -0.3, 0.3)}
CHAIN3_GOALS = [G("c0", "r0"), G("c1", "r2"), G("c2", "r1")]
CYCLE3 = {**R3, "c0": ("cube", -0.3, 0.3), "c1": ("cube", 0.0, 0.3), "c2": ("cube", 0.3, 0.3)}
CYCLE3_GOALS = [G("c0", "r1"), G("c1", "r2"), G("c2", "r0")]
BOUNDS = (-0.45, 0.45, -0.30, 0.45)


def task(layout, goals):
    return TaskSpec(name="executor_test", instruction="", layout=layout, goal=list(goals))


class Handoff:
    """Scripted prefix turns (e.g. observe + pick), then the wrapped client."""

    def __init__(self, prefix, then):
        self.prefix, self.then = list(prefix), then

    def step(self, system, messages, tools):
        if self.prefix:
            return AssistantTurn(tool_calls=self.prefix.pop(0))
        return self.then.step(system, messages, tools)


def run(tmp_path, layout, goals, faults=None, exec_goals=None, prefix=(), max_turns=40, max_failures=8, **kw):
    ex = PlanExecutor(list(goals if exec_goals is None else exec_goals), **kw)
    env = MockTabletopEnv(faults)
    loop = AgentLoop(env, Handoff(prefix, ex) if prefix else ex, trace_dir=tmp_path, max_turns=max_turns,
                     max_failures=max_failures)
    r = loop.run(task(layout, goals), seed=0)
    ev = loop.tracer.events
    check_invariants(ev)
    return r, env, ev, ex


def of(ev, name):
    return [e for e in ev if e["event"] == name]


def exec_turns(ev):
    return [e for e in of(ev, "llm_turn") if e["meta"].get("executor") == "plan"]


def outcome(r, ev):
    """Terminal outcome classifier: the five outcomes must be distinguishable from the trace alone."""
    if r.success:
        assert r.reason == "goal_met" and of(ev, "goal_check")[-1]["passed"]
        return "goal_met"
    if r.reason == "llm_stopped":
        meta = exec_turns(ev)[-1]["meta"]
        assert meta["mode"] == "stop"
        return "planning_failure" if "planning_failure" in meta else "executor_failure:" + meta["executor_failure"]["code"]
    return r.reason  # failure_budget | max_steps


# ------------------------------------------------------------------ independent physical oracle

def physical_violations(ws: dict) -> list[str]:
    held = ws["gripper"]["holding"]
    objs = ws["objects"]
    cubes = {k: o["pose"] for k, o in objs.items() if o["kind"] == "cube"}
    regions = {k: o["pose"] for k, o in objs.items() if o["kind"] == "target_region"}
    out = []
    resting = {k: p for k, p in cubes.items() if k != held}
    for k, p in resting.items():
        if abs(p["z"]) > 1e-9:
            out.append(f"{k} not on the table")
    names = sorted(resting)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            pa, pb = resting[a], resting[b]
            if abs(pa["x"] - pb["x"]) < 0.04 and abs(pa["y"] - pb["y"]) < 0.04:
                out.append(f"{a} overlaps {b}")
    for rid, r in regions.items():
        inside = [k for k, p in resting.items() if abs(p["x"] - r["x"]) < 0.045 and abs(p["y"] - r["y"]) < 0.045]
        if len(inside) > 1:
            out.append(f"{rid} holds {inside}")
    if held is not None and (held not in cubes or cubes[held]["z"] < 0.05):
        out.append(f"held {held} not carried")
    return out


def check_invariants(ev):
    states = [e["world_state"] for e in of(ev, "tool_result")] + [e["state_after"] for e in of(ev, "env_outcome")]
    states += [e["state"] for e in ev if e["event"] in ("episode_start", "episode_end")]
    for ws in states:
        assert physical_violations(ws) == []
    errors = [e["error"]["code"] for e in of(ev, "tool_result") if e["error"]]
    assert "UNGROUNDED_ACTION" not in errors
    delivered, last = None, {}  # last (plan_id, step_index, ok) for advancement checks
    pending_ok, plans, prev_plan_id = None, {}, None
    for e in ev:
        if e["event"] == "llm_turn" and e["meta"].get("executor") == "plan":
            m = e["meta"]
            if "plan" in m and m["mode"] != "stop":
                assert m["plan_id"] not in plans
                plans[m["plan_id"]] = Plan.model_validate(m["plan"])
            if m.get("step_index") == 0:  # a step 0 is only ever the first step of a newly announced plan
                assert "plan" in m and m["plan_id"] != prev_plan_id
            if m.get("step_index") is not None:  # the emitted call is the announced plan's step
                st, [tc] = plans[m["plan_id"]].steps[m["step_index"]], e["tool_calls"]
                assert tc["name"] == st.op == m["mode"] and m["purpose"] == st.purpose
                if st.op == "pick":
                    assert tc["args"]["object_id"] == st.object_id
                elif st.op == "place" and st.purpose == "goal":
                    r = delivered["objects"][st.region]["pose"]
                    assert (tc["args"]["x"], tc["args"]["y"]) == (r["x"], r["y"])
                elif st.op == "place":
                    assert (tc["args"]["x"], tc["args"]["y"]) == (st.x, st.y)
                assert delivered["gripper"]["holding"] == (st.object_id if st.op == "place" else None) \
                    or st.op == "finish"
            if m.get("plan_id") is not None:
                prev_plan_id = m["plan_id"]
            assert len(e["tool_calls"]) <= 1
            if m["mode"] == "stop":
                assert e["tool_calls"] == []
            else:
                assert len(e["tool_calls"]) == 1
            for tc in e["tool_calls"]:
                if tc["name"] == "pick":  # coordinates from the newest delivered state
                    p = delivered["objects"][tc["args"]["object_id"]]["pose"]
                    assert (tc["args"]["x"], tc["args"]["y"]) == (p["x"], p["y"])
            k = m.get("step_index")
            if k:  # advanced within a plan: the previous step of that plan was reconciled as ok
                assert last.get("key") == (m["plan_id"], k - 1) and pending_ok, m
            last["key"] = (m.get("plan_id"), k) if k is not None else None
        if e["event"] == "tool_result":
            delivered, pending_ok = e["world_state"], e["ok"]


# ------------------------------------------------------------------ nominal execution

@pytest.mark.parametrize("name, layout, goals, n", [
    ("clear", CLEAR, STD, 4), ("clear_blue_first", CLEAR, STD[::-1], 4), ("swap", SWAP, STD, 6),
    ("chain3", CHAIN3, CHAIN3_GOALS, 6), ("cycle3", CYCLE3, CYCLE3_GOALS, 8),
])
def test_nominal_execution(tmp_path, name, layout, goals, n):
    r, env, ev, ex = run(tmp_path, layout, goals)
    assert outcome(r, ev) == "goal_met"
    assert [e for e in of(ev, "tool_result") if not e["ok"]] == [] and r.failures == 0
    assert env.execute_calls == n and ex.replans == 0 and ex.plan_id == 0
    turns = exec_turns(ev)
    assert [t["meta"]["mode"] for t in turns][0] == "observe" and len(turns) == n + 2
    announced = [t for t in turns if "plan" in t["meta"]]
    assert len(announced) == 1 and announced[0] is turns[1]  # recorded once, before its first step runs
    assert Plan.model_validate(announced[0]["meta"]["plan"]).physical_actions == n
    assert [t["meta"]["step_index"] for t in turns[1:]] == list(range(n + 1))


def test_start_while_holding_plans_from_first_delivered_state(tmp_path):
    prefix = [[call("observe")], [call("pick", object_id="cube_red", x=RED0[0], y=RED0[1])]]
    r, env, ev, ex = run(tmp_path, CLEAR, STD, prefix=prefix)
    assert outcome(r, ev) == "goal_met" and ex.replans == 0 and ex.plan_id == 0 and r.failures == 0
    first = exec_turns(ev)[0]
    assert first["meta"]["mode"] == "place" and first["meta"]["plan"]["steps"][0]["purpose"] == "stage"
    assert first["grounding_step"] == 1  # planned from the delivered post-pick state, no extra observe
    assert env.execute_calls == 1 + 5  # prefix pick + stage red, blue to parking, red to target


def test_final_success_only_through_agentloop_goal_check(tmp_path):
    # The executor only knows red -> target; the task also requires blue -> parking. Its finish is
    # rejected by the loop, it replans once (GOAL_NOT_MET), and then detects that it would repeat itself.
    r, _, ev, ex = run(tmp_path, CLEAR, STD, exec_goals=[G("cube_red", "target")])
    assert not r.success and outcome(r, ev) == "executor_failure:LIVELOCK_DETECTED"
    checks = of(ev, "goal_check")
    assert [c["passed"] for c in checks] == [False, False]
    assert exec_turns(ev)[-1]["meta"]["replan_reason"]["trigger"] == "GOAL_NOT_MET"
    assert [t["meta"]["replan_reason"]["trigger"] for t in exec_turns(ev) if "replan_reason" in t["meta"]] == \
        ["GOAL_NOT_MET", "GOAL_NOT_MET", "GOAL_NOT_MET"]  # replanned turn, stop turn, repeated stop


# ------------------------------------------------------------------ disturbance and recovery

def test_object_moved_into_goal_region_replans(tmp_path):
    # FREE plan: blue -> parking, red -> target. While red is carried, blue is moved into the target
    # (index 3 = red's place); the env refuses the place and the executor clears the target again.
    r, env, ev, ex = run(tmp_path, FREE, STD, faults=Faults(perturb_at={3: ("cube_blue", *TARGET)}))
    assert outcome(r, ev) == "goal_met" and ex.replans == 1 and ex.plan_id == 1
    [rr] = [t["meta"]["replan_reason"] for t in exec_turns(ev) if "replan_reason" in t["meta"]]
    assert (rr["trigger"], rr["code"], rr["plan_id"], rr["step_index"]) == ("EXECUTION_FAILURE", "DESTINATION_OCCUPIED", 0, 3)
    assert env.execute_calls == 4 + 5  # stage red, blue -> parking, red -> target


def test_unexpected_state_change_after_ok_result_replans(tmp_path):
    # Red is moved to a free spot during blue's (successful) place: no failure, but a state deviation.
    r, env, ev, ex = run(tmp_path, CLEAR, STD, faults=Faults(perturb_at={1: ("cube_red", 0.20, -0.10)}))
    assert outcome(r, ev) == "goal_met" and ex.replans == 1 and r.failures == 0
    [t] = [t for t in exec_turns(ev) if "replan_reason" in t["meta"]]
    assert t["meta"]["replan_reason"]["trigger"] == "STATE_DEVIATION" and t["meta"]["replan_reason"]["step_index"] == 1
    assert t["tool_calls"][0]["args"] == {"object_id": "cube_red", "x": 0.20, "y": -0.10}
    assert env.execute_calls == 4


def test_silent_drop_to_free_location_replans(tmp_path):
    r, env, ev, ex = run(tmp_path, CLEAR, STD, faults=Faults(silent_drop_on_place=(0.20, 0.0)))
    assert outcome(r, ev) == "goal_met" and ex.replans == 1
    [rr] = [t["meta"]["replan_reason"] for t in exec_turns(ev) if "replan_reason" in t["meta"]]
    assert (rr["trigger"], rr["code"]) == ("EXECUTION_FAILURE", "POSTCONDITION_FAILED")
    assert env.execute_calls == 6  # blue re-picked from the drop point


def test_non_transient_place_failure_replans(tmp_path):
    r, env, ev, ex = run(tmp_path, CLEAR, STD, faults=Faults(fail_once={"place": F.POSTCONDITION_FAILED}))
    assert outcome(r, ev) == "goal_met" and ex.replans == 1 and of(ev, "retry") == []
    [rr] = [t["meta"]["replan_reason"] for t in exec_turns(ev) if "replan_reason" in t["meta"]]
    assert (rr["trigger"], rr["code"], rr["step_index"]) == ("EXECUTION_FAILURE", "POSTCONDITION_FAILED", 1)
    assert env.execute_calls == 5


@pytest.mark.parametrize("skill", ["pick", "place"])
def test_transient_slip_is_retried_by_agentloop_without_replan(tmp_path, skill):
    r, env, ev, ex = run(tmp_path, CLEAR, STD, faults=Faults(fail_once={skill: F.GRASP_SLIP}))
    assert outcome(r, ev) == "goal_met" and ex.replans == 0 and ex.plan_id == 0
    assert len(of(ev, "retry")) == 1 and r.failures == 0 and env.execute_calls == 5


def test_small_drift_is_tolerated_and_pick_uses_delivered_pose(tmp_path):
    r, _, ev, ex = run(tmp_path, CLEAR, STD, faults=Faults(perturb_at={0: ("cube_red", 0.11, -0.20)}))
    assert outcome(r, ev) == "goal_met" and ex.replans == 0
    picks = [t["tool_calls"][0]["args"] for t in exec_turns(ev) if t["meta"]["mode"] == "pick"]
    assert picks[1] == {"object_id": "cube_red", "x": 0.11, "y": -0.20}  # not the planned (0.10, -0.20)


def test_accumulated_drift_fails_the_pick_precheck_and_replans(tmp_path):
    # Two 0.015 m moves stay within PLACE_TOL of each prediction but put red 0.03 m from its planned pose.
    faults = Faults(perturb_at={0: ("cube_red", 0.115, -0.20), 1: ("cube_red", 0.13, -0.20)})
    r, _, ev, ex = run(tmp_path, CLEAR, STD, faults=faults)
    assert outcome(r, ev) == "goal_met" and ex.replans == 1
    [t] = [t for t in exec_turns(ev) if "replan_reason" in t["meta"]]
    assert t["meta"]["replan_reason"]["trigger"] == "PRECHECK_FAILED" and t["meta"]["replan_reason"]["step_index"] == 2
    assert t["tool_calls"][0]["args"] == {"object_id": "cube_red", "x": 0.13, "y": -0.20}


# ------------------------------------------------------------------ failure handling

def test_replan_budget_exhaustion(tmp_path):
    faults = Faults(perturb_at={1: ("cube_red", 0.20, -0.10), 2: ("cube_red", 0.25, -0.10)})
    r, env, ev, ex = run(tmp_path, CLEAR, STD, faults=faults, max_replans=1)
    assert outcome(r, ev) == "executor_failure:REPLAN_BUDGET_EXHAUSTED" and ex.replans == 1
    stop = exec_turns(ev)[-1]["meta"]
    assert stop["replan_reason"]["trigger"] == "EXECUTION_FAILURE" and stop["replan_reason"]["code"] == "OBJECT_NOT_AT_POSE"
    assert of(ev, "goal_check") == []


def test_zero_replan_budget_stops_on_first_deviation(tmp_path):
    r, env, ev, ex = run(tmp_path, CLEAR, STD, faults=Faults(perturb_at={1: ("cube_red", 0.20, -0.10)}), max_replans=0)
    assert outcome(r, ev) == "executor_failure:REPLAN_BUDGET_EXHAUSTED" and env.execute_calls == 2


def test_initially_infeasible_goals_are_a_planning_failure(tmp_path):
    goals = [G("cube_red", "target"), G("cube_blue", "target")]
    r, env, ev, ex = run(tmp_path, CLEAR, goals)
    assert outcome(r, ev) == "planning_failure" and env.execute_calls == 0
    pf = exec_turns(ev)[-1]["meta"]["planning_failure"]
    assert (pf["code"], pf["subcode"]) == ("INFEASIBLE_GOALS", "REGION_CONFLICT")
    assert [t["meta"]["mode"] for t in exec_turns(ev)] == ["observe", "stop", "stop"]


def test_goals_infeasible_after_disturbance_are_a_planning_failure(tmp_path):
    # Red is pushed outside the executor's bounds (still inside WORKSPACE): no plan can pick it.
    r, env, ev, ex = run(tmp_path, CLEAR, STD, faults=Faults(perturb_at={1: ("cube_red", 0.10, -0.40)}),
                         bounds=BOUNDS)
    assert outcome(r, ev) == "planning_failure" and ex.replans == 1
    stop = exec_turns(ev)[-1]["meta"]
    assert (stop["planning_failure"]["code"], stop["planning_failure"]["subcode"]) == ("OUT_OF_BOUNDS", "PICK")
    assert stop["replan_reason"]["trigger"] == "STATE_DEVIATION"


def test_unchanged_state_failure_gets_one_reattempt_even_on_a_plans_first_step(tmp_path):
    # Start holding red: the first plan's first step (stage red) fails non-transiently with the scene unchanged,
    # i.e. from the plan's own source state. As at any later step, one re-attempt is made (and succeeds).
    prefix = [[call("observe")], [call("pick", object_id="cube_red", x=RED0[0], y=RED0[1])]]
    r, env, ev, ex = run(tmp_path, CLEAR, STD, prefix=prefix, faults=Faults(fail_once={"place": F.POSTCONDITION_FAILED}))
    assert outcome(r, ev) == "goal_met" and ex.replans == 1 and env.execute_calls == 1 + 1 + 5


def test_livelock_after_second_identical_failure():
    ex, msgs, s0, t1 = started()
    err = {"code": "OBJECT_NOT_AT_POSE", "detail": "injected"}
    deliver(msgs, t1, s0.model_copy(update={"step": 1, "t": 0.5}), ok=False, error=err)  # nothing moved
    t2 = ex.step("", msgs, [])
    assert (t2.tool_calls[0].name, t2.meta["plan_id"], ex.replans) == ("pick", 1, 1)
    deliver(msgs, t2, s0.model_copy(update={"step": 2, "t": 1.0}), ok=False, error=err)
    t3 = ex.step("", msgs, [])
    assert t3.tool_calls == [] and t3.meta["executor_failure"]["code"] == "LIVELOCK_DETECTED"
    assert ex.replans == 1  # the refused replan is not counted


def test_livelock_after_repeated_missing_feedback_from_unchanged_state():
    ex, msgs, s0, t1 = started()
    turn = t1
    for i in range(2):
        msgs.append({"role": "assistant", "text": "", "tool_calls": turn.tool_calls})  # result never arrives
        obs = ex.step("", msgs, [])
        assert obs.tool_calls[0].name == "observe" and obs.meta["replan_reason"]["trigger"] == "MISSING_FEEDBACK"
        deliver(msgs, obs, s0.model_copy(update={"t": 0.1 * (i + 1)}))  # the pick never ran
        turn = ex.step("", msgs, [])
    assert turn.tool_calls == [] and turn.meta["executor_failure"]["code"] == "LIVELOCK_DETECTED"


def test_executor_instance_is_reset_between_episodes(tmp_path):
    ex = PlanExecutor(STD)
    for i, faults in enumerate([Faults(fail_once={"place": F.POSTCONDITION_FAILED}), None]):
        loop = AgentLoop(MockTabletopEnv(faults), ex, trace_dir=tmp_path / str(i), max_turns=40)
        r = loop.run(task(CLEAR, STD), seed=0)
        check_invariants(loop.tracer.events)
        assert r.success and ex.replans == (1 if i == 0 else 0) and ex.plan_id == ex.replans


def test_pick_outside_executor_bounds_fails_precheck():
    ex = PlanExecutor(STD, bounds=(-0.45, 0.45, -0.30, 0.45))
    s = S({**CLEAR, "cube_red": ("cube", 0.10, -0.31)}, STD)
    planned = PlanStep(op="pick", purpose="goal", object_id="cube_red", x=0.10, y=-0.295)  # within GRASP_TOL
    assert "outside the executor bounds" in ex._ground(s, planned)


def test_planner_maps_simulate_rejection_to_internal_failure(monkeypatch):
    from robot_agent import planner as planner_module

    def reject(state, step):
        raise InvalidStep("injected")
    monkeypatch.setattr(planner_module, "simulate", reject)
    f = plan(STD, S(CLEAR, STD))
    assert (f.code, f.subcode, f.partial_steps) == ("PLANNER_INTERNAL", "SIMULATE_REJECTED", [])


def test_agentloop_budgets_still_apply(tmp_path):
    r, _, ev, _ = run(tmp_path / "a", CYCLE3, CYCLE3_GOALS, max_turns=4)
    assert outcome(r, ev) == "max_steps"
    r, _, ev, ex = run(tmp_path / "b", FREE, STD, faults=Faults(perturb_at={3: ("cube_blue", *TARGET)}), max_failures=1)
    assert outcome(r, ev) == "failure_budget" and ex.replans == 0


# ------------------------------------------------------------------ reconciliation (synthetic messages, no env)

def S(layout, goals=()):
    return MockTabletopEnv().reset(task(layout, goals))


def result(tc, s: WorldState | None, ok=True, error=None):
    content = {"ok": ok}
    if error:
        content["error"] = error
    if s is not None:
        content["world_state"] = s.model_dump(mode="json")
    return {"role": "tool", "tool_call_id": tc.id, "name": tc.name, "content": content}


def deliver(messages, turn, s, **kw):
    messages.append({"role": "assistant", "text": turn.text, "tool_calls": turn.tool_calls})
    for tc in turn.tool_calls:
        messages.append(result(tc, s, **kw))


def started(layout=CLEAR, goals=STD):
    ex, msgs = PlanExecutor(goals), [{"role": "user", "content": "Task"}]
    t0 = ex.step("", msgs, [])
    assert t0.tool_calls[0].name == "observe"
    s0 = S(layout, goals)
    deliver(msgs, t0, s0)
    t1 = ex.step("", msgs, [])
    assert t1.meta["step_index"] == 0 and t1.tool_calls[0].name == "pick"
    return ex, msgs, s0, t1


def after_pick(s0, oid="cube_blue"):
    p = s0.objects[oid].pose
    return simulate(s0, PlanStep(op="pick", purpose="goal", object_id=oid, x=p.x, y=p.y)).model_copy(update={"t": 0.5})


def test_missing_feedback_is_not_success():
    ex, msgs, s0, t1 = started()
    msgs.append({"role": "assistant", "text": "", "tool_calls": t1.tool_calls})  # no tool result
    t2 = ex.step("", msgs, [])
    assert t2.tool_calls[0].name == "observe" and ex.index == 0
    assert t2.meta["replan_reason"]["trigger"] == "MISSING_FEEDBACK" and ex.replans == 1
    deliver(msgs, t2, after_pick(s0))  # the pick did happen
    t3 = ex.step("", msgs, [])
    assert (t3.tool_calls[0].name, t3.meta["plan_id"], t3.meta["step_index"]) == ("place", 1, 0)


def test_result_without_world_state_is_missing_feedback():
    ex, msgs, s0, t1 = started()
    deliver(msgs, t1, None)
    t2 = ex.step("", msgs, [])
    assert t2.tool_calls[0].name == "observe" and t2.meta["replan_reason"]["trigger"] == "MISSING_FEEDBACK"


@pytest.mark.parametrize("make, ok", [
    (lambda s0: s0, True),  # ok, but nothing executed (step unchanged)
    (lambda s0: s0.model_copy(update={"step": -1}), False),  # failed result carrying a state older than the grounding
    (lambda s0: s0.model_copy(update={"t": -1.0}), False),
])
def test_stale_feedback_is_not_success(make, ok):
    ex, msgs, s0, t1 = started()
    deliver(msgs, t1, make(s0), ok=ok, error=None if ok else {"code": "OBJECT_NOT_AT_POSE", "detail": ""})
    t2 = ex.step("", msgs, [])
    assert t2.tool_calls[0].name == "observe" and ex.index == 0
    assert t2.meta["replan_reason"]["trigger"] == "STALE_FEEDBACK"
    deliver(msgs, t2, s0)  # the action did not run: re-plan once from the observed state
    t3 = ex.step("", msgs, [])
    assert (t3.tool_calls[0].name, t3.meta["plan_id"], t3.meta["step_index"]) == ("pick", 1, 0)
    assert t3.meta["replan_reason"]["trigger"] == "STALE_FEEDBACK" and ex.replans == 1


def test_missing_feedback_counts_against_the_replan_budget():
    ex, msgs, s0, t1 = started()
    for _ in range(3):
        msgs.append({"role": "assistant", "text": "", "tool_calls": []})
        t = ex.step("", msgs, [])
        assert t.tool_calls[0].name == "observe"
    t = ex.step("", msgs, [])
    assert t.tool_calls == [] and t.meta["executor_failure"]["code"] == "REPLAN_BUDGET_EXHAUSTED"
    assert ex.step("", msgs, []).meta == t.meta  # the stop is repeated


def test_advance_only_after_matching_feedback():
    ex, msgs, s0, t1 = started()
    deliver(msgs, t1, after_pick(s0).model_copy(update={"step": 1}))
    t2 = ex.step("", msgs, [])
    assert (t2.tool_calls[0].name, t2.meta["step_index"], ex.replans) == ("place", 1, 0)


def test_finish_requires_goals_in_delivered_state():
    ex = PlanExecutor(STD)
    s = S(CLEAR, STD)
    assert "goals not met" in ex._ground(s, PlanStep(op="finish", purpose="finish"))
    tc, _ = ex._ground(S({**CLEAR, "cube_blue": ("cube", *PARKING), "cube_red": ("cube", *TARGET)}),
                       PlanStep(op="finish", purpose="finish"))
    assert tc.name == "finish"


def test_staging_place_is_revalidated_with_placement_conflict():
    ex = PlanExecutor(STD)
    s = S(CLEAR, STD)
    held = simulate(s, PlanStep(op="pick", purpose="goal", object_id="cube_red", x=RED0[0], y=RED0[1]))
    onto_blue = PlanStep(op="place", purpose="stage", object_id="cube_red", x=TARGET[0], y=TARGET[1])
    assert "region occupied" in ex._ground(held, onto_blue)


def test_goal_place_uses_delivered_region_centre():
    ex = PlanExecutor(STD)
    s = S({**FREE, "target": ("target_region", TARGET[0] + 0.01, TARGET[1])}, STD)
    held = simulate(s, PlanStep(op="pick", purpose="goal", object_id="cube_red", x=RED0[0], y=RED0[1]))
    planned = PlanStep(op="place", purpose="goal", object_id="cube_red", region="target", x=TARGET[0], y=TARGET[1])
    tc, expected = ex._ground(held, planned)
    assert tc.args == {"x": TARGET[0] + 0.01, "y": TARGET[1]} and expected.objects["cube_red"].pose.x == TARGET[0] + 0.01


def test_deviation_tolerance():
    s = S(CLEAR, STD)
    assert deviation(s, s) is None
    near = s.model_copy(deep=True)
    near.objects["cube_red"].pose.x += 0.019
    far = s.model_copy(deep=True)
    far.objects["cube_red"].pose.x += 0.021
    held = s.model_copy(deep=True)
    held.gripper.holding = "cube_red"
    assert deviation(s, near) is None and "cube_red" in deviation(s, far) and "holding" in deviation(s, held)


def test_executor_has_no_environment_access():
    assert list(inspect.signature(PlanExecutor).parameters) == ["goals", "bounds", "max_replans", "tolerances", "clearance"]
    src = Path(inspect.getsourcefile(executor_module)).read_text()
    for forbidden in ("MockTabletopEnv", "RobotEnv", "RobosuiteEnv", ".execute(", ".observe(", "task.goal"):
        assert forbidden not in src
    ex, msgs, s0, t1 = started()  # driven by synthetic messages only
    assert t1.tool_calls[0].args == {"object_id": "cube_blue", "x": TARGET[0], "y": TARGET[1]}


# ------------------------------------------------------------------ planner.simulate

def test_simulate_matches_mock_kinematics_and_is_pure():
    env = MockTabletopEnv()
    s0 = env.reset(task(CLEAR, STD))
    before = s0.model_dump_json()
    p = plan(STD, s0)
    s = s0
    for st in p.steps[:-1]:
        s = simulate(s, st)
        if st.op == "pick":
            env.execute("pick", PickArgs(object_id=st.object_id, x=st.x, y=st.y))
        else:
            env.execute("place", PlaceArgs(x=st.x, y=st.y))
        live = env.observe()
        assert deviation(s, live) is None and s.gripper == live.gripper and s.step == live.step
    assert s0.model_dump_json() == before


@pytest.mark.parametrize("st, why", [
    (PlanStep(op="finish", purpose="finish"), "no physical transition"),
    (PlanStep(op="pick", purpose="goal", object_id="cube_red"), "needs x and y"),
    (PlanStep(op="pick", purpose="goal", object_id="target", x=TARGET[0], y=TARGET[1]), "not a known cube"),
    (PlanStep(op="pick", purpose="goal", object_id="cube_red", x=0.2, y=0.2), "GRASP_TOL"),
    (PlanStep(op="place", purpose="goal", object_id="cube_red", x=0.0, y=0.0), "gripper is empty"),
])
def test_simulate_rejects_invalid_steps(st, why):
    with pytest.raises(InvalidStep, match=why):
        simulate(S(CLEAR, STD), st)


def test_simulate_rejects_conflicting_place_and_wrong_object():
    s = S(CLEAR, STD)
    held = simulate(s, PlanStep(op="pick", purpose="goal", object_id="cube_red", x=RED0[0], y=RED0[1]))
    with pytest.raises(InvalidStep, match="region occupied"):
        simulate(held, PlanStep(op="place", purpose="goal", x=TARGET[0], y=TARGET[1]))
    with pytest.raises(InvalidStep, match="holding cube_red"):
        simulate(held, PlanStep(op="place", purpose="stage", object_id="cube_blue", x=0.2, y=0.0))
    with pytest.raises(InvalidStep, match="already holding"):
        simulate(held, PlanStep(op="pick", purpose="goal", object_id="cube_blue", x=TARGET[0], y=TARGET[1]))


# ------------------------------------------------------------------ B006: fault fixtures stay physical

def env_for(layout=CLEAR, **faults):
    env = MockTabletopEnv(Faults(**faults))
    env.reset(task(layout, STD))
    return env


@pytest.mark.parametrize("dest, why", [
    ((0.105, -0.195), "cube overlap"),          # onto red
    ((PARKING[0] + 0.03, PARKING[1]), None),     # parking is empty: valid
])
def test_perturb_cube_onto_cube_is_refused(dest, why):
    env = env_for(perturb_at={0: ("cube_blue", *dest)})
    if why is None:
        assert not env.execute("pick", PickArgs(object_id="cube_blue", x=TARGET[0], y=TARGET[1])).ok
        assert (env.observe().objects["cube_blue"].pose.x, env.observe().objects["cube_blue"].pose.y) == dest
    else:
        with pytest.raises(InvalidFault, match=why):
            env.execute("pick", PickArgs(object_id="cube_red", x=RED0[0], y=RED0[1]))


def test_perturb_second_cube_into_occupied_region_is_refused():
    env = env_for(perturb_at={0: ("cube_red", TARGET[0] + 0.041, TARGET[1])})  # no overlap, but target is full
    with pytest.raises(InvalidFault, match="region occupied"):
        env.execute("pick", PickArgs(object_id="cube_red", x=RED0[0], y=RED0[1]))


def test_perturb_held_object_is_refused_and_non_held_is_allowed():
    env = env_for(perturb_at={1: ("cube_red", 0.2, 0.0)})
    assert env.execute("pick", PickArgs(object_id="cube_red", x=RED0[0], y=RED0[1])).ok
    with pytest.raises(InvalidFault, match="is held"):
        env.execute("place", PlaceArgs(x=0.0, y=0.0))
    env = env_for(perturb_at={1: ("cube_blue", *RED0)})  # directly below the carried red cube: allowed
    assert env.execute("pick", PickArgs(object_id="cube_red", x=RED0[0], y=RED0[1])).ok
    assert env.execute("place", PlaceArgs(x=0.2, y=0.0)).ok
    assert (env.observe().objects["cube_blue"].pose.x, env.observe().objects["cube_blue"].pose.y) == RED0


@pytest.mark.parametrize("oid, why", [("target", "only cubes"), ("cube_green", "unknown object")])
def test_perturb_non_cube_or_unknown_is_refused(oid, why):
    env = env_for(perturb_at={0: (oid, 0.2, 0.0)})
    with pytest.raises(InvalidFault, match=why):
        env.execute("pick", PickArgs(object_id="cube_red", x=RED0[0], y=RED0[1]))


@pytest.mark.parametrize("oid, pose, drop, why", [
    ("cube_red", RED0, TARGET, "region occupied"),                           # into blue's region
    ("cube_blue", TARGET, (RED0[0] + 0.03, RED0[1]), "cube overlap"),        # onto red, outside every region
])
def test_silent_drop_onto_occupied_spot_is_refused(oid, pose, drop, why):
    env = env_for(silent_drop_on_place=drop)
    env.execute("pick", PickArgs(object_id=oid, x=pose[0], y=pose[1]))
    with pytest.raises(InvalidFault, match=why):
        env.execute("place", PlaceArgs(x=0.2, y=0.0))
    assert env.holding == oid and env.observe().objects[oid].pose.z == CARRY_Z  # no pose or holding change
    assert env.observe().gripper.pose.z == CARRY_Z


def test_silent_drop_to_free_location_still_reports_ok():
    env = env_for(silent_drop_on_place=(0.2, 0.0))
    env.execute("pick", PickArgs(object_id="cube_red", x=RED0[0], y=RED0[1]))
    assert env.execute("place", PlaceArgs(x=-0.1, y=0.0)).ok
    s = env.observe()
    assert (s.objects["cube_red"].pose.x, s.objects["cube_red"].pose.y) == (0.2, 0.0)
    assert (s.gripper.pose.x, s.gripper.pose.y) == (-0.1, 0.0) and s.gripper.holding is None


def test_fail_once_place_keeps_cube_held_and_is_consumed():
    env = env_for(fail_once={"place": F.POSTCONDITION_FAILED})
    env.execute("pick", PickArgs(object_id="cube_red", x=RED0[0], y=RED0[1]))
    out = env.execute("place", PlaceArgs(x=0.2, y=0.0))
    assert not out.ok and out.failure.code == "POSTCONDITION_FAILED"
    assert env.holding == "cube_red" and env.observe().objects["cube_red"].pose.z == CARRY_Z
    assert env.execute("place", PlaceArgs(x=0.2, y=0.0)).ok
