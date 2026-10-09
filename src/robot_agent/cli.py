"""Run one scripted episode: `uv run robot-agent --scenario perturb`."""

import argparse

from robot_agent.env import Faults, MockTabletopEnv
from robot_agent.goals import pick_place_task
from robot_agent.llm import PickPlacePolicyLLM, ScriptedLLM, call
from robot_agent.loop import AgentLoop
from robot_agent.world import FailureCode


def build(scenario: str):
    task = pick_place_task()
    policy = PickPlacePolicyLLM("cube_red", "plate")
    if scenario == "happy":
        return task, Faults(), policy
    if scenario == "slip":
        return task, Faults(fail_once={"pick": FailureCode.GRASP_SLIP}), policy
    if scenario == "perturb":
        return task, Faults(perturb_at={0: ("cube_red", 0.25, 0.05)}), policy
    if scenario == "silent_drop":
        return task, Faults(silent_drop_on_place=(-0.05, 0.30)), policy
    if scenario == "premature_finish":
        return task, Faults(), ScriptedLLM([[call("observe")], [call("finish", summary="done")]])
    raise ValueError(scenario)


SCENARIOS = ["happy", "slip", "perturb", "silent_drop", "premature_finish"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=SCENARIOS, default="happy")
    ap.add_argument("--trace-dir", default="traces")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    task, faults, llm = build(a.scenario)
    result = AgentLoop(MockTabletopEnv(faults), llm, trace_dir=a.trace_dir).run(task, seed=a.seed)
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
