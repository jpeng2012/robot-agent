"""Run episodes: `uv run robot-agent --backend ollama --scenario perturb --repeat 3`."""

import argparse
import json

from robot_agent.env import Faults, MockTabletopEnv
from robot_agent.goals import pick_place_task
from robot_agent.llm import OllamaLLMClient, PickPlacePolicyLLM, ScriptedLLM, call
from robot_agent.loop import AgentLoop
from robot_agent.world import FailureCode

SCENARIOS = {
    "happy": Faults,
    "slip": lambda: Faults(fail_once={"pick": FailureCode.GRASP_SLIP}),
    "perturb": lambda: Faults(perturb_at={0: ("cube_red", 0.25, 0.05)}),
    "silent_drop": lambda: Faults(silent_drop_on_place=(-0.05, 0.30)),
    "premature_finish": Faults,
}


def make_llm(a: argparse.Namespace):
    if a.backend == "ollama":
        return OllamaLLMClient(model=a.model, base_url=a.base_url, think=a.think, temperature=a.temperature)
    if a.backend == "policy":
        return PickPlacePolicyLLM("cube_red", "plate")
    if a.scenario == "premature_finish":
        return ScriptedLLM([[call("observe")], [call("finish", summary="done")]])
    # Fixed open-loop script: succeeds only when nothing goes wrong.
    return ScriptedLLM([
        [call("observe")],
        [call("pick", object_id="cube_red", x=0.10, y=-0.20)],
        [call("place", x=-0.20, y=0.15)],
        [call("finish", summary="cube on plate")],
    ])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["scripted", "policy", "ollama"], default="policy")
    ap.add_argument("--scenario", choices=list(SCENARIOS), default="happy")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--max-turns", type=int, default=20)
    ap.add_argument("--max-failures", type=int, default=8, help="global budget of failed tool results")
    ap.add_argument("--max-retries", type=int, default=2, help="automatic retries per call (transient only)")
    ap.add_argument("--model", default="qwen3:8b")
    ap.add_argument("--base-url", default="http://localhost:11434/v1")
    ap.add_argument("--think", action="store_true", help="enable Qwen3 thinking (slower)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--trace-dir", default="traces")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    for i in range(a.repeat):
        loop = AgentLoop(MockTabletopEnv(SCENARIOS[a.scenario]()), make_llm(a), trace_dir=a.trace_dir,
                         max_turns=a.max_turns, max_retries=a.max_retries, max_failures=a.max_failures)
        result = loop.run(pick_place_task(), seed=a.seed + i)
        label = {"backend": a.backend, "model": a.model if a.backend == "ollama" else None,
                 "think": a.think if a.backend == "ollama" else None, "scenario": a.scenario}
        print(json.dumps({**label, **result.model_dump(mode="json")}))


if __name__ == "__main__":
    main()
