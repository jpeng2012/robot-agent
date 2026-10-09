"""RobotEnv backed by robosuite, running in a separate interpreter.

robosuite/MuJoCo live in a conda env (Python 3.10) that cannot import robot_agent (Python >= 3.11),
so `robosuite_worker.py` runs there and this client talks to it over JSON lines on stdin/stdout.
Point `ROBOT_AGENT_ROBOSUITE_PYTHON` (or `python=`) at that env's interpreter.
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path

from pydantic import BaseModel

from robot_agent.goals import InRegion, TaskSpec
from robot_agent.world import SkillOutcome, WorldState

WORKER = Path(__file__).with_name("robosuite_worker.py")
PYTHON_ENV_VAR = "ROBOT_AGENT_ROBOSUITE_PYTHON"


class RobosuiteEnv:
    def __init__(self, python: str | None = None, render: bool = False):
        python = python or os.environ.get(PYTHON_ENV_VAR)
        if not python:
            raise RuntimeError(f"set {PYTHON_ENV_VAR} to the python of the conda env that has robosuite")
        self._log = tempfile.NamedTemporaryFile("w+", prefix="robosuite_worker_", suffix=".log", delete=False)
        # -I: don't put the worker's directory (which has a module named `trace`) on sys.path.
        self._proc = subprocess.Popen([python, "-I", str(WORKER)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                      stderr=self._log, text=True, cwd=WORKER.parent)
        self._rpc("init", render=render)
        self.backend_info: dict = {}

    def _rpc(self, op: str, **kw):
        try:
            self._proc.stdin.write(json.dumps({"op": op, **kw}) + "\n")
            self._proc.stdin.flush()
            line = self._proc.stdout.readline()
        except BrokenPipeError:
            line = ""
        if not line:
            raise RuntimeError(f"robosuite worker exited during {op!r}; log tail:\n{self._log_tail()}")
        reply = json.loads(line)
        if not reply["ok"]:
            raise RuntimeError(f"robosuite worker failed on {op!r}:\n{reply['error']}")
        return reply["result"]

    def _log_tail(self, n: int = 4000) -> str:
        self._log.flush()
        return Path(self._log.name).read_text()[-n:]

    def reset(self, task: TaskSpec, seed: int | None = None) -> WorldState:
        state = WorldState.model_validate(self._rpc("reset", layout=task.layout, seed=seed))
        self.backend_info = self._rpc("info")
        hx, hy = self.backend_info["target_half_size"]
        for g in task.goal:  # the goal's region must be the one the simulator draws and checks against
            if isinstance(g, InRegion) and (abs(g.half_x - hx) > 1e-9 or abs(g.half_y - hy) > 1e-9):
                raise ValueError(f"{g} half sizes ({g.half_x}, {g.half_y}) != simulator target ({hx}, {hy})")
        return state

    def observe(self) -> WorldState:
        return WorldState.model_validate(self._rpc("observe"))

    def execute(self, name: str, args: BaseModel) -> SkillOutcome:
        return SkillOutcome.model_validate(self._rpc("execute", name=name, args=args.model_dump()))

    def close(self) -> None:
        if self._proc.poll() is None:
            try:
                self._rpc("close")
            except RuntimeError:
                pass
            self._proc.wait(timeout=30)
        self._log.close()
        os.unlink(self._log.name)

    def __enter__(self) -> "RobosuiteEnv":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
