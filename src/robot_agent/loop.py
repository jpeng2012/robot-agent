"""Reactive observe -> act -> verify loop with bounded retries and goal-gated termination."""

import time
from pathlib import Path

from pydantic import BaseModel

from robot_agent.env import RobotEnv
from robot_agent.goals import TaskSpec, check_goal
from robot_agent.llm import AssistantTurn, LLMClient, ToolCall
from robot_agent.skills import precondition, tool_schemas, validate_call, verify
from robot_agent.trace import Tracer
from robot_agent.world import TRANSIENT, Failure, FailureCode, WorldState

SYSTEM_PROMPT = """\
You control a robot arm on a tabletop by calling tools.
- Ground every action on the most recent world_state; object poses can change at any time.
- Every tool result includes a fresh world_state, also after failures.
- Use exactly the parameter names defined in each tool's schema.
- Failed results include an error {code, detail}. Read it and choose a new action;
  transient failures have already been retried for you. If a call fails, the remaining
  calls in that turn are skipped.
- Call finish only when the task is done. The goal is verified independently and
  finish is rejected with GOAL_NOT_MET if it is not satisfied."""


NUDGE = "No tool was called. Continue the task by calling a tool; call finish only when the task is done."


class EpisodeResult(BaseModel):
    success: bool
    reason: str  # goal_met | max_steps | failure_budget | llm_stopped | llm_error
    llm_turns: int  # model calls
    tool_calls: int
    skills_executed: int
    retries: int
    failures: int  # failed tool results returned to the LLM (counts against max_failures)
    llm_latency_s: float
    wall_time_s: float
    trace_path: Path


class AgentLoop:
    def __init__(self, env: RobotEnv, llm: LLMClient, trace_dir: Path | str = "traces",
                 max_turns: int = 20, max_retries: int = 2, max_failures: int = 8):
        self.env, self.llm = env, llm
        self.trace_dir = Path(trace_dir)
        self.max_turns, self.max_retries, self.max_failures = max_turns, max_retries, max_failures

    def run(self, task: TaskSpec, seed: int | None = None) -> EpisodeResult:
        self.tracer = Tracer(self.trace_dir, task.name)
        self.retries = self.failures = self.n_tool_calls = 0
        self.llm_latency = 0.0
        self.t0 = time.perf_counter()
        state = self.env.reset(task, seed)
        self.tracer.log("episode_start", task=task, seed=seed, state=state, llm=type(self.llm).__name__,
                        model=getattr(self.llm, "model", None), max_turns=self.max_turns,
                        max_retries=self.max_retries, max_failures=self.max_failures)
        messages: list[dict] = [{"role": "user", "content": f"Task: {task.instruction}"}]
        tools = tool_schemas()
        idle = False
        try:
            for turn_idx in range(self.max_turns):
                t = time.perf_counter()
                try:
                    turn: AssistantTurn = self.llm.step(SYSTEM_PROMPT, messages, tools)
                except Exception as e:  # provider/transport failure: end the episode, keep the trace
                    self.tracer.log("llm_error", turn=turn_idx, error=repr(e))
                    return self._end(task, False, "llm_error", turn_idx + 1)
                latency = time.perf_counter() - t
                self.llm_latency += latency
                self.tracer.log("llm_turn", turn=turn_idx, latency_s=round(latency, 3), text=turn.text,
                                tool_calls=turn.tool_calls, meta=turn.meta)
                messages.append({"role": "assistant", "text": turn.text, "tool_calls": turn.tool_calls})
                if not turn.tool_calls:
                    # Real models sometimes narrate an intended action without calling a tool.
                    # Nudge once; two text-only turns in a row end the episode.
                    if idle:
                        return self._end(task, False, "llm_stopped", turn_idx + 1)
                    idle = True
                    self.tracer.log("nudge", turn=turn_idx)
                    messages.append({"role": "user", "content": NUDGE})
                    continue
                idle = False
                turn_failed = False
                for tc in turn.tool_calls:
                    self.n_tool_calls += 1
                    if turn_failed:  # later calls were planned on grounding that is now stale
                        skip = Failure(code=FailureCode.SKIPPED, detail="an earlier call in this turn failed")
                        content, done = _error(skip, self.env.observe()), False
                        self.tracer.log("tool_call", call_id=tc.id, name=tc.name, args=tc.args, skipped=True)
                    else:
                        content, done = self._handle(task, tc)
                        if not content["ok"]:
                            turn_failed = True
                            self.failures += 1
                    self.tracer.log("tool_result", call_id=tc.id, ok=content["ok"], error=content.get("error"),
                                    world_state=content.get("world_state"))
                    messages.append({"role": "tool", "tool_call_id": tc.id, "name": tc.name, "content": content})
                    if done:
                        return self._end(task, True, "goal_met", turn_idx + 1)
                if self.failures >= self.max_failures:
                    return self._end(task, False, "failure_budget", turn_idx + 1)
            return self._end(task, False, "max_steps", self.max_turns)
        finally:
            self.tracer.close()

    def _handle(self, task: TaskSpec, tc: ToolCall) -> tuple[dict, bool]:
        """Run one tool call. Returns (tool_result content, episode_done)."""
        self.tracer.log("tool_call", call_id=tc.id, name=tc.name, args=tc.args)
        before = self.env.observe()
        args = validate_call(tc.name, tc.args, before)
        if isinstance(args, Failure):
            self.tracer.log("validation_error", call_id=tc.id, failure=args)
            return _error(args, before), False

        if tc.name == "observe":
            return _ok(before), False
        if tc.name == "finish":
            unmet = check_goal(task, before)
            self.tracer.log("goal_check", call_id=tc.id, passed=not unmet, unmet=unmet)
            if unmet:
                return _error(Failure(code=FailureCode.GOAL_NOT_MET, detail=f"unmet: {unmet}"), before), False
            return {"ok": True, "goal_met": True, "world_state": before.model_dump(mode="json")}, True

        attempt = 0
        while True:
            outcome = self.env.execute(tc.name, args)
            after = self.env.observe()  # refresh on success and failure alike
            failure = verify(tc.name, args, before, outcome, after)
            self.tracer.log("env_outcome", call_id=tc.id, attempt=attempt, outcome=outcome, state_after=after)
            self.tracer.log("verify", call_id=tc.id, attempt=attempt, ok=failure is None, failure=failure)
            if failure is None:
                return _ok(after, attempt), False
            # Retry only transient failures, within budget, and only if the refreshed state
            # still satisfies the call's preconditions (i.e. the original grounding is valid).
            retryable = (failure.code in TRANSIENT and attempt < self.max_retries
                         and precondition(tc.name, args, after) is None)
            if retryable:
                attempt += 1
                self.retries += 1
                self.tracer.log("retry", call_id=tc.id, attempt=attempt, reason=failure.code)
                before = after
                continue
            return _error(failure, after, attempt), False

    def _end(self, task: TaskSpec, success: bool, reason: str, turns: int) -> EpisodeResult:
        final = self.env.observe()
        result = EpisodeResult(
            success=success, reason=reason, llm_turns=turns, tool_calls=self.n_tool_calls,
            skills_executed=final.step, retries=self.retries, failures=self.failures,
            llm_latency_s=round(self.llm_latency, 3), wall_time_s=round(time.perf_counter() - self.t0, 3),
            trace_path=self.tracer.path,
        )
        self.tracer.log("episode_end", result=result, unmet=check_goal(task, final), state=final)
        return result


def _ok(state: WorldState, retries_used: int = 0) -> dict:
    return {"ok": True, "retries_used": retries_used, "world_state": state.model_dump(mode="json")}


def _error(failure: Failure, state: WorldState, retries_used: int = 0) -> dict:
    return {
        "ok": False,
        "error": failure.model_dump(mode="json"),
        "retries_used": retries_used,
        "world_state": state.model_dump(mode="json"),
    }
