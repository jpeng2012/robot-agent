"""OllamaLLMClient conversion + loop integration, with the HTTP layer stubbed (no server needed)."""

import json
import os

import pytest

from robot_agent.env import Faults, MockTabletopEnv
from robot_agent.goals import pick_place_task
from robot_agent.llm import OllamaLLMClient, ToolCall, to_openai_messages
from robot_agent.loop import AgentLoop
from robot_agent.skills import tool_schemas


def response(*calls, content="", reasoning=None):
    """Build an OpenAI-style chat completion containing the given (id, name, arguments) calls."""
    msg = {"role": "assistant", "content": content, "reasoning": reasoning,
           "tool_calls": [{"id": i, "type": "function", "function": {"name": n, "arguments": a}}
                          for i, n, a in calls]}
    return {"model": "qwen3:8b", "choices": [{"message": msg, "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}}


class FakeOllama(OllamaLLMClient):
    def __init__(self, responses):
        super().__init__()
        self.responses, self.requests = list(responses), []

    def _post(self, path, body):
        self.requests.append(json.loads(json.dumps(body)))  # must be JSON-serializable
        return self.responses.pop(0)


def test_message_conversion_preserves_ids_and_results():
    msgs = [
        {"role": "user", "content": "Task: x"},
        {"role": "assistant", "text": "", "tool_calls": [ToolCall(id="call_a", name="observe", args={})]},
        {"role": "tool", "tool_call_id": "call_a", "name": "observe", "content": {"ok": True, "n": 1}},
    ]
    out = to_openai_messages("sys", msgs)
    assert out[0] == {"role": "system", "content": "sys"}
    assert out[2]["tool_calls"] == [
        {"id": "call_a", "type": "function", "function": {"name": "observe", "arguments": "{}"}}]
    assert out[3]["role"] == "tool" and out[3]["tool_call_id"] == "call_a"
    assert json.loads(out[3]["content"]) == {"ok": True, "n": 1}


def test_step_parses_calls_and_disables_thinking():
    llm = FakeOllama([response(("c1", "pick", '{"object_id": "cube_red", "x": 0.1, "y": -0.2}'),
                               content="<think>hmm</think>picking", reasoning="r")])
    turn = llm.step("sys", [{"role": "user", "content": "go"}], tool_schemas())
    assert turn.text == "picking"
    assert turn.tool_calls == [ToolCall(id="c1", name="pick", args={"object_id": "cube_red", "x": 0.1, "y": -0.2})]
    assert turn.meta["reasoning"] == "r" and turn.meta["usage"]["prompt_tokens"] == 10
    req = llm.requests[0]
    assert req["model"] == "qwen3:8b" and req["reasoning_effort"] == "none"
    assert req["tools"][1]["function"]["name"] == "pick"
    assert req["tools"][1]["function"]["parameters"]["required"] == ["object_id", "x", "y"]


def test_full_episode_round_trips_ids_and_handles_bad_json(tmp_path):
    llm = FakeOllama([
        response(("c1", "observe", "{}")),
        response(("c2", "pick", '{"object_id": "cube_red", "x": 0.1,')),  # truncated JSON
        response(("c3", "pick", '{"object_id": "cube_red", "x": 0.1, "y": -0.2}'),
                 ("c4", "place", '{"x": -0.2, "y": 0.15}')),  # two sequential calls in one turn
        response(("c5", "finish", '{"summary": "done"}')),
    ])
    result = AgentLoop(MockTabletopEnv(), llm, trace_dir=tmp_path).run(pick_place_task())
    assert result.success and result.llm_turns == 4 and result.tool_calls == 5 and result.failures == 1
    last = llm.requests[-1]["messages"]
    call_ids = [tc["id"] for m in last if m["role"] == "assistant" for tc in m.get("tool_calls", [])]
    result_ids = [m["tool_call_id"] for m in last if m["role"] == "tool"]
    assert call_ids == result_ids == ["c1", "c2", "c3", "c4"]
    bad = json.loads(next(m for m in last if m.get("tool_call_id") == "c2")["content"])
    assert bad["error"]["code"] == "INVALID_ARGS"


def test_calls_after_a_failure_in_same_turn_are_skipped(tmp_path):
    llm = FakeOllama([
        response(("c1", "pick", '{"object_id": "cube_red", "x": 0.1, "y": -0.2}'),
                 ("c2", "place", '{"x": -0.2, "y": 0.15}')),
    ])
    env = MockTabletopEnv(Faults(perturb_at={0: ("cube_red", 0.25, 0.05)}))
    loop = AgentLoop(env, llm, trace_dir=tmp_path)
    loop.run(pick_place_task())
    codes = [e["error"]["code"] for e in loop.tracer.events if e["event"] == "tool_result"]
    assert codes == ["OBJECT_NOT_AT_POSE", "SKIPPED"]
    assert env.execute_calls == 1


def test_failure_budget_ends_episode(tmp_path):
    bad = response(("c", "pick", '{"object": "cube_red"}'))
    llm = FakeOllama([bad] * 10)
    result = AgentLoop(MockTabletopEnv(), llm, trace_dir=tmp_path, max_failures=3).run(pick_place_task())
    assert not result.success and result.reason == "failure_budget" and result.failures == 3


def test_transport_error_ends_episode_with_trace(tmp_path):
    llm = OllamaLLMClient(base_url="http://127.0.0.1:9/v1", timeout=2)
    result = AgentLoop(MockTabletopEnv(), llm, trace_dir=tmp_path).run(pick_place_task())
    assert result.reason == "llm_error"
    assert any(json.loads(line)["event"] == "llm_error" for line in result.trace_path.read_text().splitlines())


@pytest.mark.skipif(not os.environ.get("RUN_OLLAMA"), reason="set RUN_OLLAMA=1 to hit a live Ollama server")
def test_live_qwen_pick_place(tmp_path):
    result = AgentLoop(MockTabletopEnv(), OllamaLLMClient(), trace_dir=tmp_path).run(pick_place_task())
    assert result.success
