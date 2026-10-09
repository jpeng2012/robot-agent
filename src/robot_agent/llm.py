"""Provider-neutral LLM interface plus deterministic scripted clients for credential-free runs.

Message format (adapters for real providers translate to/from this):
  {"role": "user", "content": str}
  {"role": "assistant", "text": str, "tool_calls": [ToolCall, ...]}
  {"role": "tool", "tool_call_id": str, "name": str, "content": dict}
"""

import json
import re
import urllib.error
import urllib.request
from itertools import count
from typing import Any, Protocol

from pydantic import BaseModel

from robot_agent.world import WorldState


class ToolCall(BaseModel):
    id: str
    name: str
    args: Any  # raw, unvalidated; may be malformed


class AssistantTurn(BaseModel):
    text: str = ""
    tool_calls: list[ToolCall] = []
    meta: dict[str, Any] = {}  # backend-specific extras (reasoning, token usage, ...), traced verbatim


class LLMClient(Protocol):
    def step(self, system: str, messages: list[dict], tools: list[dict]) -> AssistantTurn: ...


_ids = count()


def call(name: str, **args: Any) -> ToolCall:
    return ToolCall(id=f"call_{next(_ids)}", name=name, args=args)


class ScriptedLLM:
    """Replays a fixed list of turns; returns an empty turn once exhausted."""

    def __init__(self, turns: list[list[ToolCall]]):
        self.turns = list(turns)

    def step(self, system: str, messages: list[dict], tools: list[dict]) -> AssistantTurn:
        if not self.turns:
            return AssistantTurn(text="script exhausted")
        return AssistantTurn(tool_calls=self.turns.pop(0))


def latest_state(messages: list[dict]) -> WorldState | None:
    for m in reversed(messages):
        if m["role"] == "tool" and "world_state" in m["content"]:
            return WorldState.model_validate(m["content"]["world_state"])
    return None


class PickPlacePolicyLLM:
    """Rule-based stand-in for an LLM: always grounds its next action on the latest observation."""

    def __init__(self, obj: str, support: str, tol: float = 0.05):
        self.obj, self.support, self.tol = obj, support, tol

    def step(self, system: str, messages: list[dict], tools: list[dict]) -> AssistantTurn:
        s = latest_state(messages)
        if s is None:
            return AssistantTurn(text="No observation yet.", tool_calls=[call("observe")])
        o, sup = s.objects[self.obj].pose, s.objects[self.support].pose
        if s.gripper.holding == self.obj:
            return AssistantTurn(text=f"Holding {self.obj}; placing on {self.support}.",
                                 tool_calls=[call("place", x=sup.x, y=sup.y)])
        if o.dist_xy(sup.x, sup.y) <= self.tol:
            return AssistantTurn(text="Object appears to be on the support.",
                                 tool_calls=[call("finish", summary=f"{self.obj} is on {self.support}")])
        return AssistantTurn(text=f"Picking {self.obj} at its latest observed pose.",
                             tool_calls=[call("pick", object_id=self.obj, x=o.x, y=o.y)])


def to_openai_tools(tools: list[dict]) -> list[dict]:
    return [
        {"type": "function",
         "function": {"name": t["name"], "description": t["description"], "parameters": t["input_schema"]}}
        for t in tools
    ]


def to_openai_messages(system: str, messages: list[dict]) -> list[dict]:
    """Translate the loop's message format into OpenAI chat messages, preserving tool-call IDs."""
    out: list[dict] = [{"role": "system", "content": system}]
    for m in messages:
        if m["role"] == "user":
            out.append({"role": "user", "content": m["content"]})
        elif m["role"] == "assistant":
            msg: dict[str, Any] = {"role": "assistant", "content": m["text"]}
            if m["tool_calls"]:
                msg["tool_calls"] = [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.name,
                                  "arguments": tc.args if isinstance(tc.args, str) else json.dumps(tc.args)}}
                    for tc in m["tool_calls"]
                ]
            out.append(msg)
        elif m["role"] == "tool":
            out.append({"role": "tool", "tool_call_id": m["tool_call_id"],
                        "content": json.dumps(m["content"], separators=(",", ":"))})
    return out


def _parse_args(raw: Any) -> Any:
    """Decode JSON arguments; keep undecodable payloads raw so skill validation rejects them."""
    if not isinstance(raw, str):
        return raw
    try:
        return json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        return raw


_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


class OllamaLLMClient:
    """Tool-calling client for Ollama's OpenAI-compatible endpoint (stdlib HTTP, no SDK)."""

    def __init__(self, model: str = "qwen3:8b", base_url: str = "http://localhost:11434/v1",
                 think: bool = False, temperature: float = 0.0, timeout: float = 300.0):
        self.model, self.base_url = model, base_url.rstrip("/")
        self.think, self.temperature, self.timeout = think, temperature, timeout

    def step(self, system: str, messages: list[dict], tools: list[dict]) -> AssistantTurn:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": to_openai_messages(system, messages),
            "tools": to_openai_tools(tools),
            "temperature": self.temperature,
        }
        if not self.think:
            body["reasoning_effort"] = "none"  # Ollama: disables Qwen3 thinking
        resp = self._post("/chat/completions", body)
        choice = resp["choices"][0]
        msg = choice["message"]
        calls = [
            ToolCall(id=c.get("id") or f"call_{next(_ids)}", name=c["function"]["name"],
                     args=_parse_args(c["function"].get("arguments")))
            for c in msg.get("tool_calls") or []
        ]
        content = msg.get("content") or ""
        return AssistantTurn(
            text=_THINK.sub("", content).strip(),
            tool_calls=calls,
            meta={"model": resp.get("model"), "finish_reason": choice.get("finish_reason"),
                  "reasoning": msg.get("reasoning"), "usage": resp.get("usage")},
        )

    def _post(self, path: str, body: dict) -> dict:
        req = urllib.request.Request(self.base_url + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"Ollama HTTP {e.code}: {e.read().decode(errors='replace')[:500]}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"cannot reach Ollama at {self.base_url}: {e.reason}") from e
