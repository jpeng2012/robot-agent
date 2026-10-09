"""Provider-neutral LLM interface plus deterministic scripted clients for credential-free runs.

Message format (adapters for real providers translate to/from this):
  {"role": "user", "content": str}
  {"role": "assistant", "text": str, "tool_calls": [ToolCall, ...]}
  {"role": "tool", "tool_call_id": str, "name": str, "content": dict}
"""

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
