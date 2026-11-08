"""Tests for persistent episodic memory (F009a)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from robot_agent.memory import EpisodeMemory, MemoryStore


@pytest.fixture(scope="function")
def temp_traces():
    """Create temporary directory with test trace files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        trace_dir = Path(tmpdir) / "traces"
        trace_dir.mkdir()
        yield trace_dir


@pytest.fixture
def successful_trace(temp_traces):
    """Create a successful episode trace."""
    trace = [
        {
            "seq": 0,
            "ts": 1000.0,
            "event": "episode_start",
            "task": {
                "name": "pick_place",
                "instruction": "Put the red cube on the plate.",
                "layout": {"cube_red": ["cube", 0.1, -0.2], "plate": ["plate", -0.2, 0.15]},
                "goal": [{"obj": "cube_red", "support": "plate", "tol": 0.05}],
            },
            "seed": 42,
            "llm": "OllamaLLM",
            "model": "qwen2.5:latest",
        },
        {
            "seq": 1,
            "ts": 1000.1,
            "event": "llm_turn",
            "turn": 0,
            "text": "I need to observe the scene first.",
            "tool_calls": [{"id": "call_0", "name": "observe", "args": {}}],
        },
        {
            "seq": 2,
            "ts": 1000.2,
            "event": "tool_call",
            "call_id": "call_0",
            "name": "observe",
            "args": {},
        },
        {
            "seq": 3,
            "ts": 1000.3,
            "event": "tool_result",
            "call_id": "call_0",
            "ok": True,
            "error": None,
        },
        {
            "seq": 4,
            "ts": 1000.4,
            "event": "llm_turn",
            "turn": 1,
            "text": "Picking the red cube.",
            "tool_calls": [{"id": "call_1", "name": "pick", "args": {"object_id": "cube_red", "x": 0.1, "y": -0.2}}],
        },
        {
            "seq": 5,
            "ts": 1000.5,
            "event": "tool_call",
            "call_id": "call_1",
            "name": "pick",
            "args": {"object_id": "cube_red", "x": 0.1, "y": -0.2},
        },
        {
            "seq": 6,
            "ts": 1000.6,
            "event": "env_outcome",
            "call_id": "call_1",
            "attempt": 0,
            "outcome": {"ok": True, "failure": None},
            "state_after": {"t": 0.5, "step": 1},
        },
        {
            "seq": 7,
            "ts": 1000.7,
            "event": "verify",
            "call_id": "call_1",
            "attempt": 0,
            "ok": True,
            "failure": None,
        },
        {
            "seq": 8,
            "ts": 1000.8,
            "event": "tool_result",
            "call_id": "call_1",
            "ok": True,
            "error": None,
        },
        {
            "seq": 9,
            "ts": 1001.0,
            "event": "llm_turn",
            "turn": 2,
            "text": "Placing cube on plate.",
            "tool_calls": [{"id": "call_2", "name": "place", "args": {"x": -0.2, "y": 0.15}}],
        },
        {
            "seq": 10,
            "ts": 1001.1,
            "event": "tool_call",
            "call_id": "call_2",
            "name": "place",
            "args": {"x": -0.2, "y": 0.15},
        },
        {
            "seq": 11,
            "ts": 1001.2,
            "event": "env_outcome",
            "call_id": "call_2",
            "attempt": 0,
            "outcome": {"ok": True, "failure": None},
            "state_after": {"t": 1.0, "step": 2},
        },
        {
            "seq": 12,
            "ts": 1001.3,
            "event": "verify",
            "call_id": "call_2",
            "attempt": 0,
            "ok": True,
            "failure": None,
        },
        {
            "seq": 13,
            "ts": 1001.4,
            "event": "tool_result",
            "call_id": "call_2",
            "ok": True,
            "error": None,
        },
        {
            "seq": 14,
            "ts": 1001.5,
            "event": "llm_turn",
            "turn": 3,
            "text": "Task complete.",
            "tool_calls": [{"id": "call_3", "name": "finish", "args": {"summary": "Done"}}],
        },
        {
            "seq": 15,
            "ts": 1001.6,
            "event": "tool_call",
            "call_id": "call_3",
            "name": "finish",
            "args": {"summary": "Done"},
        },
        {
            "seq": 16,
            "ts": 1001.7,
            "event": "goal_check",
            "call_id": "call_3",
            "passed": True,
            "unmet": [],
        },
        {
            "seq": 17,
            "ts": 1001.8,
            "event": "tool_result",
            "call_id": "call_3",
            "ok": True,
            "error": None,
        },
        {
            "seq": 18,
            "ts": 1002.0,
            "event": "episode_end",
            "result": {
                "success": True,
                "reason": "goal_met",
                "llm_turns": 4,
                "tool_calls": 4,
                "skills_executed": 2,
                "retries": 0,
                "failures": 0,
            },
            "unmet": [],
        },
    ]

    path = temp_traces / "pick_place_success_abc123.jsonl"
    with path.open("w") as f:
        for event in trace:
            f.write(json.dumps(event) + "\n")

    return path


@pytest.fixture
def failed_trace_with_recovery(temp_traces):
    """Create a failed episode with recovery attempts."""
    trace = [
        {
            "seq": 0,
            "ts": 2000.0,
            "event": "episode_start",
            "task": {
                "name": "pick_place",
                "instruction": "Put the red cube on the plate.",
                "layout": {"cube_red": ["cube", 0.1, -0.2], "plate": ["plate", -0.2, 0.15]},
                "goal": [{"obj": "cube_red", "support": "plate", "tol": 0.05}],
            },
            "seed": 42,
            "llm": "OllamaLLM",
            "model": "qwen2.5:latest",
        },
        {
            "seq": 1,
            "ts": 2000.1,
            "event": "llm_turn",
            "turn": 0,
            "text": "Attempting pick.",
            "tool_calls": [{"id": "call_0", "name": "pick", "args": {"object_id": "cube_red", "x": 0.1, "y": -0.2}}],
        },
        {
            "seq": 2,
            "ts": 2000.2,
            "event": "tool_call",
            "call_id": "call_0",
            "name": "pick",
            "args": {"object_id": "cube_red", "x": 0.1, "y": -0.2},
        },
        {
            "seq": 3,
            "ts": 2000.3,
            "event": "env_outcome",
            "call_id": "call_0",
            "attempt": 0,
            "outcome": {"ok": False, "failure": {"code": "GRASP_SLIP", "detail": "Object slipped"}},
            "state_after": {"t": 0.5, "step": 1},
        },
        {
            "seq": 4,
            "ts": 2000.4,
            "event": "verify",
            "call_id": "call_0",
            "attempt": 0,
            "ok": False,
            "failure": {"code": "GRASP_SLIP", "detail": "Object slipped"},
        },
        {
            "seq": 5,
            "ts": 2000.5,
            "event": "retry",
            "call_id": "call_0",
            "attempt": 1,
            "reason": "GRASP_SLIP",
        },
        {
            "seq": 6,
            "ts": 2000.6,
            "event": "env_outcome",
            "call_id": "call_0",
            "attempt": 1,
            "outcome": {"ok": False, "failure": {"code": "OBJECT_NOT_AT_POSE", "detail": "Object moved"}},
            "state_after": {"t": 1.0, "step": 2},
        },
        {
            "seq": 7,
            "ts": 2000.7,
            "event": "verify",
            "call_id": "call_0",
            "attempt": 1,
            "ok": False,
            "failure": {"code": "OBJECT_NOT_AT_POSE", "detail": "Object moved"},
        },
        {
            "seq": 8,
            "ts": 2000.8,
            "event": "tool_result",
            "call_id": "call_0",
            "ok": False,
            "error": {"code": "OBJECT_NOT_AT_POSE", "detail": "Object moved"},
        },
        {
            "seq": 9,
            "ts": 2001.0,
            "event": "llm_turn",
            "turn": 1,
            "text": "Re-observing after failure.",
            "tool_calls": [{"id": "call_1", "name": "observe", "args": {}}],
        },
        {
            "seq": 10,
            "ts": 2001.1,
            "event": "tool_call",
            "call_id": "call_1",
            "name": "observe",
            "args": {},
        },
        {
            "seq": 11,
            "ts": 2001.2,
            "event": "tool_result",
            "call_id": "call_1",
            "ok": True,
            "error": None,
        },
        {
            "seq": 12,
            "ts": 2002.0,
            "event": "episode_end",
            "result": {
                "success": False,
                "reason": "llm_stopped",
                "llm_turns": 2,
                "tool_calls": 2,
                "skills_executed": 0,
                "retries": 1,
                "failures": 2,
            },
            "unmet": ["on(cube_red, plate)"],
        },
    ]

    path = temp_traces / "pick_place_failed_def456.jsonl"
    with path.open("w") as f:
        for event in trace:
            f.write(json.dumps(event) + "\n")

    return path


@pytest.fixture
def scripted_trace(temp_traces):
    """Create a ScriptedLLM trace (should be excluded)."""
    trace = [
        {
            "seq": 0,
            "ts": 3000.0,
            "event": "episode_start",
            "task": {
                "name": "pick_place",
                "instruction": "Put the red cube on the plate.",
                "layout": {"cube_red": ["cube", 0.1, -0.2], "plate": ["plate", -0.2, 0.15]},
                "goal": [{"obj": "cube_red", "support": "plate", "tol": 0.05}],
            },
            "seed": 0,
            "llm": "ScriptedLLM",
            "model": None,
        },
        {
            "seq": 1,
            "ts": 3000.1,
            "event": "goal_check",
            "call_id": "call_0",
            "passed": True,
            "unmet": [],
        },
        {
            "seq": 2,
            "ts": 3000.2,
            "event": "episode_end",
            "result": {
                "success": True,
                "reason": "goal_met",
                "llm_turns": 1,
                "tool_calls": 1,
                "skills_executed": 1,
                "retries": 0,
                "failures": 0,
            },
            "unmet": [],
        },
    ]

    path = temp_traces / "pick_place_scripted_ghi789.jsonl"
    with path.open("w") as f:
        for event in trace:
            f.write(json.dumps(event) + "\n")

    return path


@pytest.fixture
def incomplete_trace(temp_traces):
    """Create an incomplete trace (missing episode_end)."""
    trace = [
        {
            "seq": 0,
            "ts": 4000.0,
            "event": "episode_start",
            "task": {
                "name": "pick_place",
                "instruction": "Put the red cube on the plate.",
                "layout": {"cube_red": ["cube", 0.1, -0.2], "plate": ["plate", -0.2, 0.15]},
                "goal": [{"obj": "cube_red", "support": "plate", "tol": 0.05}],
            },
            "seed": 42,
            "llm": "OllamaLLM",
            "model": "qwen2.5:latest",
        },
        {
            "seq": 1,
            "ts": 4000.1,
            "event": "llm_turn",
            "turn": 0,
            "text": "Starting task.",
            "tool_calls": [],
        },
        # Missing episode_end
    ]

    path = temp_traces / "pick_place_incomplete_jkl012.jsonl"
    with path.open("w") as f:
        for event in trace:
            f.write(json.dumps(event) + "\n")

    return path


@pytest.fixture
def unverified_success_trace(temp_traces):
    """Create a trace marked success but goal_check failed."""
    trace = [
        {
            "seq": 0,
            "ts": 5000.0,
            "event": "episode_start",
            "task": {
                "name": "pick_place",
                "instruction": "Put the red cube on the plate.",
                "layout": {"cube_red": ["cube", 0.1, -0.2], "plate": ["plate", -0.2, 0.15]},
                "goal": [{"obj": "cube_red", "support": "plate", "tol": 0.05}],
            },
            "seed": 42,
            "llm": "OllamaLLM",
            "model": "qwen2.5:latest",
        },
        {
            "seq": 1,
            "ts": 5000.1,
            "event": "goal_check",
            "call_id": "call_0",
            "passed": False,  # Goal NOT met
            "unmet": ["on(cube_red, plate)"],
        },
        {
            "seq": 2,
            "ts": 5000.2,
            "event": "episode_end",
            "result": {
                "success": True,  # Incorrectly marked as success
                "reason": "goal_met",
                "llm_turns": 1,
                "tool_calls": 1,
                "skills_executed": 1,
                "retries": 0,
                "failures": 0,
            },
            "unmet": ["on(cube_red, plate)"],
        },
    ]

    path = temp_traces / "pick_place_unverified_mno345.jsonl"
    with path.open("w") as f:
        for event in trace:
            f.write(json.dumps(event) + "\n")

    return path


@pytest.fixture
def malformed_trace(temp_traces):
    """Create a malformed JSON trace."""
    path = temp_traces / "pick_place_malformed_pqr678.jsonl"
    with path.open("w") as f:
        f.write('{"seq": 0, "event": "episode_start"')  # Incomplete JSON
        f.write("\n")
        f.write("{invalid json}")

    return path


class TestEpisodeMemory:
    """Test EpisodeMemory dataclass serialization."""

    def test_to_dict_and_from_dict(self):
        """Test round-trip serialization."""
        memory = EpisodeMemory(
            episode_id="test_123",
            task_name="pick_place",
            timestamp=1000.0,
            success=True,
            reason="goal_met",
            failure_codes=["GRASP_SLIP"],
            llm_reasoning=["Picking cube."],
            recovery_actions=[{"name": "observe", "args_schema": {}}],
            provenance={"llm": "OllamaLLM", "model": "qwen2.5:latest"},
            trace_path="/path/to/trace.jsonl",
        )

        data = memory.to_dict()
        restored = EpisodeMemory.from_dict(data)

        assert restored.episode_id == memory.episode_id
        assert restored.task_name == memory.task_name
        assert restored.timestamp == memory.timestamp
        assert restored.success == memory.success
        assert restored.reason == memory.reason
        assert restored.failure_codes == memory.failure_codes
        assert restored.llm_reasoning == memory.llm_reasoning
        assert restored.recovery_actions == memory.recovery_actions
        assert restored.provenance == memory.provenance
        assert restored.trace_path == memory.trace_path


class TestMemoryStore:
    """Test MemoryStore indexing, retrieval, and filtering."""

    def test_index_successful_episode(self, temp_traces, successful_trace):
        """Test indexing a successful episode."""
        index_path = temp_traces / ".test_index.json"
        store = MemoryStore(trace_dirs=[temp_traces], index_path=index_path)
        count = store.index_episodes()

        assert count == 1
        assert len(store.memories) == 1

        memory = store.memories[0]
        assert memory.episode_id == "pick_place_success_abc123"
        assert memory.task_name == "pick_place"
        assert memory.success is True
        assert memory.reason == "goal_met"
        assert memory.provenance["llm"] == "OllamaLLM"

    def test_index_failed_episode_with_failures(self, temp_traces, failed_trace_with_recovery):
        """Test indexing a failed episode extracts failure codes."""
        index_path = temp_traces / ".test_index.json"
        store = MemoryStore(trace_dirs=[temp_traces], index_path=index_path)
        count = store.index_episodes()

        assert count == 1
        assert len(store.memories) == 1

        memory = store.memories[0]
        assert memory.episode_id == "pick_place_failed_def456"
        assert memory.success is False
        assert "GRASP_SLIP" in memory.failure_codes
        assert "OBJECT_NOT_AT_POSE" in memory.failure_codes

    def test_index_extracts_llm_reasoning(self, temp_traces, successful_trace):
        """Test LLM reasoning text is extracted."""
        index_path = temp_traces / ".test_index.json"
        store = MemoryStore(trace_dirs=[temp_traces], index_path=index_path)
        store.index_episodes()

        memory = store.memories[0]
        assert len(memory.llm_reasoning) > 0
        assert "I need to observe the scene first." in memory.llm_reasoning
        assert "Picking the red cube." in memory.llm_reasoning

    def test_index_extracts_recovery_actions(self, temp_traces, failed_trace_with_recovery):
        """Test recovery actions after failures are extracted."""
        index_path = temp_traces / ".test_index.json"
        store = MemoryStore(trace_dirs=[temp_traces], index_path=index_path)
        store.index_episodes()

        memory = store.memories[0]
        assert len(memory.recovery_actions) > 0

        # After failure, the LLM called observe
        recovery = memory.recovery_actions[0]
        assert recovery["name"] == "observe"

    def test_recovery_actions_exclude_coordinates(self, temp_traces, failed_trace_with_recovery):
        """Test recovery actions don't include x,y coordinates or schema."""
        index_path = temp_traces / ".test_index.json"
        store = MemoryStore(trace_dirs=[temp_traces], index_path=index_path)
        store.index_episodes()

        memory = store.memories[0]
        for action in memory.recovery_actions:
            # Should not have x, y coordinates
            assert "x" not in action
            assert "y" not in action
            # Should not have args_schema (prevents coordinate structure leakage)
            assert "args_schema" not in action
            # Should only contain name and optionally object_id
            assert "name" in action
            # object_id is allowed (safe, no coordinates)
            if "object_id" in action:
                assert isinstance(action["object_id"], str)

    def test_exclude_scripted_llm(self, temp_traces, scripted_trace):
        """Test ScriptedLLM episodes are excluded."""
        index_path = temp_traces / ".test_index.json"
        store = MemoryStore(trace_dirs=[temp_traces], index_path=index_path)
        count = store.index_episodes()

        assert count == 0
        assert len(store.memories) == 0

    def test_exclude_incomplete_episode(self, temp_traces, incomplete_trace):
        """Test incomplete episodes (missing episode_end) are excluded."""
        index_path = temp_traces / ".test_index.json"
        store = MemoryStore(trace_dirs=[temp_traces], index_path=index_path)
        count = store.index_episodes()

        assert count == 0
        assert len(store.memories) == 0

    def test_exclude_unverified_success(self, temp_traces, unverified_success_trace):
        """Test episodes marked success but goal_check failed are excluded."""
        index_path = temp_traces / ".test_index.json"
        store = MemoryStore(trace_dirs=[temp_traces], index_path=index_path)
        count = store.index_episodes()

        assert count == 0
        assert len(store.memories) == 0

    def test_exclude_malformed_trace(self, temp_traces, malformed_trace):
        """Test malformed JSON traces are skipped gracefully."""
        index_path = temp_traces / ".test_index.json"
        store = MemoryStore(trace_dirs=[temp_traces], index_path=index_path)
        count = store.index_episodes()

        assert count == 0
        assert len(store.memories) == 0

    def test_idempotent_indexing(self, temp_traces, successful_trace):
        """Test unchanged files are not reparsed on subsequent indexing."""
        store = MemoryStore(trace_dirs=[temp_traces], index_path=temp_traces / ".index.json")

        # First index
        count1 = store.index_episodes()
        assert count1 == 1

        # Second index without changes
        count2 = store.index_episodes()
        assert count2 == 0  # No new files indexed
        assert len(store.memories) == 1  # Still have the memory

    def test_force_reindex(self, temp_traces, successful_trace):
        """Test force_reindex reparses all files."""
        store = MemoryStore(trace_dirs=[temp_traces], index_path=temp_traces / ".index.json")

        # First index
        store.index_episodes()

        # Force reindex
        count = store.index_episodes(force_reindex=True)
        assert count == 1  # Reindexed the file
        assert len(store.memories) == 1

    def test_retrieve_exact_task_match(self, temp_traces, successful_trace):
        """Test retrieval only returns episodes matching task name."""
        # Create trace with different task
        other_trace = [
            {
                "seq": 0,
                "ts": 6000.0,
                "event": "episode_start",
                "task": {
                    "name": "region_pick_place",  # Different task
                    "instruction": "Different task.",
                    "layout": {},
                    "goal": [],
                },
                "seed": 42,
                "llm": "OllamaLLM",
                "model": "qwen2.5:latest",
            },
            {
                "seq": 1,
                "ts": 6000.1,
                "event": "goal_check",
                "call_id": "call_0",
                "passed": True,
                "unmet": [],
            },
            {
                "seq": 2,
                "ts": 6000.2,
                "event": "episode_end",
                "result": {"success": True, "reason": "goal_met", "llm_turns": 1},
                "unmet": [],
            },
        ]

        other_path = temp_traces / "region_pick_place_xyz.jsonl"
        with other_path.open("w") as f:
            for event in other_trace:
                f.write(json.dumps(event) + "\n")

        index_path = temp_traces / ".test_index.json"
        store = MemoryStore(trace_dirs=[temp_traces], index_path=index_path)
        store.index_episodes()

        # Retrieve pick_place only
        results = store.retrieve(task_name="pick_place")
        assert len(results) == 1
        assert results[0].task_name == "pick_place"

        # Retrieve region_pick_place only
        results = store.retrieve(task_name="region_pick_place")
        assert len(results) == 1
        assert results[0].task_name == "region_pick_place"

    def test_retrieve_bounded_by_limit(self, temp_traces):
        """Test retrieval respects limit parameter."""
        # Create multiple traces for same task
        for i in range(5):
            trace = [
                {
                    "seq": 0,
                    "ts": 7000.0 + i,
                    "event": "episode_start",
                    "task": {
                        "name": "pick_place",
                        "instruction": "Put the red cube on the plate.",
                        "layout": {},
                        "goal": [],
                    },
                    "seed": 42,
                    "llm": "OllamaLLM",
                    "model": "qwen2.5:latest",
                },
                {
                    "seq": 1,
                    "ts": 7000.1 + i,
                    "event": "goal_check",
                    "call_id": "call_0",
                    "passed": True,
                    "unmet": [],
                },
                {
                    "seq": 2,
                    "ts": 7000.2 + i,
                    "event": "episode_end",
                    "result": {"success": True, "reason": "goal_met", "llm_turns": 1},
                    "unmet": [],
                },
            ]

            path = temp_traces / f"pick_place_{i}_abc.jsonl"
            with path.open("w") as f:
                for event in trace:
                    f.write(json.dumps(event) + "\n")

        index_path = temp_traces / ".test_index.json"
        store = MemoryStore(trace_dirs=[temp_traces], index_path=index_path)
        store.index_episodes()

        # Retrieve with limit=3
        results = store.retrieve(task_name="pick_place", limit=3)
        assert len(results) == 3

    def test_retrieve_sorted_by_recency(self, temp_traces):
        """Test retrieval returns most recent episodes first."""
        # Create traces with different timestamps
        timestamps = [8000.0, 8001.0, 8002.0]
        for i, ts in enumerate(timestamps):
            trace = [
                {
                    "seq": 0,
                    "ts": ts,
                    "event": "episode_start",
                    "task": {
                        "name": "pick_place",
                        "instruction": "Put the red cube on the plate.",
                        "layout": {},
                        "goal": [],
                    },
                    "seed": 42,
                    "llm": "OllamaLLM",
                    "model": "qwen2.5:latest",
                },
                {
                    "seq": 1,
                    "ts": ts + 0.1,
                    "event": "goal_check",
                    "call_id": "call_0",
                    "passed": True,
                    "unmet": [],
                },
                {
                    "seq": 2,
                    "ts": ts + 0.2,
                    "event": "episode_end",
                    "result": {"success": True, "reason": "goal_met", "llm_turns": 1},
                    "unmet": [],
                },
            ]

            path = temp_traces / f"pick_place_ts{i}_abc.jsonl"
            with path.open("w") as f:
                for event in trace:
                    f.write(json.dumps(event) + "\n")

        index_path = temp_traces / ".test_index.json"
        store = MemoryStore(trace_dirs=[temp_traces], index_path=index_path)
        store.index_episodes()

        results = store.retrieve(task_name="pick_place", limit=3)

        # Should be sorted newest first
        assert results[0].timestamp == 8002.0
        assert results[1].timestamp == 8001.0
        assert results[2].timestamp == 8000.0

    def test_retrieve_filtered_by_failure_codes(self, temp_traces, failed_trace_with_recovery):
        """Test retrieval can filter by failure codes."""
        index_path = temp_traces / ".test_index.json"
        store = MemoryStore(trace_dirs=[temp_traces], index_path=index_path)
        store.index_episodes()

        # Retrieve episodes with GRASP_SLIP
        results = store.retrieve(
            task_name="pick_place",
            failure_codes={"GRASP_SLIP"}
        )
        assert len(results) == 1
        assert "GRASP_SLIP" in results[0].failure_codes

        # Retrieve episodes with non-existent failure code
        results = store.retrieve(
            task_name="pick_place",
            failure_codes={"NONEXISTENT_CODE"}
        )
        assert len(results) == 0

    def test_retrieve_empty_for_nonmatching_task(self, temp_traces, successful_trace):
        """Test retrieval returns empty list for non-matching task."""
        index_path = temp_traces / ".test_index.json"
        store = MemoryStore(trace_dirs=[temp_traces], index_path=index_path)
        store.index_episodes()

        results = store.retrieve(task_name="nonexistent_task")
        assert len(results) == 0

    def test_persistence_save_and_load(self, temp_traces, successful_trace):
        """Test index persists to disk and loads correctly."""
        index_path = temp_traces / ".test_index.json"

        # Create and save index
        store1 = MemoryStore(trace_dirs=[temp_traces], index_path=index_path)
        store1.index_episodes()
        assert len(store1.memories) == 1

        # Load index in new instance
        store2 = MemoryStore(trace_dirs=[temp_traces], index_path=index_path)
        store2.index_episodes()  # Should load from disk
        assert len(store2.memories) == 1
        assert store2.memories[0].episode_id == store1.memories[0].episode_id

    def test_persistence_handles_corrupt_index(self, temp_traces, successful_trace):
        """Test graceful handling of corrupted index file."""
        index_path = temp_traces / ".corrupt_index.json"

        # Create corrupt index
        with index_path.open("w") as f:
            f.write("{invalid json")

        # Should start fresh, not crash
        store = MemoryStore(trace_dirs=[temp_traces], index_path=index_path)
        count = store.index_episodes()
        assert count == 1
        assert len(store.memories) == 1

    def test_exclude_patterns(self, temp_traces):
        """Test exclude_patterns filter works."""
        # Create subdirectory with test_ prefix
        test_dir = temp_traces / "test_traces"
        test_dir.mkdir()

        trace = [
            {
                "seq": 0,
                "ts": 9000.0,
                "event": "episode_start",
                "task": {"name": "pick_place", "instruction": "", "layout": {}, "goal": []},
                "seed": 42,
                "llm": "OllamaLLM",
                "model": "qwen2.5:latest",
            },
            {
                "seq": 1,
                "ts": 9000.1,
                "event": "goal_check",
                "call_id": "call_0",
                "passed": True,
                "unmet": [],
            },
            {
                "seq": 2,
                "ts": 9000.2,
                "event": "episode_end",
                "result": {"success": True, "reason": "goal_met", "llm_turns": 1},
                "unmet": [],
            },
        ]

        path = test_dir / "pick_place_test_xyz.jsonl"
        with path.open("w") as f:
            for event in trace:
                f.write(json.dumps(event) + "\n")

        # Index with exclude pattern
        store = MemoryStore(
            trace_dirs=[temp_traces, test_dir],
            exclude_patterns=["test_"]
        )
        count = store.index_episodes()

        # Should not index files from test_ directory
        assert count == 0

    def test_format_memory_excludes_coordinates(self, temp_traces, successful_trace):
        """Test formatted memory does not include object coordinates."""
        index_path = temp_traces / ".test_index.json"
        store = MemoryStore(trace_dirs=[temp_traces], index_path=index_path)
        store.index_episodes()

        memory = store.memories[0]
        formatted = store.format_memory(memory)

        # Should not contain coordinate patterns
        assert "x" not in formatted.lower() or "xyz" in formatted  # Allow in episode_id
        assert "0.1" not in formatted  # Specific coordinate value
        assert "-0.2" not in formatted  # Specific coordinate value

        # Should contain task info
        assert "pick_place" in formatted
        assert memory.episode_id in formatted

    def test_format_memory_includes_failure_codes(self, temp_traces, failed_trace_with_recovery):
        """Test formatted memory includes failure codes."""
        index_path = temp_traces / ".test_index.json"
        store = MemoryStore(trace_dirs=[temp_traces], index_path=index_path)
        store.index_episodes()

        memory = store.memories[0]
        formatted = store.format_memory(memory)

        assert "GRASP_SLIP" in formatted
        assert "OBJECT_NOT_AT_POSE" in formatted

    def test_format_memory_includes_recovery_actions(self, temp_traces, failed_trace_with_recovery):
        """Test formatted memory includes recovery actions."""
        index_path = temp_traces / ".test_index.json"
        store = MemoryStore(trace_dirs=[temp_traces], index_path=index_path)
        store.index_episodes()

        memory = store.memories[0]
        formatted = store.format_memory(memory)

        assert "Recovery actions" in formatted or "observe" in formatted

    def test_no_world_state_in_memory(self, temp_traces, successful_trace):
        """Test memory dataclass does not store world_state."""
        index_path = temp_traces / ".test_index.json"
        store = MemoryStore(trace_dirs=[temp_traces], index_path=index_path)
        store.index_episodes()

        memory = store.memories[0]
        memory_dict = memory.to_dict()

        # Verify no world_state anywhere in the memory
        assert "world_state" not in memory_dict
        assert "objects" not in memory_dict
        assert "gripper" not in memory_dict
        assert "pose" not in memory_dict

    def test_multiple_trace_directories(self, temp_traces):
        """Test indexing from multiple trace directories."""
        # Create two subdirectories
        dir1 = temp_traces / "dir1"
        dir2 = temp_traces / "dir2"
        dir1.mkdir()
        dir2.mkdir()

        # Create trace in each
        for i, trace_dir in enumerate([dir1, dir2]):
            trace = [
                {
                    "seq": 0,
                    "ts": 10000.0 + i,
                    "event": "episode_start",
                    "task": {"name": "pick_place", "instruction": "", "layout": {}, "goal": []},
                    "seed": 42,
                    "llm": "OllamaLLM",
                    "model": "qwen2.5:latest",
                },
                {
                    "seq": 1,
                    "ts": 10000.1 + i,
                    "event": "goal_check",
                    "call_id": "call_0",
                    "passed": True,
                    "unmet": [],
                },
                {
                    "seq": 2,
                    "ts": 10000.2 + i,
                    "event": "episode_end",
                    "result": {"success": True, "reason": "goal_met", "llm_turns": 1},
                    "unmet": [],
                },
            ]

            path = trace_dir / f"pick_place_{i}_abc.jsonl"
            with path.open("w") as f:
                for event in trace:
                    f.write(json.dumps(event) + "\n")

        # Index both directories
        index_path = temp_traces / ".test_index.json"
        store = MemoryStore(trace_dirs=[dir1, dir2], index_path=index_path)
        count = store.index_episodes()

        assert count == 2
        assert len(store.memories) == 2
