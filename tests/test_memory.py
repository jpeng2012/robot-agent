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

    def test_format_memory_excludes_llm_reasoning(self, temp_traces):
        """Test format_memory() does NOT include raw LLM reasoning (F009b safety)."""
        # Create trace with LLM reasoning containing sensitive content
        trace_file = temp_traces / "sample.jsonl"
        with trace_file.open("w") as f:
            json.dump({"event": "episode_start", "ts": 1234567890.0,
                      "task": {"name": "pick_place", "instruction": "Pick cube"},
                      "llm": "OllamaLLMClient", "model": "qwen3"}, f)
            f.write("\n")
            json.dump({"event": "llm_turn", "turn": 0,
                      "text": "I will use coordinates x=0.5, y=0.3 to pick the object"}, f)
            f.write("\n")
            json.dump({"event": "goal_check", "passed": True}, f)
            f.write("\n")
            json.dump({"event": "episode_end", "result": {"success": True, "reason": "goal_met"}}, f)
            f.write("\n")

        index_path = temp_traces / ".test_index.json"
        store = MemoryStore(trace_dirs=[temp_traces], index_path=index_path)
        store.index_episodes()

        memory = store.memories[0]
        formatted = store.format_memory(memory)

        # Assert raw LLM reasoning NOT in formatted output
        assert "coordinates x=0.5" not in formatted
        assert "I will use" not in formatted

        # Assert only structured facts present
        assert "Episode from" in formatted
        assert "Task: pick_place" in formatted
        assert "Outcome: SUCCESS" in formatted

    def test_recovery_actions_only_successful(self, temp_traces):
        """Test recovery actions only include successful actions, not rejected attempts."""
        # Create trace with failed pick → rejected place → successful pick
        trace_file = temp_traces / "recovery_test.jsonl"
        with trace_file.open("w") as f:
            json.dump({"event": "episode_start", "ts": 1234567890.0,
                      "task": {"name": "pick_place", "instruction": "Pick and place"},
                      "llm": "OllamaLLMClient", "model": "qwen3"}, f)
            f.write("\n")

            # First pick attempt - POSTCONDITION_FAILED
            json.dump({"event": "tool_call", "call_id": "call_1", "name": "pick",
                      "args": {"object_id": "cube_red", "x": 0.1, "y": -0.2}}, f)
            f.write("\n")
            json.dump({"event": "env_outcome", "call_id": "call_1",
                      "outcome": {"ok": True, "failure": None}}, f)
            f.write("\n")
            json.dump({"event": "verify", "call_id": "call_1", "ok": False,
                      "failure": {"code": "POSTCONDITION_FAILED", "detail": "not holding"}}, f)
            f.write("\n")

            # Place attempt - NOT_HOLDING (rejected by validation)
            json.dump({"event": "tool_call", "call_id": "call_2", "name": "place",
                      "args": {"x": -0.2, "y": 0.15}}, f)
            f.write("\n")
            json.dump({"event": "validation_error", "call_id": "call_2",
                      "failure": {"code": "NOT_HOLDING", "detail": "gripper empty"}}, f)
            f.write("\n")

            # Second pick attempt - SUCCESS
            json.dump({"event": "tool_call", "call_id": "call_3", "name": "pick",
                      "args": {"object_id": "cube_red", "x": 0.1, "y": -0.2}}, f)
            f.write("\n")
            json.dump({"event": "env_outcome", "call_id": "call_3",
                      "outcome": {"ok": True, "failure": None}}, f)
            f.write("\n")
            json.dump({"event": "verify", "call_id": "call_3", "ok": True, "failure": None}, f)
            f.write("\n")

            # Place - SUCCESS
            json.dump({"event": "tool_call", "call_id": "call_4", "name": "place",
                      "args": {"x": -0.2, "y": 0.15}}, f)
            f.write("\n")
            json.dump({"event": "env_outcome", "call_id": "call_4",
                      "outcome": {"ok": True, "failure": None}}, f)
            f.write("\n")
            json.dump({"event": "verify", "call_id": "call_4", "ok": True, "failure": None}, f)
            f.write("\n")

            json.dump({"event": "goal_check", "passed": True}, f)
            f.write("\n")
            json.dump({"event": "episode_end", "result": {"success": True, "reason": "goal_met"}}, f)
            f.write("\n")

        index_path = temp_traces / ".test_index.json"
        store = MemoryStore(trace_dirs=[temp_traces], index_path=index_path)
        store.index_episodes()

        memory = store.memories[0]

        # Should only have the successful pick (call_3), not the rejected place (call_2)
        assert len(memory.recovery_actions) == 1
        assert memory.recovery_actions[0]["name"] == "pick"
        assert memory.recovery_actions[0]["object_id"] == "cube_red"

        # Verify NOT_HOLDING is in failure_codes but place is not in recovery_actions
        assert "NOT_HOLDING" in memory.failure_codes
        assert "POSTCONDITION_FAILED" in memory.failure_codes

    def test_recovery_actions_reject_validation_errors(self, temp_traces):
        """Test that validation errors (rejected preconditions) are not recovery actions."""
        # Create trace with GRASP_SLIP → rejected place → successful pick
        trace_file = temp_traces / "validation_error_test.jsonl"
        with trace_file.open("w") as f:
            json.dump({"event": "episode_start", "ts": 1234567890.0,
                      "task": {"name": "pick_place", "instruction": "Pick and place"},
                      "llm": "OllamaLLMClient", "model": "qwen3"}, f)
            f.write("\n")

            # Pick attempt - GRASP_SLIP
            json.dump({"event": "tool_call", "call_id": "call_1", "name": "pick",
                      "args": {"object_id": "cube_red", "x": 0.1, "y": -0.2}}, f)
            f.write("\n")
            json.dump({"event": "env_outcome", "call_id": "call_1",
                      "outcome": {"ok": False, "failure": {"code": "GRASP_SLIP", "detail": "slipped"}}}, f)
            f.write("\n")

            # Place attempt immediately after - validation_error NOT_HOLDING
            json.dump({"event": "tool_call", "call_id": "call_2", "name": "place",
                      "args": {"x": -0.2, "y": 0.15}}, f)
            f.write("\n")
            json.dump({"event": "validation_error", "call_id": "call_2",
                      "failure": {"code": "NOT_HOLDING", "detail": "gripper empty"}}, f)
            f.write("\n")

            # Second pick - SUCCESS
            json.dump({"event": "tool_call", "call_id": "call_3", "name": "pick",
                      "args": {"object_id": "cube_red", "x": 0.2, "y": -0.2}}, f)
            f.write("\n")
            json.dump({"event": "env_outcome", "call_id": "call_3",
                      "outcome": {"ok": True, "failure": None}}, f)
            f.write("\n")
            json.dump({"event": "verify", "call_id": "call_3", "ok": True, "failure": None}, f)
            f.write("\n")

            json.dump({"event": "goal_check", "passed": True}, f)
            f.write("\n")
            json.dump({"event": "episode_end", "result": {"success": True, "reason": "goal_met"}}, f)
            f.write("\n")

        index_path = temp_traces / ".test_index.json"
        store = MemoryStore(trace_dirs=[temp_traces], index_path=index_path)
        store.index_episodes()

        memory = store.memories[0]

        # Should ONLY have the successful pick (call_3)
        # The rejected place (call_2) should NOT be in recovery_actions
        assert len(memory.recovery_actions) == 1
        assert memory.recovery_actions[0]["name"] == "pick"

        # Both failure codes should be present
        assert "GRASP_SLIP" in memory.failure_codes
        assert "NOT_HOLDING" in memory.failure_codes

    def test_recovery_actions_mixed_sequence(self, temp_traces):
        """Test complex sequence with multiple failures and mixed success/failure recoveries."""
        trace_file = temp_traces / "mixed_sequence.jsonl"
        with trace_file.open("w") as f:
            json.dump({"event": "episode_start", "ts": 1234567890.0,
                      "task": {"name": "pick_place", "instruction": "Pick and place"},
                      "llm": "OllamaLLMClient", "model": "qwen3"}, f)
            f.write("\n")

            # Failure 1: GRASP_SLIP
            json.dump({"event": "tool_call", "call_id": "c1", "name": "pick",
                      "args": {"object_id": "cube_red", "x": 0.1, "y": -0.2}}, f)
            f.write("\n")
            json.dump({"event": "env_outcome", "call_id": "c1",
                      "outcome": {"ok": False, "failure": {"code": "GRASP_SLIP"}}}, f)
            f.write("\n")

            # Recovery attempt 1: observe (successful)
            json.dump({"event": "tool_call", "call_id": "c2", "name": "observe", "args": {}}, f)
            f.write("\n")
            json.dump({"event": "env_outcome", "call_id": "c2",
                      "outcome": {"ok": True, "failure": None}}, f)
            f.write("\n")

            # Recovery attempt 2: pick again (POSTCONDITION_FAILED)
            json.dump({"event": "tool_call", "call_id": "c3", "name": "pick",
                      "args": {"object_id": "cube_red", "x": 0.2, "y": -0.2}}, f)
            f.write("\n")
            json.dump({"event": "env_outcome", "call_id": "c3",
                      "outcome": {"ok": True, "failure": None}}, f)
            f.write("\n")
            json.dump({"event": "verify", "call_id": "c3", "ok": False,
                      "failure": {"code": "POSTCONDITION_FAILED"}}, f)
            f.write("\n")

            # Failed recovery attempt: place (NOT_HOLDING)
            json.dump({"event": "tool_call", "call_id": "c4", "name": "place",
                      "args": {"x": -0.2, "y": 0.15}}, f)
            f.write("\n")
            json.dump({"event": "validation_error", "call_id": "c4",
                      "failure": {"code": "NOT_HOLDING"}}, f)
            f.write("\n")

            # Final successful recovery: pick
            json.dump({"event": "tool_call", "call_id": "c5", "name": "pick",
                      "args": {"object_id": "cube_red", "x": 0.2, "y": -0.2}}, f)
            f.write("\n")
            json.dump({"event": "env_outcome", "call_id": "c5",
                      "outcome": {"ok": True, "failure": None}}, f)
            f.write("\n")
            json.dump({"event": "verify", "call_id": "c5", "ok": True, "failure": None}, f)
            f.write("\n")

            json.dump({"event": "goal_check", "passed": True}, f)
            f.write("\n")
            json.dump({"event": "episode_end", "result": {"success": True, "reason": "goal_met"}}, f)
            f.write("\n")

        index_path = temp_traces / ".test_index.json"
        store = MemoryStore(trace_dirs=[temp_traces], index_path=index_path)
        store.index_episodes()

        memory = store.memories[0]

        # Should have: observe (c2), pick (c5)
        # Should NOT have: pick (c3, failed postcondition), place (c4, validation error)
        assert len(memory.recovery_actions) == 2
        assert memory.recovery_actions[0]["name"] == "observe"
        assert memory.recovery_actions[1]["name"] == "pick"
        assert memory.recovery_actions[1]["object_id"] == "cube_red"

        # All failure codes should be present
        assert "GRASP_SLIP" in memory.failure_codes
        assert "POSTCONDITION_FAILED" in memory.failure_codes
        assert "NOT_HOLDING" in memory.failure_codes


# --- Recovery-action correlation in the real loop event format (every call ends with tool_result) ---

def _write_episode(trace_dir, name, body_events, success=True):
    events = [{"event": "episode_start", "ts": 1234567890.0,
               "task": {"name": "pick_place", "instruction": "Pick and place"},
               "llm": "OllamaLLMClient", "model": "qwen3"}]
    events += body_events
    events += [{"event": "goal_check", "passed": success},
               {"event": "episode_end",
                "result": {"success": success, "reason": "goal_met" if success else "max_steps"}}]
    path = trace_dir / f"{name}.jsonl"
    path.write_text("\n".join(json.dumps(e) for e in events) + "\n")
    store = MemoryStore(trace_dirs=[trace_dir], index_path=trace_dir / ".idx.json")
    store.index_episodes()
    return next(m for m in store.memories if m.episode_id == name)


def _call(cid, name, **args):
    return {"event": "tool_call", "call_id": cid, "name": name, "args": args}


def _skill(cid, ok, verify_ok=True, code=None, attempt=0):
    """env_outcome + verify, as the loop logs them for an executed skill attempt."""
    fail = {"code": code, "detail": "x"} if code else None
    evs = [{"event": "env_outcome", "call_id": cid, "attempt": attempt,
            "outcome": {"ok": ok, "failure": None if ok else fail}}]
    evs.append({"event": "verify", "call_id": cid, "attempt": attempt,
                "ok": ok and verify_ok, "failure": None if (ok and verify_ok) else fail})
    return evs


def _result(cid, ok, code=None):
    return {"event": "tool_result", "call_id": cid, "ok": ok,
            "error": {"code": code, "detail": "x"} if code else None}


def test_recovery_observe_known_only_from_tool_result(temp_traces):
    """observe has no env_outcome/verify; its tool_result alone marks it successful."""
    m = _write_episode(temp_traces, "obs_only", [
        _call("c1", "pick", object_id="cube_red", x=0.1, y=-0.2),
        *_skill("c1", False, code="GRASP_SLIP"), _result("c1", False, "GRASP_SLIP"),
        _call("c2", "observe"), _result("c2", True),
    ], success=False)
    assert m.recovery_actions == [{"name": "observe"}]


def test_retry_succeeding_under_same_call_id_is_not_a_failure_or_recovery(temp_traces):
    """Loop-internal retry: attempt 0 fails, attempt 1 succeeds, same call_id; final outcome is success."""
    m = _write_episode(temp_traces, "retry_ok", [
        _call("c1", "pick", object_id="cube_red", x=0.1, y=-0.2),
        *_skill("c1", False, code="GRASP_SLIP", attempt=0),
        {"event": "retry", "call_id": "c1", "attempt": 1, "reason": "GRASP_SLIP"},
        *_skill("c1", True, attempt=1), _result("c1", True),
        _call("c2", "place", x=-0.2, y=0.15), *_skill("c2", True), _result("c2", True),
    ])
    assert m.recovery_actions == []
    assert "GRASP_SLIP" in m.failure_codes  # the fault is still reported


def test_retry_exhausted_then_new_successful_pick_is_recovery(temp_traces):
    m = _write_episode(temp_traces, "retry_fail", [
        _call("c1", "pick", object_id="cube_red", x=0.1, y=-0.2),
        *_skill("c1", False, code="GRASP_SLIP", attempt=0),
        {"event": "retry", "call_id": "c1", "attempt": 1, "reason": "GRASP_SLIP"},
        *_skill("c1", False, code="GRASP_SLIP", attempt=1), _result("c1", False, "GRASP_SLIP"),
        _call("c2", "pick", object_id="cube_red", x=0.2, y=-0.2), *_skill("c2", True), _result("c2", True),
    ])
    assert m.recovery_actions == [{"name": "pick", "object_id": "cube_red"}]


def test_skipped_call_is_not_a_recovery(temp_traces):
    """A call skipped because an earlier call in the same turn failed is a failure, not a recovery."""
    m = _write_episode(temp_traces, "skipped", [
        _call("c1", "pick", object_id="cube_red", x=0.1, y=-0.2),
        *_skill("c1", False, code="GRASP_SLIP"), _result("c1", False, "GRASP_SLIP"),
        {"event": "tool_call", "call_id": "c2", "name": "place", "args": {"x": -0.2, "y": 0.15}, "skipped": True},
        _result("c2", False, "SKIPPED"),
        _call("c3", "pick", object_id="cube_red", x=0.2, y=-0.2), *_skill("c3", True), _result("c3", True),
    ])
    assert m.recovery_actions == [{"name": "pick", "object_id": "cube_red"}]


def test_tool_call_without_call_id_is_never_recorded(temp_traces):
    m = _write_episode(temp_traces, "no_id", [
        _call("c1", "pick", object_id="cube_red", x=0.1, y=-0.2),
        *_skill("c1", False, code="GRASP_SLIP"), _result("c1", False, "GRASP_SLIP"),
        {"event": "tool_call", "name": "observe", "args": {}},
        {"event": "tool_result", "ok": True, "error": None},
    ], success=False)
    assert m.recovery_actions == []


def test_failures_followed_only_by_failures_record_nothing(temp_traces):
    m = _write_episode(temp_traces, "all_fail", [
        _call("c1", "pick", object_id="cube_red", x=0.1, y=-0.2),
        *_skill("c1", True, verify_ok=False, code="POSTCONDITION_FAILED"),
        _result("c1", False, "POSTCONDITION_FAILED"),
        _call("c2", "place", x=-0.2, y=0.15),
        {"event": "validation_error", "call_id": "c2", "failure": {"code": "NOT_HOLDING", "detail": "x"}},
        _result("c2", False, "NOT_HOLDING"),
    ], success=False)
    assert m.recovery_actions == []
    assert {"POSTCONDITION_FAILED", "NOT_HOLDING"} <= set(m.failure_codes)


# --- Hardening: index migration, duplicate call IDs, malformed args, failed finish, end-to-end ---

from robot_agent.memory import INDEX_VERSION  # noqa: E402


def _slip_then_pick(prefix="c"):
    return [
        _call(f"{prefix}1", "pick", object_id="cube_red", x=0.1, y=-0.2),
        *_skill(f"{prefix}1", False, code="GRASP_SLIP"), _result(f"{prefix}1", False, "GRASP_SLIP"),
        _call(f"{prefix}2", "place", x=-0.2, y=0.15),
        {"event": "validation_error", "call_id": f"{prefix}2", "failure": {"code": "NOT_HOLDING", "detail": "x"}},
        _result(f"{prefix}2", False, "NOT_HOLDING"),
        _call(f"{prefix}3", "pick", object_id="cube_red", x=0.2, y=-0.2),
        *_skill(f"{prefix}3", True), _result(f"{prefix}3", True),
    ]


def test_index_from_previous_parser_version_is_rebuilt_even_if_mtimes_match(temp_traces):
    m = _write_episode(temp_traces, "migrate", _slip_then_pick())
    trace = temp_traces / "migrate.jsonl"
    idx = temp_traces / ".idx.json"
    # Simulate an index written by the version-1 parser: stale recovery list, matching mtime.
    stale = m.to_dict()
    stale["recovery_actions"] = [{"name": "place"}, {"name": "pick", "object_id": "cube_red"}]
    idx.write_text(json.dumps({"version": 1, "indexed_at": 0.0, "memories": [stale],
                               "file_mtimes": {str(trace): trace.stat().st_mtime}}))

    store = MemoryStore(trace_dirs=[temp_traces], index_path=idx)
    assert store.index_episodes() == 1  # reparsed despite unchanged mtime
    assert store.memories[0].recovery_actions == [{"name": "pick", "object_id": "cube_red"}]
    assert json.loads(idx.read_text())["version"] == INDEX_VERSION

    # A current-version index is reused without reparsing.
    again = MemoryStore(trace_dirs=[temp_traces], index_path=idx)
    assert again.index_episodes() == 0
    assert again.memories[0].recovery_actions == [{"name": "pick", "object_id": "cube_red"}]


def test_index_without_version_field_is_rebuilt(temp_traces):
    _write_episode(temp_traces, "nover", _slip_then_pick())
    trace = temp_traces / "nover.jsonl"
    idx = temp_traces / ".idx.json"
    idx.write_text(json.dumps({"memories": [], "file_mtimes": {str(trace): trace.stat().st_mtime}}))
    store = MemoryStore(trace_dirs=[temp_traces], index_path=idx)
    assert store.index_episodes() == 1


def test_duplicate_call_id_is_never_labelled_a_recovery(temp_traces):
    """Failed and successful picks share id "x": their events cannot be told apart."""
    m = _write_episode(temp_traces, "dup", [
        _call("a", "pick", object_id="cube_red", x=0.1, y=-0.2),
        *_skill("a", False, code="GRASP_SLIP"), _result("a", False, "GRASP_SLIP"),
        _call("x", "pick", object_id="cube_red", x=0.3, y=-0.2),
        *_skill("x", False, code="GRASP_SLIP"), _result("x", False, "GRASP_SLIP"),
        _call("x", "pick", object_id="cube_red", x=0.2, y=-0.2),
        *_skill("x", True), _result("x", True),
    ])
    assert m.recovery_actions == []
    assert "GRASP_SLIP" in m.failure_codes


def test_duplicate_call_id_does_not_block_later_verified_recovery(temp_traces):
    m = _write_episode(temp_traces, "dup_then", [
        _call("x", "observe"), _result("x", True),
        _call("x", "observe"), _result("x", True),
        _call("b", "pick", object_id="cube_red", x=0.5, y=0.5),
        {"event": "validation_error", "call_id": "b", "failure": {"code": "OBJECT_NOT_AT_POSE", "detail": "x"}},
        _result("b", False, "OBJECT_NOT_AT_POSE"),
        _call("c", "pick", object_id="cube_red", x=0.1, y=-0.2), *_skill("c", True), _result("c", True),
    ])
    assert m.recovery_actions == [{"name": "pick", "object_id": "cube_red"}]


@pytest.mark.parametrize("args", ['{"object_id": "cube_red", x', None, ["object_id"], {"object_id": 7}])
def test_non_dict_or_malformed_args_do_not_raise(temp_traces, args):
    m = _write_episode(temp_traces, "badargs", [
        _call("c1", "pick", object_id="cube_red", x=0.1, y=-0.2),
        *_skill("c1", False, code="GRASP_SLIP"), _result("c1", False, "GRASP_SLIP"),
        {"event": "tool_call", "call_id": "c2", "name": "pick", "args": args},
        *_skill("c2", True), _result("c2", True),
    ])
    assert m.recovery_actions == [{"name": "pick"}]  # recorded without an object_id


def test_failed_finish_is_a_failure_and_a_later_success_is_the_recovery(temp_traces):
    # Extraction only (parsed directly). Eligibility of this trace shape is covered by
    # test_early_failed_finish_then_verified_completion_is_eligible.
    events = [
        {"event": "episode_start", "ts": 1.0, "task": {"name": "pick_place"}, "llm": "OllamaLLMClient"},
        _call("c1", "finish", summary="done"),
        {"event": "goal_check", "call_id": "c1", "passed": False, "unmet": ["On(cube_red, plate)"]},
        _result("c1", False, "GOAL_NOT_MET"),
        _call("c2", "pick", object_id="cube_red", x=0.1, y=-0.2), *_skill("c2", True), _result("c2", True),
        _call("c3", "place", x=-0.2, y=0.15), *_skill("c3", True), _result("c3", True),
        _call("c4", "finish", summary="done"),
        {"event": "goal_check", "call_id": "c4", "passed": True, "unmet": []},
        _result("c4", True),
        {"event": "episode_end", "result": {"success": True, "reason": "goal_met"}},
    ]
    trace = temp_traces / "finish_fail.jsonl"
    trace.write_text("\n".join(json.dumps(e) for e in events) + "\n")
    m = MemoryStore(trace_dirs=[temp_traces], index_path=temp_traces / ".idx.json")._parse_trace(trace)
    assert m.recovery_actions == [{"name": "pick", "object_id": "cube_red"}]


def test_successful_finish_right_after_failure_is_recorded_as_finish(temp_traces):
    """Documents existing semantics: any first independently successful call fills the slot."""
    m = _write_episode(temp_traces, "finish_ok", [
        _call("c1", "observe"), _result("c1", True),
        _call("c2", "place", x=-0.2, y=0.15),
        {"event": "validation_error", "call_id": "c2", "failure": {"code": "NOT_HOLDING", "detail": "x"}},
        _result("c2", False, "NOT_HOLDING"),
        _call("c3", "finish", summary="done"),
        {"event": "goal_check", "call_id": "c3", "passed": True, "unmet": []},
        _result("c3", True),
    ])
    assert m.recovery_actions == [{"name": "finish"}]


def test_end_to_end_agent_loop_trace_records_only_verified_recovery(tmp_path):
    """Real AgentLoop + MockTabletopEnv: rejected pick, rejected place, then a verified pick."""
    from robot_agent.env import MockTabletopEnv
    from robot_agent.goals import pick_place_task
    from robot_agent.llm import ScriptedLLM, ToolCall
    from robot_agent.loop import AgentLoop

    def tc(i, name, **args):
        return [ToolCall(id=f"call_{i}", name=name, args=args)]

    llm = ScriptedLLM([
        tc(0, "pick", object_id="cube_red", x=0.5, y=0.5),      # wrong pose -> validation_error
        tc(1, "place", x=-0.2, y=0.15),                         # empty gripper -> NOT_HOLDING
        tc(2, "observe"),
        tc(3, "pick", object_id="cube_red", x=0.1, y=-0.2),
        tc(4, "place", x=-0.2, y=0.15),
        tc(5, "finish", summary="done"),
    ])
    result = AgentLoop(MockTabletopEnv(), llm, trace_dir=tmp_path / "e2e").run(pick_place_task(), seed=0)
    assert result.success

    store = MemoryStore(trace_dirs=[tmp_path / "e2e"], index_path=tmp_path / "idx.json")
    # ScriptedLLM episodes are ineligible for the index by design, so parse the trace directly.
    m = store._parse_trace(Path(result.trace_path))
    assert m is not None
    assert m.recovery_actions == [{"name": "observe"}]  # first independently successful call
    assert {"OBJECT_NOT_AT_POSE", "NOT_HOLDING"} <= set(m.failure_codes)
    text = store.format_memory(m)
    assert "place()" not in text and "0.5" not in text and "0.1" not in text


# --- Success eligibility: terminal episode_end must agree with the FINAL goal_check ---

def _index_events(trace_dir, name, body):
    events = [{"event": "episode_start", "ts": 1.0, "task": {"name": "pick_place"},
               "llm": "OllamaLLMClient", "model": "qwen3"}] + body
    (trace_dir / f"{name}.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\n")
    store = MemoryStore(trace_dirs=[trace_dir], index_path=trace_dir / ".idx.json")
    store.index_episodes()
    return [m.episode_id for m in store.memories]


def _finish(cid, passed):
    return [_call(cid, "finish", summary="done"),
            {"event": "goal_check", "call_id": cid, "passed": passed,
             "unmet": [] if passed else ["On(cube_red, plate)"]},
            _result(cid, passed, None if passed else "GOAL_NOT_MET")]


def test_early_failed_finish_then_verified_completion_is_eligible(temp_traces):
    ids = _index_events(temp_traces, "late_success", [
        *_finish("c1", False),
        _call("c2", "pick", object_id="cube_red", x=0.1, y=-0.2), *_skill("c2", True), _result("c2", True),
        _call("c3", "place", x=-0.2, y=0.15), *_skill("c3", True), _result("c3", True),
        *_finish("c4", True),
        {"event": "episode_end", "result": {"success": True, "reason": "goal_met"}},
    ])
    assert ids == ["late_success"]


def test_earlier_passing_goal_check_does_not_rescue_a_failed_final_check(temp_traces):
    ids = _index_events(temp_traces, "late_fail", [
        *_finish("c1", True),
        *_finish("c2", False),
        {"event": "episode_end", "result": {"success": True, "reason": "goal_met"}},  # inconsistent
    ])
    assert ids == []


@pytest.mark.parametrize("result", [
    {"success": True, "reason": "max_steps"},
    {"success": True},
    {"success": "true", "reason": "goal_met"},
])
def test_success_requires_episode_end_goal_met(temp_traces, result):
    ids = _index_events(temp_traces, "bad_end", [*_finish("c1", True), {"event": "episode_end", "result": result}])
    assert ids == []
