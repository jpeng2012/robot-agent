"""
Persistent episodic memory for robot-agent.

This module provides storage and retrieval of verified episodic experiences
from JSONL trace files. Memories exclude object coordinates and world state
to prevent stale-state risks.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EpisodeMemory:
    """Coordinate-free summary of one verified episode."""

    episode_id: str
    task_name: str
    timestamp: float
    success: bool
    reason: str
    failure_codes: list[str] = field(default_factory=list)
    llm_reasoning: list[str] = field(default_factory=list)
    recovery_actions: list[dict[str, Any]] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
    trace_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for JSON storage."""
        return {
            "episode_id": self.episode_id,
            "task_name": self.task_name,
            "timestamp": self.timestamp,
            "success": self.success,
            "reason": self.reason,
            "failure_codes": self.failure_codes,
            "llm_reasoning": self.llm_reasoning,
            "recovery_actions": self.recovery_actions,
            "provenance": self.provenance,
            "trace_path": self.trace_path,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EpisodeMemory:
        """Deserialize from dict."""
        return cls(
            episode_id=data["episode_id"],
            task_name=data["task_name"],
            timestamp=data["timestamp"],
            success=data["success"],
            reason=data["reason"],
            failure_codes=data.get("failure_codes", []),
            llm_reasoning=data.get("llm_reasoning", []),
            recovery_actions=data.get("recovery_actions", []),
            provenance=data.get("provenance", {}),
            trace_path=data.get("trace_path", ""),
        )


class MemoryStore:
    """
    Persistent storage and retrieval of episodic memories.

    Stores coordinate-free episode summaries extracted from JSONL traces.
    Filters out test traces, incomplete episodes, and unverified successes.
    """

    def __init__(
        self,
        trace_dirs: list[Path | str],
        index_path: Path | str = ".memory_index.json",
        exclude_patterns: list[str] | None = None,
    ):
        """
        Initialize memory store.

        Args:
            trace_dirs: Directories containing JSONL trace files to index
            index_path: Path to persistent index file (for idempotent indexing)
            exclude_patterns: Directory name patterns to exclude (e.g., ["test_"])
        """
        self.trace_dirs = [Path(d) for d in trace_dirs]
        self.index_path = Path(index_path)
        self.exclude_patterns = exclude_patterns or []
        self.memories: list[EpisodeMemory] = []
        self._file_mtimes: dict[str, float] = {}

    def index_episodes(self, force_reindex: bool = False) -> int:
        """
        Build or update the episode index from trace files.

        Idempotent: only reparses files that changed since last index.

        Args:
            force_reindex: If True, reparse all files ignoring mtimes

        Returns:
            Number of episodes indexed
        """
        # Load existing index if available
        if not force_reindex and self.index_path.exists():
            self._load_index()

        indexed_count = 0
        for trace_dir in self.trace_dirs:
            if not trace_dir.exists():
                continue

            # Skip excluded directories
            if any(pattern in trace_dir.name for pattern in self.exclude_patterns):
                continue

            for trace_file in trace_dir.glob("*.jsonl"):
                # Check if file changed since last index
                mtime = trace_file.stat().st_mtime
                trace_key = str(trace_file)

                if not force_reindex and trace_key in self._file_mtimes:
                    if self._file_mtimes[trace_key] == mtime:
                        continue  # File unchanged, skip

                # Parse trace and extract memory
                memory = self._parse_trace(trace_file)
                if memory and self._is_eligible(memory):
                    # Remove old version if exists
                    self.memories = [m for m in self.memories if m.episode_id != memory.episode_id]
                    self.memories.append(memory)
                    self._file_mtimes[trace_key] = mtime
                    indexed_count += 1

        # Save updated index
        self._save_index()
        return indexed_count

    def retrieve(
        self,
        task_name: str,
        limit: int = 3,
        failure_codes: set[str] | None = None,
    ) -> list[EpisodeMemory]:
        """
        Retrieve relevant episodic memories for a task.

        Args:
            task_name: Exact task name to match (e.g., "pick_place")
            limit: Maximum number of memories to return
            failure_codes: Optional filter for episodes with these failure codes

        Returns:
            List of relevant memories, sorted by relevance and recency
        """
        # Exact task name match
        candidates = [m for m in self.memories if m.task_name == task_name]

        # Filter by failure codes if specified
        if failure_codes:
            candidates = [
                m for m in candidates
                if any(fc in m.failure_codes for fc in failure_codes)
            ]

        # Sort by: 1) failure overlap (descending), 2) recency (descending)
        if failure_codes:
            def score(m: EpisodeMemory) -> tuple[int, float]:
                overlap = len(set(m.failure_codes) & failure_codes)
                return (overlap, m.timestamp)
            candidates.sort(key=score, reverse=True)
        else:
            # Just sort by recency
            candidates.sort(key=lambda m: m.timestamp, reverse=True)

        return candidates[:limit]

    def _parse_trace(self, trace_path: Path) -> EpisodeMemory | None:
        """
        Extract coordinate-free memory from a trace file.

        Returns None if trace is malformed or incomplete.
        """
        try:
            events = []
            with trace_path.open("r") as f:
                for line in f:
                    if not line.strip():
                        continue
                    events.append(json.loads(line))

            if not events:
                return None

            # Extract episode metadata
            episode_start = next((e for e in events if e["event"] == "episode_start"), None)
            episode_end = next((e for e in events if e["event"] == "episode_end"), None)

            if not episode_start or not episode_end:
                return None  # Incomplete episode

            task = episode_start.get("task", {})
            result = episode_end.get("result", {})

            # Extract episode ID from filename
            episode_id = trace_path.stem

            # Collect failure codes
            failure_codes = []
            for event in events:
                # Check env_outcome failures
                if event["event"] == "env_outcome":
                    outcome = event.get("outcome", {})
                    if not outcome.get("ok") and outcome.get("failure"):
                        code = outcome["failure"].get("code")
                        if code and code not in failure_codes:
                            failure_codes.append(code)

                # Check verification failures
                if event["event"] == "verify":
                    if not event.get("ok") and event.get("failure"):
                        code = event["failure"].get("code")
                        if code and code not in failure_codes:
                            failure_codes.append(code)

                # Check validation errors
                if event["event"] == "validation_error":
                    failure = event.get("failure", {})
                    code = failure.get("code")
                    if code and code not in failure_codes:
                        failure_codes.append(code)

            # Collect LLM reasoning (coordinate-free text)
            llm_reasoning = []
            for event in events:
                if event["event"] == "llm_turn":
                    text = event.get("text", "").strip()
                    if text and text not in llm_reasoning:
                        llm_reasoning.append(text)

            # Collect recovery actions (tool calls after failures)
            recovery_actions = []
            prev_failed = False
            for i, event in enumerate(events):
                # Track if previous action failed
                if event["event"] in ("env_outcome", "verify", "validation_error"):
                    if event["event"] == "env_outcome":
                        prev_failed = not event.get("outcome", {}).get("ok", True)
                    elif event["event"] == "verify":
                        prev_failed = not event.get("ok", True)
                    elif event["event"] == "validation_error":
                        prev_failed = True

                # If this is a tool call after failure, it's a recovery attempt
                if event["event"] == "tool_call" and prev_failed:
                    # Coordinate-free: only store action name and object_id (no args, no schema)
                    tool_call = {"name": event.get("name")}

                    # Preserve object_id if present (safe, no coordinates)
                    if "object_id" in event.get("args", {}):
                        tool_call["object_id"] = event["args"]["object_id"]

                    recovery_actions.append(tool_call)
                    prev_failed = False  # Reset after recording recovery

            # Build provenance
            provenance = {
                "llm": episode_start.get("llm"),
                "model": episode_start.get("model"),
                "trace_dir": str(trace_path.parent.name),
            }

            return EpisodeMemory(
                episode_id=episode_id,
                task_name=task.get("name", "unknown"),
                timestamp=episode_start.get("ts", 0.0),
                success=result.get("success", False),
                reason=result.get("reason", "unknown"),
                failure_codes=failure_codes,
                llm_reasoning=llm_reasoning,
                recovery_actions=recovery_actions,
                provenance=provenance,
                trace_path=str(trace_path),
            )

        except (json.JSONDecodeError, KeyError, ValueError):
            # Malformed trace, skip
            return None

    def _is_eligible(self, memory: EpisodeMemory) -> bool:
        """
        Filter out test traces and unverified episodes.

        Excludes:
        - ScriptedLLM episodes (test policy)
        - Episodes from test directories
        - Successful episodes without verified goal achievement
        """
        # Exclude ScriptedLLM (test policy)
        if memory.provenance.get("llm") == "ScriptedLLM":
            return False

        # Exclude test traces (check episode_id to avoid matching parent directories)
        if any(pattern in memory.episode_id for pattern in self.exclude_patterns):
            return False

        # For successful episodes, verify goal was actually achieved
        # This requires checking the goal_check event in the original trace
        if memory.success:
            try:
                with open(memory.trace_path, "r") as f:
                    events = [json.loads(line) for line in f if line.strip()]

                # Find goal_check event
                goal_check = next(
                    (e for e in events if e["event"] == "goal_check"),
                    None
                )

                # If no goal_check or goal not passed, exclude
                if not goal_check or not goal_check.get("passed", False):
                    return False

            except (FileNotFoundError, json.JSONDecodeError, KeyError):
                # Can't verify, exclude for safety
                return False

        return True

    def _save_index(self) -> None:
        """Persist index to disk."""
        index_data = {
            "version": 1,
            "indexed_at": time.time(),
            "memories": [m.to_dict() for m in self.memories],
            "file_mtimes": self._file_mtimes,
        }

        with self.index_path.open("w") as f:
            json.dump(index_data, f, indent=2)

    def _load_index(self) -> None:
        """Load index from disk."""
        try:
            with self.index_path.open("r") as f:
                index_data = json.load(f)

            self.memories = [
                EpisodeMemory.from_dict(m) for m in index_data.get("memories", [])
            ]
            self._file_mtimes = index_data.get("file_mtimes", {})

        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            # Index doesn't exist or is corrupted, start fresh
            self.memories = []
            self._file_mtimes = {}

    def format_memory(self, memory: EpisodeMemory) -> str:
        """
        Format memory as coordinate-free narrative for LLM context.

        Excludes raw LLM reasoning, coordinates, world states, and file paths.
        Emits only validated structured facts.
        """
        # Compute time ago
        time_ago = time.time() - memory.timestamp
        if time_ago < 3600:
            time_str = f"{int(time_ago / 60)} minutes ago"
        elif time_ago < 86400:
            time_str = f"{int(time_ago / 3600)} hours ago"
        else:
            time_str = f"{int(time_ago / 86400)} days ago"

        lines = [f"Episode from {time_str} ({memory.episode_id}):"]
        lines.append(f"  Task: {memory.task_name}")
        lines.append(f"  Outcome: {'SUCCESS' if memory.success else 'FAILED'} ({memory.reason})")

        if memory.failure_codes:
            lines.append(f"  Failures encountered: {', '.join(memory.failure_codes)}")

        if memory.recovery_actions:
            lines.append("  Recovery actions:")
            for action in memory.recovery_actions:
                if "object_id" in action:
                    lines.append(f"    - {action['name']}({action['object_id']})")
                else:
                    lines.append(f"    - {action['name']}()")

        return "\n".join(lines)
