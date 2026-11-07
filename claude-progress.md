# Robot-Agent Development Progress

## Session 2026-11-07: F006 Verification Gap Tests

### Changes
- Added test_pick_postcondition_catches_false_success to test_loop.py
- Added test_inconsistent_state_holding_without_contact to test_loop.py (characterization test)
- Both tests limited to test files only, no production code changes

### Verification
```bash
# New tests pass
pytest tests/test_loop.py::test_pick_postcondition_catches_false_success -v
pytest tests/test_loop.py::test_inconsistent_state_holding_without_contact -v
# Result: 2/2 passed

# Full deterministic suite
pytest tests/ -v -k 'not requires_server and not requires_robosuite'
# Result: 41 passed, 5 skipped in 0.16s (was 39 passed before)

# F006 referenced test still passes
pytest tests/test_loop.py::test_failed_pick_recovers_by_bounded_retry -v
# Result: PASSED
```

### Design Decisions
- Test 1 verifies postcondition catches false-positive grasp reports (env reports ok=True, gripper.holding=None)
- Test 2 characterizes spatial consistency limitation: verifier accepts gripper.holding="cube_red" even when object pose is 0.5m away
- Used test subclasses (FalseSuccessEnv, InconsistentStateEnv) to avoid modifying production code
- Both tests follow existing patterns (SlipAndNudgeEnv, test_silent_failure_cannot_produce_false_success)

### Observed Limitations
- **Postcondition validation works correctly**: Test 1 confirms _pick_post catches env false-positive reports
- **Spatial consistency not verified**: Test 2 reveals postcondition only checks gripper.holding field, NOT spatial relationship between gripper and object poses
- **Goal check is safety net**: When postcondition accepts inconsistent spatial state, goal verification still prevents false task completion
- **F006 acceptance criterion #3 now satisfied**: "Fault injection tests demonstrate postcondition catches execution failures"

### Remaining Work
- F006 not marked complete pending review of test evidence
- Spatial consistency limitation documented but not addressed (out of scope for F006)

### Test Evidence
Test 1 demonstrates postcondition correctly rejects false success:
- Env reports SkillOutcome(ok=True)
- Env does NOT set self.holding
- Postcondition detects after.gripper.holding != "cube_red"
- Returns POSTCONDITION_FAILED
- Error propagates to agent, task completion blocked

Test 2 documents spatial consistency gap:
- Env sets self.holding="cube_red" (postcondition field check passes)
- Object pose placed 0.5m from gripper (physically impossible)
- Postcondition accepts this (only checks holding field)
- Goal check fails (object on table, not on plate)
- Task correctly fails despite postcondition passing

---

## Session 2026-11-07: Harness Engineering Setup

### Changes
- Added feature_list.json with baseline feature inventory
- Created claude-progress.md for session continuity
- Added init.sh for automated startup verification
- Documented test commands and baseline verification results
- Added README.md with project documentation and onboarding guide

### Verification
```bash
# Baseline test status
.venv/bin/python -m pytest tests/ -v
# Result: 39 passed, 5 skipped in 0.16s
```

### Design Decisions
- Adopted Learn Harness Engineering framework (instructions, project state, task scope, verification, session continuity)
- Chose to track features via feature_list.json using existing test suite as acceptance criteria
- Decided NOT to create new tests; use existing 44 tests as baseline
- Kept CLAUDE.md focused on rules; moved feature tracking to feature_list.json
- Tracked F006 (Verified pick execution) as "planned" rather than "complete" - implementation exists but needs explicit acceptance test

### Remaining Work
- None (harness setup complete)

### Known Limitations
- 5 tests skipped due to external dependencies (1 Ollama server, 4 robosuite)
- Trace data (83 episodes) not version controlled

---

## Session 2026-10-08 23:29: Robosuite Integration

### Changes
- Added RobosuiteEnv backend via subprocess bridge (commit 45cd36b)
- Implemented Python 3.10 worker isolation (robosuite_worker.py)
- Added region-based goal checking for ConstrainedPickPlace scene
- Coordinate frame alignment (z=0 at rest matches mock convention)

### Verification
```bash
pytest tests/test_robosuite_env.py -v -k 'not requires_robosuite'
# Result: 7/11 pass, 4 skip
```

### Design Decisions
- Subprocess isolation chosen over conda env to maintain Python 3.11+ for main agent
- JSON-lines protocol for worker communication (stdin/stdout)
- OSC controller at 10Hz (robosuite default)

### Remaining Work
- Full robosuite testing requires installation
- Document setup process

---

## Session 2026-10-08 23:29: LLM Backend

### Changes
- Added OllamaLLMClient with OpenAI-compatible HTTP interface (commit 8a6d228)
- CLI options for model selection, base URL, thinking mode, temperature
- Repeated run support (--repeat N with seed offset)
- Collected 75 LLM episode traces (qwen, qwen2 directories)

### Verification
```bash
pytest tests/test_ollama_client.py -v -k 'not requires_server'
# Result: 6/7 pass, 1 skip
```

### Design Decisions
- Chose Ollama for local LLM hosting (no API keys, privacy-preserving)
- Disabled Qwen3 thinking by default (slower, not always beneficial)
- Temperature default 0.0 for reproducibility

### Remaining Work
- Document Ollama setup for future contributors
- Full integration testing requires live server

---

## Session 2026-10-08 21:38: Initial Implementation

### Changes
- Core architecture: RobotEnv protocol, typed skills, agent loop (commit 719d248)
- MockTabletopEnv with deterministic kinematics and fault injection
- PickPlacePolicyLLM (rule-based) and ScriptedLLM (fixed sequences)
- Goal verification independent of skill postconditions
- JSONL trace logging

### Verification
```bash
pytest tests/ -v
# Result: 26/26 pass (mock-only subset)
```

### Design Decisions
- Protocol-based backends (RobotEnv) for mock/simulator/real-robot extensibility
- Bounded retry only for transient failures (GRASP_SLIP)
- Separate goal checking from skill execution (controller success ≠ task success)
- Deterministic mock for reproducible testing

### Known Limitations
- Mock is 2D only (no z-axis dynamics)
- Retry logic assumes state freshness validation in skill preconditions
