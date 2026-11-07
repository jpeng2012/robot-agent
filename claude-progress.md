# Robot-Agent Development Progress

## Session 2026-11-07: F008 Baseline Qwen3 Replanning Evaluation

### Changes
- Added test_failure_context_structure (deterministic infrastructure check)
- Added test_qwen3_nominal_pick_and_place (Eval 1: baseline)
- Added test_qwen3_recovers_from_moved_object (Eval 2: OBJECT_NOT_AT_POSE recovery)
- Added test_qwen3_recovers_from_postcondition_failure (Eval 3: POSTCONDITION_FAILED recovery)
- All live tests marked with @pytest.mark.requires_server

### Verification
```bash
# Deterministic infrastructure test (no Ollama required)
pytest tests/test_loop.py::test_failure_context_structure -v
# Result: 1/1 passed

# Live Qwen3 evaluations (requires Ollama at localhost:11434)
pytest tests/test_loop.py -k 'qwen3' -v -s --tb=short
# Result: 3/3 passed
```

### Design Decisions
- **Evaluation focus**: Measure baseline Qwen3 capability, not predetermined sequences
- **Minimal deterministic testing**: One infrastructure check, not scripted action sequences
- **Live LLM evaluations**: Three scenarios with trace analysis
- **No prompt enhancements**: Defer guidance/examples until after baseline measurements
- **Metrics collection**: Task success, recovery success, LLM turns, tool calls, termination reasons
- **Trace analysis**: Extract Qwen3's coordinate choices and recovery patterns from JSONL logs

### Baseline Measurements
```
Nominal execution: success=True, llm_turns=4, skills_executed=2, failures=0
Tool call sequence: observe → pick → place → finish

Moved-object recovery: success=True, llm_turns=5, skills_executed=3, failures=1, reason=goal_met
Pick coordinates: [(0.1, -0.2), (0.2, -0.2)]
Coordinate change: Δx=0.1000, Δy=0.0000

Postcondition-failure recovery: success=True, llm_turns=5, skills_executed=3, failures=1, reason=goal_met
Event sequence: observe → pick → FAIL(POSTCONDITION_FAILED) → pick → place → finish
```

### Observed Behavior
- ✅ Qwen3 interprets failure messages and updates coordinates after OBJECT_NOT_AT_POSE
- ✅ Qwen3 retries pick after POSTCONDITION_FAILED
- ✅ Baseline success rate in nominal execution: 100% (1/1 runs, seed=0)
- ✅ Recovery success rate: 100% (2/2 failure scenarios)
- ✅ Average recovery overhead: +1 LLM turn, +1 skill execution per failure

### Key Findings
1. **Replanning infrastructure works**: Qwen3 successfully receives and interprets structured failure messages
2. **Coordinate updating confirmed**: Δx=0.10m after SlipAndNudgeEnv nudge (expected ~0.10m from line 54)
3. **Postcondition recovery confirmed**: Qwen3 retries pick after false-success detection
4. **Efficient recovery**: Only +1 turn overhead for both moved-object and postcondition failures
5. **No prompt guidance needed**: Qwen3 achieves 100% success with minimal 9-line system prompt
6. **Functional validation passed**: All evaluations use seed=0; reliability under randomized disturbances not yet characterized

### Limitations
- **Single-seed evaluation**: All tests run with seed=0; statistical confidence across random seeds not established
- **Limited failure scenarios**: Only three failure types tested (GRASP_SLIP, POSTCONDITION_FAILED, OBJECT_NOT_AT_POSE)
- **No multi-failure sequences**: Recovery from consecutive different failure types not evaluated
- **No stress testing**: Behavior near budget limits (max_failures=8, max_turns=20) not measured

### Remaining Work
- None (F008 complete with documented limitations)

---

## Session 2026-11-07: F007 Spatial Consistency Verification

### Changes
- Added Pose.dist_xyz() method to world.py
- Added CARRY_XY_TOL and MIN_CARRY_Z constants to skills.py
- Enhanced _pick_post with spatial consistency checks (XY distance and Z height)
- Modified test_inconsistent_state_holding_without_contact to expect postcondition failure
- Added test_pick_postcondition_rejects_object_on_table
- Added test_pick_postcondition_accepts_minimal_carry_height
- Added test_pick_postcondition_accepts_max_carry_xy_error

### Verification
```bash
# F007 tests pass
pytest tests/test_loop.py::test_inconsistent_state_holding_without_contact \
  tests/test_loop.py::test_pick_postcondition_rejects_object_on_table \
  tests/test_loop.py::test_pick_postcondition_accepts_minimal_carry_height \
  tests/test_loop.py::test_pick_postcondition_accepts_max_carry_xy_error -v
# Result: 4/4 passed

# Full deterministic suite (no regressions)
pytest tests/ -v -k 'not requires_server and not requires_robosuite'
# Result: 44 passed, 5 skipped in 0.17s (was 41 passed before)

# F006 tests still pass
pytest tests/test_loop.py::test_pick_postcondition_catches_false_success \
  tests/test_loop.py::test_failed_pick_recovers_by_bounded_retry -v
# Result: 2/2 passed
```

### Design Decisions
- **Threshold selection**: CARRY_XY_TOL=0.05m (2.5x larger than GRASP_TOL) accommodates controller variance during lift
- **Z-height threshold**: MIN_CARRY_Z=0.05m (below typical 0.08-0.10m carry heights, catches z≈0.0 resting objects)
- **Coordinate frame**: Both backends use agent frame z=0 as resting height (MockTabletopEnv and RobosuiteEnv)
- **Reused existing utilities**: Pose.dist_xy pattern, similar to dist_xyz addition
- **No new failure codes**: Uses existing POSTCONDITION_FAILED

### Implementation Details
Enhanced _pick_post now performs three checks:
1. Gripper holding field matches expected object_id (existing check)
2. Object XY position within CARRY_XY_TOL of gripper (new spatial check)
3. Object Z height >= MIN_CARRY_Z (new elevation check)

Test coverage includes:
- XY inconsistency detection (object 0.5m away from gripper)
- Z inconsistency detection (object at z=0.0 while held)
- Boundary case: minimal carry height (z=0.051m, just above threshold)
- Boundary case: maximal XY error (0.049m, within tolerance)

### Observed Behavior Changes
test_inconsistent_state_holding_without_contact now detects spatial inconsistency:
- **Before F007**: Postcondition passes, error codes = ["GOAL_NOT_MET"]
- **After F007**: Postcondition fails, error codes = ["POSTCONDITION_FAILED", "GOAL_NOT_MET"]
- **Earlier detection**: Spatial violation caught at postcondition instead of goal check

### Remaining Work
- None (F007 complete)

### Robosuite Compatibility

**Verified by source analysis:**
- MIN_CARRY_Z (0.05m): Compatible with LIFTED_Z=0.08m requirement ([robosuite_worker.py:27,209](src/robot_agent/robosuite_worker.py#L27))
- Coordinate frame: Both backends use agent frame z=0 as resting ([robosuite_worker.py:9-11,73-74](src/robot_agent/robosuite_worker.py#L9-L11))
- Expert lifts to ~0.10m, providing 60% margin above MIN_CARRY_Z

**Empirically unverified:**
- CARRY_XY_TOL (0.05m): No live Robosuite data available
- Physical analysis suggests 0-0.03m typical offset (within threshold)
- Risk: LOW (2.5x GRASP_TOL margin)
- Mitigation: B003 backlog item for empirical validation

**Design decision:**
- Prioritize detecting false grasps (holding=True but incompatible pose) over Robosuite deployment risk
- Thresholds can be adjusted via one-line constant change if empirical data shows issues
- Test infrastructure exists in test_direct_pick_then_place_meets_goal to measure XY distance

---

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
