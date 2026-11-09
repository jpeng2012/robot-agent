# Robot-Agent Development Progress

## Session 2026-11-08: F015 — two-cube Robosuite manipulation with planner + PlanExecutor

### Changes
- robosuite_worker.py: TwoCubeScene + TwoCubeBackend (scene="two_cube"); single-cube Backend unchanged.
- robosuite_env.py: `scene=` and test-only `test_perturb_at` (sent after each reset).
- executor.py: optional `tolerances` (PHYSICAL_TOLERANCES); mock default unchanged.
- tests/test_robosuite_multi.py (new, 20); tests/test_executor.py signature assertion includes `tolerances`.

### Verification
```bash
ROBOT_AGENT_ROBOSUITE_PYTHON=~/miniconda3/envs/robot_policy/bin/python uv run python -m pytest tests/test_robosuite_multi.py -q  # 20 passed
ROBOT_AGENT_ROBOSUITE_PYTHON=... uv run python -m pytest tests/test_robosuite_env.py -q   # 22 passed
ROBOT_AGENT_ROBOSUITE_PYTHON=... uv run python -m pytest -q -m "not requires_server"       # 316 passed, 1 skipped
```
- Measured: nominal goal_met, 4 skills, 0 failures, 0 replans; perturbed run 5 skills, 1 failure
  (OBJECT_NOT_AT_POSE, no_grasp_contact), 1 replan, goal_met; same seed reproduced exactly.
- Mutation script (scratchpad): 10/11 caught; transit non-convergence guard survives (never triggered).
- Architecture + verification reviews: no blocking findings; should-fix items applied (subclass instead of
  setattr, fixed cube_b size, perturbation record kept on early refusals, transit fail-closed, reset
  clearance checks, non-vacuous reproducibility, independent ordering/displacement checks, more branches).

### Remaining limitations
- One layout; no staging/cycles/barrier; planner staging unaware of PLACE_CLEARANCE (would livelock-stop).
- Held-pose relaxation not exercised physically; reproducibility within one process/build only.

## Session 2026-11-08: F011–F014 consolidation and milestone closure (documentation only)

- Findings report: docs/robot_agent_f011_f014_findings.md (architecture evolution, result tables with denominators, failure examples,
  observed vs interpreted, limitations, future directions). Every number checked against its saved source.
- Closed: F012 end-to-end evaluation (complete), F013 and F014 (evaluation only; nothing promoted to src/).
  Experimental scripts and traces stay uncommitted in the repo root / traces/.
- Key results: trusted goals + planner + PlanExecutor solved every tested mock condition incl. one recovery;
  language grounding remains open (Qwen3 never abstained in F014; F014's gain came from the deterministic
  analysis, LLM added no measured value on the dev set).
- Tracking fix: feature_list.json had a duplicate top-level baseline_test_status (added during F012c);
  moved into meta.
- Next research direction: undecided (language grounding with confirmation / model learning / embodied execution).

## Session 2026-11-08: F014 evidence-grounded goal extraction (evaluation only, uncommitted)

- f014_evidence.py, f014_items.json (author-written dev set, frozen sha c3ca82e2 before verifier/inference);
  traces/f014_dryrun_v2 (150 gates), traces/f014_extract_v1 (66 Qwen3 calls, P-call isolation clean).
- 0 accepted-wrong in arms A/B/C (60 items). F014dev coverage: A 1/17, B 14/17, C 14/17. Qwen3 never abstained
  (0/18); every unauthorized proposal was blocked. Evidence spans always verbatim; bindings 67/92 supported.
- Known verifier defect (frozen, not fixed): modal "should end up in"/"needs to go into" -> UNCLEAR_STRUCTURE.
- Conclusion: coverage gain is from the deterministic analyser; the LLM adds no accepted coverage beyond arm C.

## Session 2026-11-08: F012 end-to-end evaluation and F013 grounding check (evaluation only, uncommitted)

- F012-eval (f012_e2e.py, traces/f012_e2e_v1, commit d253d73): 175 gates. PlanExecutor and the dependency-aware
  policy reach goal_met with 0 excess actions and 0 rejections on S1-S5, SWAP, NO_SUBGOAL; frozen Qwen3 fails S2
  and S3. Controlled valid recovery: STATE_DEVIATION -> new BFS-optimal plan -> gold goals met (1/3 replans);
  max_replans=0 stops with REPLAN_BUDGET_EXHAUSTED. Limits: gold goals vs language; mock only; one recovery case.
- F013 (f013_grounding.py, traces/f013_grounding_v1): closed restricted grammar + exact binding match; 104 gates.
  Frozen F012a replies: 0 accepted wrong, 0 false rejections (N1/N4/N2/N3 clarify, Q1 extra goal rejected,
  C1/C2 accepted then infeasible). Adversarial dev items: 0 accepted wrong, 2 contextual false rejections.
  Closed as an evaluation milestone; parser NOT promoted to src/. Development set only, no held-out claim.
- Rejected fault injections are not transactional (env step/t advance before InvalidFault); open.

## Session 2026-11-08: F012c — closed-loop symbolic plan execution and bounded replanning (mock), B006

### Changes
- src/robot_agent/executor.py (new): PlanExecutor, an LLMClient driven by the unchanged AgentLoop.
- src/robot_agent/planner.py: `_apply` -> public `simulate(state, step)` with explicit InvalidStep; plan() maps a
  rejection to PLANNER_INTERNAL/SIMULATE_REJECTED. Plans unchanged (parity 756 gates, same sequences).
- src/robot_agent/env.py (B006): InvalidFault for perturbations/silent drops that would overlap a cube, overfill
  a region, move a held, non-cube or unknown object; fail_once now honoured for place (as documented).
- tests/test_executor.py (new, 55 tests).

### Design decisions
- Coordinates always from the newest delivered state; expected post-state = simulate(delivered, issued step).
- Livelock = second replan from a physically identical state (deterministic planner => same plan), so every
  unchanged-state non-transient failure gets exactly one re-attempt; checked before a replan is counted.
- Missing/stale feedback costs one replan and an observe; never treated as success.
- Executor goals passed separately from TaskSpec.goal; AgentLoop goal_check stays the only success authority.

### Verification
```bash
uv run python -m pytest tests/test_executor.py -q            # 55 passed
uv run python -m pytest -q -m "not requires_server"          # 280 passed, 17 skipped, 3 deselected
ROBOT_AGENT_ROBOSUITE_PYTHON=~/miniconda3/envs/robot_policy/bin/python uv run python -m pytest tests/test_robosuite_env.py -q  # 22 passed
uv run python f012b2_parity.py f012c_parity_v2               # 756 gates, all pass, sequences unchanged
```
- Mutation script (scratchpad, uncommitted): 26/26 mutants of executor/planner/env caught by test_executor.py.
- Independent architecture and verification reviews: no blocking findings; should-fix items addressed
  (position-dependent livelock, planner exception as llm_error, instance reuse, bounds on pick, weaker
  advancement invariant / self-referential assertions).

### Remaining limitations
- Mock only; MISSING/STALE and place/finish prechecks reachable only via synthetic messages / _ground.
- STATE_DEVIATION end-to-end only for a moved bystander cube; silent drop and moved-into-region surface as
  execution failures (verify / env refusal) before the executor's own detection.
- physical_violations oracle mirrors placement_conflict geometry; faults do not check WORKSPACE bounds.
- InvalidFault aborts the episode without episode_end; refused silent drop already advanced step/t.
- A stop emitted on the last allowed turn ends as max_steps.

## Session 2026-11-08: F011c–F012b-2 — Qwen3 multi-object evaluation, goal grounding, symbolic planner

### Evaluation-only work (uncommitted scripts in repo root; evidence gitignored under traces/)
- F011c Stage 1 (f011c_eval.py, traces/f011c_clear_target_v1/): Qwen3:8b, P-call isolation, 15/15 valid.
  S1/S4/S5 goal_met optimal; S2 (unhinted red-first) max_steps and S3 (rejected prefix) failure_budget,
  both context-limited after the decisive error. First harmful decision: red placed into parking (blue's goal).
- Plan-first (f011c_planfirst.py, traces/f011c_planfirst_v1/): the plan was emitted with the observe call
  (not_separate, unparseable prose) 3/3; failure_budget.
- Plan-only on the authoritative S2 state (f011c_planonly.py): invalid red-first plan 3/3. Goal order reversed
  (f011d_goalorder.py): optimal 3/3, no dependency stated. Validate-and-repair (f011d_repair.py): same invalid
  plan for 3 candidates, 0/3 repaired.
- F012a goal extraction (f012a_goals.py, f012a_items.json, traces/f012a_extract_v1/): 22 items + 6 repeats.
  Clear instructions 15/16 exact (pairs order-invariant, contradictions extracted and rejected as infeasible);
  0/4 abstentions (invented goals; 2 would pass validation); no-state items invent ids. Not safe to connect
  to execution.
- F012b-1 planner (f012b_planner.py, traces/f012b_planner_v4/): 703 gates; 109 feasible cases valid and equal
  to an independent BFS relaxation optimum (witness replayed); goal re-prioritisation fixed 4/30 suboptimal cases.

### F012b-2 production changes
- src/robot_agent/planner.py (new): plan(goals, state, bounds, max_physical) -> Plan | PlanningFailure.
- src/robot_agent/policies.py: bounded staging shared via _staging_points(bounds)/_staging(..., bounds)/
  decide(..., bounds); structured stop codes in meta["code"] (text unchanged); defaults preserve F011b-1.
- tests/test_planner.py (new): 53 tests (execution through AgentLoop, permutation invariance, chain and
  cycle ordering, structured failures, bound edges, tie-breaks, non-mutation of state/env).

### Verification
```bash
pytest tests/test_planner.py -q             # 53 passed
pytest tests/ -q -m "not requires_server"   # 225 passed, 17 skipped, 3 deselected
ROBOT_AGENT_ROBOSUITE_PYTHON=~/miniconda3/envs/robot_policy/bin/python pytest tests/test_robosuite_env.py -q  # 22 passed
python f012b2_parity.py f012b2_parity_v2    # 109 + 54 bounded cases, 756 gates pass (eval-only)
python f011c_eval.py dry-run f011c_dryrun_f012b2   # reference metrics/trajectories identical to dryrun_v5
```
- 20 mutations (ordering, priority rules, tie-break, occupancy checks, staging clearance/bounds, failure
  classification, validation checks, bound default/off-by-one, edge comparisons, input mutation, canonical
  order) each caught; sources restored by hash.
- Independent architecture and verification reviews: GO; should-fix items applied (PLANNER_INTERNAL for
  internal faults and unknown stops, explicit stop-code check instead of assert, INVALID_STATE, Literal step
  fields, bound/tie-break/boundary tests, gated and bounded parity).

### Remaining limitations
- Plans are open-loop and carry no expected post-states; F012c must ground each step and re-plan.
- Optimality is empirical (cube-only, disjoint regions, <= 4 cubes); BFS ignores bounds.
- BLOCKER_NOT_FOUND / policy CONFLICTING_GOALS stops are unreachable through validated input (tested via
  monkeypatch / decide()). Robosuite: one cube and one region; reach and barrier not modelled.
- Rerunning frozen eval harnesses now records a different src diff hash (expected).

## Session 2026-11-08: F011b-1 — naive vs dependency-aware deterministic policies (mock)

### Changes (new files only)
- src/robot_agent/policies.py: NaivePolicy and DependencyAwarePolicy over one pure decide() on the latest
  delivered world_state; InRegion subgoals only, stateless, at most one tool call per turn. DA clears an
  occupied destination (blocker to its own free goal or a staging point outside every region), then resumes.
- tests/test_policies.py: 23 tests.

### Behaviour (clear_target unless noted)
- Naive correct: goal_met, 4 executes. Naive reversed: failure_budget after 8 validation-layer
  DESTINATION_OCCUPIED (10 turns, 1 execute).
- DA correct and reversed: goal_met in the optimal 4 executes, 0 rejections.
- Swap layout: DA stages red at (-0.35, 0.3), 6 executes. Naive: failure_budget.
- Blocker without subgoal: DA stages blue at (-0.05, 0.3), 4 executes. Naive: failure_budget.
- Free target, reversed order: both follow the supplied order (no false dependency).
- Recovery after a scripted rejected plan: DA puts red back, parks blue, places red; 1 rejected plan,
  reactively recovered, 6 executes, 2 unnecessary actions.

### Verification
```bash
pytest tests/test_policies.py -v            # 23 passed
pytest tests/ -q -m "not requires_server"   # 172 passed, 17 skipped, 3 deselected
ROBOT_AGENT_ROBOSUITE_PYTHON=~/miniconda3/envs/robot_policy/bin/python pytest tests/test_robosuite_env.py -q  # 22 passed
```
- Mutations (each fails its targeted tests; policies.py restored by hash):
  - DA without the pre-check; staging inside regions; chained pick+place; hard-coded coordinates;
  - reading the first tool result; always reordering; naive resolving dependencies; hidden state;
  - changed staging tie-break; naive placing into another subgoal's region.
- All 10 recorded against the final 23-test suite: mutations_v2.txt (M1-M8) and
  mutations_followup.txt (M9-M10). mutations.txt holds the earlier run of M1-M8 against the 22-test
  version.
- Evidence (gitignored): traces/f011b1_v1/.

### Review findings and corrections
- First review: commit. Corrections applied, test-only:
  - staging points pinned;
  - the recovery test now checks grounding;
  - place checks require the held cube's own goal region;
  - unnecessary_actions is None on failed episodes;
  - violations_executed renamed to invalid_state_observations;
  - rejection grouping only when adjacent;
  - unknown-region rejections counted, not raised;
  - proactive clearance computed from the trace and reported next to the policy's meta.mode count.
- Deferred limitations:
  - The stop paths are untested: conflicting goals (needs overlapping regions), no valid staging
    location, blocked with no blocker, naive holding a cube with no subgoal, and the duplicate-object
    constructor check.
  - A subgoal object or region missing from the delivered state raises KeyError, which the loop reports
    as llm_error.
  - The _blocker/placement_conflict equivalence is tested only on states traced in four layouts.
  - On-goal tasks are unsupported; there is no evaluation harness yet.

## Session 2026-11-08: F011a — multi-object occupancy and clear-target task (mock)

### Changes
- world.py: FailureCode.DESTINATION_OCCUPIED; CUBE_HALF 0.02, REGION_HALF 0.045; pure
  placement_conflict(): R2 region capacity first (target_region box already holding another cube's
  centre, same strict test as InRegion), then R1 cube footprint non-overlap. Held cube ignored; only
  cubes are solid.
- skills.py: _place_pre runs the check after NOT_HOLDING. env.py: MockTabletopEnv._place runs it on the
  live scene before release; _pick refuses non-cube objects ("'<id>' is not graspable").
- goals.py: clear_target_task() (blue in target, red elsewhere, parking empty; goal blue in parking AND
  red in target). Order follows from region capacity; no ordering predicate.
- tests/test_clear_target.py: 21 deterministic ScriptedLLM tests with an overlap/capacity invariant on
  every traced state.
- Unchanged: F010 grounding, tool schemas, policy classes, SYSTEM_PROMPT, memory, robosuite worker.

### Verification
```bash
pytest tests/ -q -m "not requires_server"   # 149 passed, 17 skipped, 3 deselected
ROBOT_AGENT_ROBOSUITE_PYTHON=~/miniconda3/envs/robot_policy/bin/python pytest tests/test_robosuite_env.py -q  # 22 passed
```
- Mutations (each fails its targeted tests; sources restored identically): env check off; precondition
  off; held cube not excluded; plates/regions as obstacles; R2 removed; release before check (first
  attempt never released at all, redone faithfully); code made transient (caught by a direct pin, since
  no scenario can expose a retry); non-cube picks allowed.
- Evidence (gitignored): traces/f011a_v1/ (mutations.txt, mutation_m6_redo.txt, nonserver.txt,
  robosuite.txt).

### Independent review
- Verdict: commit, with warnings; no correctness defects, within scope.
- Limitations:
  - DESTINATION_OCCUPIED pre-empts UNGROUNDED_ACTION for a place chained after a pick into an occupied
    spot (validation precedes grounding; rejected without motion; untested).
  - B006: silent_drop_on_place and perturb_at can still create overlapping cubes, region over-capacity
    or invalid held-object states. Must be resolved before F011b disturbance evaluations.
  - B004: DESTINATION_OCCUPIED originates from harness validation (validation_error, no motion) or env
    execution (env_outcome failure); memory treats both alike.
  - Untested: y-axis boundaries, R2-before-R1 ordering by mutation, gripper pose after a refused
    non-cube pick, env-side error text, plate inside a region.
- Deferred to F011b: MultiPickPlacePolicyLLM, planning comparison, evaluation harness, CLI task.

## Session 2026-11-08: Milestone 1 Phase B2 — robosuite rejection and recovery paths through AgentLoop

### Changes (tests/test_robosuite_env.py only; no production change)
- Test-local fixtures: CountingEnv (records RobotEnv.execute() calls, attributed at call time to the tool
  call being handled), HybridLLM (scripted failure prefix, then PickPlacePolicyLLM reading only the
  message list), failure_layer (classifies each failed call by event chain, since codes are shared
  between harness and worker), assert_rejected_without_motion, assert_recovery_grounded.
- Counters kept distinct: RobotEnv.execute() calls (client side), backend WorldState.step (incremented
  by the worker even when it rejects), simulator control_steps (env_outcome.info, absent on rejection).

### Coverage, by layer
- Harness rejection, no execute() call: T1 blind correct pick -> UNGROUNDED_ACTION; T2 pick 0.08 m off
  after observe -> OBJECT_NOT_AT_POSE (validate_call, no grounding_check); T3 premature finish ->
  GOAL_NOT_MET; T6 place chained after a physical pick in one turn -> UNGROUNDED_ACTION ("holding changed").
- Worker rejection before motion: T4 place at (0.30, 0.30) inside the ±0.5 schema bounds but outside
  robot reach (seed 0) and on the barrier at (0.0, 0.14) (seed 7) -> OUT_OF_WORKSPACE; T5 pick("target")
  -> UNKNOWN_OBJECT with the worker's "not graspable" detail. Shown by one execute() call, step +1,
  info == {}, and unchanged sim time, object and gripper poses and holding; the grasp survives a rejected
  place; no retry.
- Physical failure after motion: NOT tested (see B005).
- Every test recovers to goal_met; each recovery pick/place is grounded by the tool_result delivered
  just before its turn (source_seq, source_call_id, source_step) and its arguments equal that state.
- T7: reset after an episode ending while holding restores the seed-0 initial state.

### Verification
```bash
ROBOT_AGENT_ROBOSUITE_PYTHON=~/miniconda3/envs/robot_policy/bin/python pytest tests/test_robosuite_env.py -v
# 22 passed (two full runs 50.7 s / 49.9 s, post-review run 49.8 s); T4 alone 2 passed, T5 alone 1 passed
pytest tests/ -q -m "not requires_server"   # 128 passed, 17 skipped, 3 deselected
```
- Per test ~4.3-4.9 s (T7 ~1.7 s).
- Mutations (all fail exactly the targeted tests):
  - classifier treats worker rejections as physical -> T4 x2, T5 fail;
  - loop grounding check disabled -> T1, T6 fail;
  - worker steps the sim 3 times before rejecting (worker restored, sha256 a6f2a01f...) -> T4 x2, T5
    fail on sim time (6.9 -> 7.2 s, 7.0 -> 7.3 s, 0.0 -> 0.3 s).
- Evidence (gitignored): traces/robosuite_phaseB2_v1/ (run1, run2, run3_after_review, t4_alone,
  t5_alone, mutation_layer, mutation_grounding, mutation_settle_before_reject, nonserver).

### Independent review findings
- Verdict commit. Applied: exact object-state comparison, worker-detail assertion in T5, corrected
  HybridLLM docstring.
- Limitations:
  - Place-argument checks cannot detect hard-coding, since the target pose is a layout constant; a
    hard-coded seed-0 pick fails only by a ~1.6e-19 settle residue.
  - Execute attribution reads the loop's own tool_call events.
  - No mutation guards the provenance fields, attribution, argument equality or T7 reset equivalence.
  - Only the default test order was run; T7 compares realized init, holding, step and t, not the full
    sim/controller state.
  - The robosuite module now takes ~50 s (was ~16 s).
- Uncovered on robosuite: B005 (physical failures after motion, retries, F010 "moved", loop terminations,
  worker-only branches, displaced-cube recovery).

## Session 2026-11-07: Milestone 1 Phases A and B1 — robosuite baseline and seeded resets

### Phase A (no code changes)
- First recorded run of the robosuite tests: 10/10 passed (4 @sim) with
  ROBOT_AGENT_ROBOSUITE_PYTHON=~/miniconda3/envs/robot_policy/bin/python. Only
  test_rule_based_agent_loop_succeeds goes through AgentLoop; the other @sim tests call the env directly.
- Evidence: traces/robosuite_phaseA_v1/ (pytest output + the single AgentLoop trace).
- Review finding: the worker's np.random.seed had no effect; robosuite 1.5.2 draws cube size and
  robot init-joint noise from env.rng, and the placement sampler has its own generator.

### Phase B1 changes
- robosuite_worker.py: every reset installs fresh generators for env.rng and the placement sampler
  (independent SeedSequence(seed).spawn(2) streams when seeded; unseeded default_rng() otherwise, so
  seed=None never replays an earlier seed). backend_info gains `init` (seed, installed RNG source and
  entropy, robot_init_qpos pre-settle, cube size, z_ref, post-settle cube/eef poses) and `versions`
  (python, robosuite, mujoco, numpy, sha256 of worker/env/expert sources).
- tests/test_robosuite_env.py: 4 @sim tests (same seed -> identical realized init; seeds vary joints
  and cube size; seed=None installs fresh unseeded generators, checked via recorded RNG metadata rather
  than sample differences; versions/hashes recorded). All 4 fail on the pre-change worker; the
  seed=None test fails when the None branch keeps the previous generators (mutation run recorded in
  traces/robosuite_phaseB1_v2/test_evidence.txt).

### Evidence (traces/robosuite_phaseB1_v2/; v1 superseded)
- init_repro/: same seed -> identical realized initial state and observed state across two fresh
  worker processes; seed 0 identical first or after seed 1; seeds 0 and 1 differ; seed=None differs
  across processes, including after seeded resets.
- baseline/results.json: 10 seeded AgentLoop episodes (PickPlacePolicyLLM, region_pick_place, one
  fresh worker per episode) + repeats of seeds 0 and 3. Worker sha256 a6f2a01f... recorded in every
  episode; `sha256sum src/robot_agent/robosuite_worker.py` and `git diff HEAD -- src/ | sha256sum`
  (27f7d45c...) were checked equal to the recorded values before commit. robot-policy-learning 2c9ec6a,
  tracked files clean. Harness robosuite_baseline.py (uncommitted) sha256 86dc33ad...; repro_check.py
  sha256 4e7fc74f....
- All 12 episodes (10 distinct seeds + repeats of 0 and 3): goal_met, one pick and one place attempt, 0 retries, 0 failures, 0 F007 rejections, both
  F010 checks ok. Over the 10 distinct seeds: pick gripper-cube XY 0.000796-0.001145 m (mean 0.000899); cube z after pick
  0.1017-0.1019 (agent frame); episode control steps 325-357.
- Same seed reproduces the observed initial state (seeds 0 and 1, two processes; seed 0 also after
  seed 1). For seeds 0 and 3 (one repeat each) it reproduces the recorded skill endpoints and every
  other recorded event except call ids and timing. Complete per-control-step trajectories were not
  compared, so full trajectory determinism is not claimed. One machine, mujoco 3.3.5.

### Verification
```bash
ROBOT_AGENT_ROBOSUITE_PYTHON=~/miniconda3/envs/robot_policy/bin/python pytest tests/test_robosuite_env.py -q  # 14 passed
pytest tests/ -q -m "not requires_server"   # 128 passed, 9 skipped, 3 deselected
```

### Limitations
- n=10, one fixed layout (layout fixes cube/target poses; seeds vary cube size and joint noise only),
  deterministic scripted policy; no statistical claim. No failure path exercised on robosuite.
- F007 z check cannot fire after a worker-ok pick; F010 checks passing do not show enforcement.
- Place-phase gripper-cube XY is post-release, not a grasp measurement.
- Worker log deleted on close; env exceptions not caught by AgentLoop; no worker RPC timeout.

## Session 2026-11-07: F010 Observation-grounding contract

### Changes
- world.py: FailureCode.UNGROUNDED_ACTION.
- skills.py: requires_grounding() (skills with coordinate args: pick, place) and pure grounding_failure().
- loop.py: Grounding record (state rebuilt from the delivered world_state, source tool_result seq and call
  id), reset per episode; snapshot taken once per llm.step turn; check in _handle after validate_call and
  before dispatch; grounding_check trace event; llm_turn records grounding_source_seq/grounding_step.
- Tests: 27 new loop tests and 2 skill unit tests; 7 older tests gained an initial observe or a restructured
  multi-call turn (tests/test_loop.py x5, tests/test_ollama_client.py x2), assertions unchanged.

### Design decisions
- Grounding comes only from world states the loop returned in tool results (incl. failure, rejection and
  SKIPPED results). Task text, memory and model output never ground.
- Per-turn snapshot: every call in a turn is judged on what the LLM had seen before that step.
- Object-specific freshness: pick needs the target in the delivered state and not moved > GRASP_TOL;
  place needs the delivered holding state to match live. Destination pose is out of scope.
- After validate_call (existing codes keep precedence); once per call (GRASP_SLIP retries unchanged).
- Rejections return fresh state, count toward max_failures; SYSTEM_PROMPT unchanged.
- Pre-implementation trace scan (151 real Qwen3 episodes): 0 multi-call turns; 21 pick/place calls before any
  delivered state, all in memory-on runs.

### Verification
```bash
pytest tests/ -q -m "not requires_server"   # 128 passed, 5 skipped, 3 deselected
pytest tests/ -q -rs                        # 131 passed, 5 skipped (Ollama 0.40.2 up)
# skips: robosuite x4 (ROBOT_AGENT_ROBOSUITE_PYTHON unset), RUN_OLLAMA x1
```
- Mutation check: grounding disabled -> all 9 original rejection tests fail.
- Independent verification review: PASS on all five contract points; commit approved.

### Review findings / remaining
- B004: UNGROUNDED_ACTION enters F009 memory failure codes and the next success is recorded as a recovery;
  separate harness/policy validation failures from environment experiences before memory-on evaluations.
- pick does not compare holding state (contrived same-turn place-then-pick passes); place destination not
  grounded; future coordinate skills need explicit rules; no max state age for real hardware.
- Recorded Qwen3 blind picks were at (0.5, 0.5) and still fail OBJECT_NOT_AT_POSE first; F010 fires only on
  blind-but-correct picks and same-turn chains.
- Robosuite tests not run.

## Session 2026-11-07: F009b Stage 2 — recovery-action fix, Ollama reproducibility audit

### Changes
- 0f73e9f: recovery_actions now list only calls whose own outcome succeeded, derived per call_id from
  validation_error, env_outcome, verify and tool_result. Episode success is never used. Tests for rejected
  placement after a failed pick, mixed sequences, tool_result-only success, retries, skipped calls, missing ids.
- Hardening commit: INDEX_VERSION=2 (older indexes rebuilt even with unchanged mtimes), duplicate call IDs
  treated as ambiguous, non-dict args tolerated, end-to-end AgentLoop trace test, failed-finish tests.
  Success eligibility now requires episode_end success/goal_met and a passing FINAL goal_check (was: first
  goal_check). Corpus eligibility unchanged (77/77 indexed, identical contents).
- Evaluation-only scripts (untracked): pilot_memory_eval.py, ablation_first_action.py,
  ablation_recovery_variants.py, ollama_repro_audit.py; scratch: compare_memory_fix.py, inspect_memory.py.

### Findings
- Old parser recorded rejected place() calls as recoveries; corrected memory removed NOT_HOLDING attempts.
- Memory-on runs then opened with an ungrounded pick(cube_red, 0.5, 0.5), always rejected by validation.
- Reproducibility audit (traces/ollama_repro_audit_v1): identical request bytes give observe (cold, repeat,
  turn-1 prime) or pick (after the B5 prompt); tool-name logprob gap +6.09 nats cold vs -0.09 after B5.
  Explicit seed has no effect. Mechanism unverified.
- All earlier memory-on/off comparisons are confounded by server cache history (traces/CONFOUNDED_COMPARISONS.md).

### Verification
```bash
pytest tests/ -q -rs
# Result: 104 passed, 5 skipped (robosuite x4, RUN_OLLAMA x1). The 3 requires_server Qwen3 tests in
# test_loop.py ran against the local Ollama (mark not registered, so they do not skip without a server).
```

### Remaining
- F009 behavioral evaluation remains partial/confounded. 30-pair memory-on/off evaluation not run; needs controlled server history per LLM call.
- Only the first success after a failure is recorded; observe can occupy the slot.
- BACKLOG — parser/loop divergence (present since 0f73e9f, not a regression, not fixed): an env SkillOutcome(ok=False,
  failure=None) is a success to the loop (verify returns None) but a failure to the parser, which can create
  a spurious pending failure. Mock backends are not known to emit it.
- BACKLOG — INDEX_VERSION=2 makes the version-1 traces/.memory_index_frozen.json stale: eval scripts that copy it now
  rebuild from live corpus dirs (exclude_patterns ["test_"]) and their frozen-index hash no longer proves a
  frozen corpus. Regenerate the frozen index at version 2 before any further evaluation.
- Observation-grounding contract: implemented as F010 (see next session entry).

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
