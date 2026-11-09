# Robot-Agent F011–F014: Findings

Status as of 2026-11-08, at commit `d253d73`. The mock backend was used throughout, and Qwen3:8b via Ollama
(digest `500a1f06…`) was the only model.

Committed production work and evaluation-only work are kept separate below. Evaluation scripts live in the
repository root and are not committed. Their evidence is under `traces/`, which git ignores.

## 1. Executive summary

- **Reactive tool calling (F011).** With Qwen3:8b choosing one tool call per turn, the multi-object
  clear-target task succeeded when the blocking cube was mentioned first or the blocker was pointed out. It
  failed whenever the goal that had to wait was listed first and no hint was given.
  - Prompting for an explicit plan did not help. Neither did planning from the authoritative state alone,
    nor a validate-and-repair loop.
  - Reversing the goal order alone made the plan optimal in 3/3 calls. In this configuration, goal
    presentation order drove the action order.
- **Hybrid planning (F012).** Given trusted goals, a symbolic dependency planner plus a closed-loop
  `PlanExecutor` behind the unchanged `AgentLoop` solved every tested mock condition with zero rejected
  calls and zero excess actions. That includes one controlled recovery from an unexpected state change.
- **Language grounding (F012a, F013, F014).** This remains the open problem.
  - Qwen3 extracted explicit goals well, but invented goals when it should have abstained.
  - A restricted grammar (F013) blocked every invented goal it saw, but covered little paraphrased language.
  - An evidence-citing extractor with a deterministic verifier (F014) raised coverage on author-written
    development items without accepting any wrong goal set. The deterministic analysis alone accounts for the
    whole gain; the LLM added no measured value on that set.

## 2. Architecture evolution

| Stage | Who decides what | Status |
|---|---|---|
| Reactive (F008–F011) | LLM chooses each tool call; `AgentLoop` validates, grounds (F010), verifies postconditions and checks the goal | committed |
| Deterministic reference (F011b-1) | `DependencyAwarePolicy` clears blockers reactively from the latest state | committed (`380b0cf`) |
| Symbolic planner (F012b) | `planner.plan(goals, state)`: canonical goals, per-turn reprioritisation, coded failures | committed (`9b518d9`) |
| Closed-loop executor (F012c) | `PlanExecutor` (an `LLMClient`): one grounded call per turn, predicted-state checks, bounded replanning; B006 fault guards | committed (`d253d73`) |
| Goal extraction and grounding (F012a, F013, F014) | LLM or grammar proposes goals; deterministic checks accept or block them | **evaluation only, not connected to execution** |

Every stage keeps the same safety boundary. All actions go through `validate_call`, F010 grounding,
`verify`, and `AgentLoop`'s `goal_check` against `TaskSpec.goal`, which is the sole authority on success.

## 3. Results

### F011 — Reactive LLM planning (Qwen3:8b)
**Setup:** P-call isolation (the model is unloaded before every call and the cache is verified empty),
temperature 0, thinking off, memory off.

**Committed groundwork:**
- F011a: occupancy rules and the `clear_target` task.
- F011b-1: naive and dependency-aware reference policies. The naive policy fails the reversed order with 8
  rejections and ends on its failure budget; the dependency-aware policy is optimal (4 actions).

**F011c Stage 1.** Source: `f011c_eval.py`, `traces/f011c_clear_target_v1/`. 5 scenarios × 3 repeats =
15/15 valid episodes over 159 isolated calls.

| Scenario | Outcome (3/3 identical) | Physical actions | Rejections per episode |
|---|---|---|---|
| S1 red-first, blocker hinted | goal_met, optimal | 4 | 0 |
| S2 red-first, no hint | max_steps | 10 | 4 DESTINATION_OCCUPIED, 3 GOAL_NOT_MET |
| S3 scripted rejected first move, then agent | failure_budget | 4 (including prefix) | 1 DESTINATION_OCCUPIED (+1 in the prefix), 6 GOAL_NOT_MET |
| S4 red-first, target free | goal_met, optimal | 4 | 0 |
| S5 blue-first, no hint | goal_met, optimal | 4 | 0 |

S2 and S3 later hit the context limit (6 prompt truncations), but their first wrong decision came before
it. For S2 that was turn 3, at 1431 prompt tokens.

**Follow-up probes on the S2 state.** All isolated; each "3/3" is three repeated calls of one fixed input.

| Probe | Source | Result |
|---|---|---|
| Plan-first prompting | `f011c_planfirst.py`, `traces/f011c_planfirst_v1/` | Control (1 episode): max_steps. Plan arm: in 3/3 the plan came as free text in the turn with `observe`, could not be parsed, and the episode ended failure_budget |
| Planning only (authoritative state, JSON plan, no tools) | `f011c_planonly.py`, `traces/f011c_planonly_v1/` | 3/3 identical **invalid** plans: red into the occupied target at step 1 |
| Goal order reversed (blue-first), otherwise identical | `f011d_goalorder.py`, `traces/f011d_goalorder_v1/` | 3/3 **valid and optimal** (4 actions), blue first, `dependencies: ""` |
| Validate, then repair with validator feedback | `f011d_repair.py`, `traces/f011d_repair_v1/` | 3 repeats × 3 candidates = 9 calls; 0/3 repaired (3/3 constraint failures in every repeat) |

### F012 — Symbolic planning and closed-loop execution

**F012a goal extraction (evaluation only).** Source: `f012a_goals.py`, `f012a_items.json`,
`traces/f012a_extract_v1/`. 22 items plus 6 determinism repeats.
- **Explicit instructions with state:** 15/16 exactly correct. Q1 ("Put the red cube in the target region.")
  gained an unrequested `red→parking`. Both contradictory requests (C1, C2) were extracted literally.
- **Should-abstain items:** 0/4 abstained, and all four produced goals.
  - N1 (green cube) and N4 ("the cube") produced feasible goals that passed validation.
  - N2 and N3 were rejected only at the feasibility stage.
- **No-state items:** 2/2 invented ids.
- **Repeats:** 6/6 identical.

**F012b-1 planner and BFS oracle (evaluation only).** Source: `f012b_planner.py`,
`traces/f012b_planner_v4/`. 703/703 gates.
- 94 feasible start layouts and 15 holding-at-start layouts (109 in total) all produced plans that replay
  validly. Each plan's length equals an independent BFS lower bound, and a replayed witness confirms that
  bound is achievable.
- A static goal ordering was suboptimal on 10/94 layouts (4/30 in the RAND3 family). This motivated
  per-turn reprioritisation.

**F012b-2 and F012c (committed).**
- Planner: 53 tests. Parity with F012b-1: 756/756 gates (109 cases plus 54 bounded ones), with identical
  sequences.
- Executor: 55 tests. 26/26 mutants were caught by `tests/test_executor.py`, using a session scratch script
  that is not in the repository (the result is recorded in `claude-progress.md`).
- Full non-server suite: 280 passed, 17 skipped, 3 deselected. Robosuite file: 22 passed.

**F012 end-to-end (evaluation only).** Source: `f012_e2e.py`, `traces/f012_e2e_v1/`. 175/175 gates.
Each cell shows success, successful actions (excess), and rejected agent calls.

| Condition | Optimum | Reactive Qwen3 (frozen F011c, 3/3) | Dependency-aware policy | PlanExecutor (gold goals) |
|---|---|---|---|---|
| S1 | 4 | ✓ 4 (0) 0 | ✓ 4 (0) 0 | ✓ 4 (0) 0 |
| S2 | 4 | ✗ max_steps, 10 skills, 7 | ✓ 4 (0) 0 | ✓ 4 (0) 0 |
| S3 | 6 | ✗ failure_budget, 4 skills, 7 | ✓ 6 (0) 0 | ✓ 6 (0) 0 |
| S4 | 4 | ✓ 4 (0) 0 | ✓ 4 (0) 0 | ✓ 4 (0) 0 |
| S5 | 4 | ✓ 4 (0) 0 | ✓ 4 (0) 0 | ✓ 4 (0) 0 |
| SWAP | 6 | N/A | ✓ 6 (0) 0 | ✓ 6 (0) 0 |
| NO_SUBGOAL | 4 | N/A | ✓ 4 (0) 0 | ✓ 4 (0) 0 |
| RECOVERY (blue moved back into target before red's pick) | 8 (given the disturbance) | N/A | ✓ 8 (0) 0 | ✓ 8 (0) 0; 1 replan |
| RECOVERY with `max_replans=0` | — | N/A | — | ✗ REPLAN_BUDGET_EXHAUSTED after 3 actions |

All runs had 0 grounding violations and 0 physically invalid traced states, checked by both the frozen
harness and an independent geometry check. In the recovery case the executor flagged an unexpected state
change at step 2, before any rejected call. It dropped the stale plan, ran a new 5-action plan (the BFS
optimum from the changed state) and met the gold goals.

### F013 — Restricted instruction grounding (evaluation only)
Source: `f013_grounding.py`, `traces/f013_grounding_v1/`. 104/104 gates, no model calls.

A closed-vocabulary command grammar derives the goals the instruction explicitly requests. The proposed set
must equal them exactly; anything outside the grammar leads to clarification.

| Set | Correct accepted | Contradictory accepted, then infeasible | Incorrect blocked | Clarified correctly | False rejections | Accepted wrong |
|---|---|---|---|---|---|---|
| Frozen F012a Qwen3 replies (28) | 17 | 4 | 7 | — | 0 | **0** |
| F012a scripted trap replies (4) | — | — | 4 | — | 0 | **0** |
| Author-written adversarial items: 12 items, 25 scripted proposals | 4 | 1 | 11 | 5 | 2 | **0** |

In the adversarial set, 2 further proposals were extractor abstentions on items that have a gold set, so
they were never accepted.

- **Blocked:** N1, N2, N3 and N4 (clarification) and the Q1 extra goal (unsupported).
- **False rejections:** "put **it**" (X5) and "**where** the blue cube is" (X6).
- **Out-of-grammar paraphrases** (an ad-hoc probe of 6 instructions; output not saved) were either refused
  with a clarification request, e.g. "Could you…", "red block", "then", "the blue one", or grounded to the
  correct goal ("in the target.", "red cube in target region."). None was grounded wrongly.

### F014 — Evidence-grounded extraction (evaluation only)
**Sources:** `f014_evidence.py`, `f014_items.json`, `traces/f014_dryrun_v2/` (150/150 gates),
`traces/f014_extract_v1/` (66 isolated calls: 60 items plus 6 repeats).
- **Isolation:** journal shows 66 requests, 0 foreign requests, 0 truncations.
- **Determinism repeats:** 3/3 identical.
- **Freezing:** the verifier's hash was unchanged from inference through scoring.

**Method.** Each goal cites `object_evidence` and `region_evidence` copied verbatim from the instruction. A
deterministic analysis of the instruction finds object and region mentions. It resolves them against the
state, including a few contextual forms ("it" with exactly one prior cube, "the other cube", "the blue one",
"both cubes", "the cube" when only one exists), and attaches destinations to objects. It fails closed with a
stated reason when it cannot. A reply is accepted only if its evidence supports the licensed bindings and its
goal set equals them exactly.

Three arms, scored on each item's first run:
- **A:** the F013 grammar.
- **B:** Qwen3 plus the F014 verifier.
- **C:** the F014 analysis alone, with no LLM.

| | A: F013 | B: Qwen3 + F014 | C: F014 analysis only |
|---|---|---|---|
| Gold-set items correctly accepted: F012a (18) / F013 adversarial (7) / F014dev (17) | 18 / 5 / **1** | 14 / 5 / **14** | 16 / 5 / **14** |
| Wrong sets accepted (60 items) | 0 | 0 | 0 |
| Should-abstain items blocked (18) | 18 | 18 | 18 |

- **Model behaviour:** Qwen3 abstained on 0/18 should-abstain items. Its raw extraction was exactly correct on
  38/42 gold-set items.
- **Evidence:** all 92 cited spans occurred verbatim in the instruction. 83 object spans and 81 region spans
  referred to the claimed id; 67/92 supported the claimed binding.
- **Contradictory requests:** extracted and accepted, then rejected as infeasible, in 5/5 cases (C1, C2, X9,
  E17, E18).
- **Contextual references accepted:**
  - "put **it**": X5, E11, and E12 (antecedent in the previous sentence);
  - "the **other cube**" (E14);
  - "the blue **one**" (E16);
  - "**both cubes**" (E17);
  - "the cube" with one cube in the scene (E21, X10).
- **Correctly refused:** "it" after two cubes (E13), and "the other cube" with no antecedent (E15, X7).
- **Falsely refused:** "where" (X6) and a relative clause (E24).

## 4. Failure examples and what they show

| Example | Observed | What it shows |
|---|---|---|
| F011c S2, turn 3 | After the rejected place into the occupied target, red was put into **parking**, blue's goal region, instead of moving blue | Recovery from a rejection did not resolve the dependency |
| Plan-only vs goal-order reversal | Same state and task; only the goal order differs: invalid 3/3 vs optimal 3/3 | In this configuration the plan follows presentation order, not occupancy |
| Repair loop | The same invalid plan for all 3 candidates in each of 3 repeats | Validator feedback did not change the plan |
| F012a N1 | "green cube" (no such object) became valid, feasible goals for real cubes | Structural validation and feasibility cannot catch unrequested goals |
| F012a Q1 | Partial instruction plus an invented `red→parking` | Over-extraction happens even on simple input |
| F014 N1, N2 | Cited "the green cube" for `cube_blue` and "storage region" for `parking`; both spans exist verbatim | Evidence that is real text is not evidence of authorisation |
| F014 E25 | "to the target region **and then** to the parking region": both goals had valid per-goal evidence; only the sequencing blocker stopped acceptance | Per-goal evidence checks are not enough on their own |
| F014 P3r/P3b/X2/E04 | "should end up in" and "needs to go into" were rejected as unclear structure | A frozen verifier defect (modal verbs), found after inference and deliberately not fixed afterwards |

## 5. Observed results vs interpretation

**Observed:**
- The trajectories, plans, decisions and counts above.
- Identical outputs for repeated temperature-zero calls of fixed inputs, on one Ollama build.
- Zero accepted wrong sets on the stated item sets.

**Interpretation:**
- Goal order drives Qwen3's action order here. This is supported by a single controlled reversal on one
  layout. It is not a general claim about the model's reasoning, which was disabled and is not observable.
- The F012 gains come from moving ordering out of the LLM into a verified planner.
- The F014 gains come from the deterministic analysis. Arm C matches arm B on F014dev and beats it on F012a,
  and a complete deterministic verifier caps what B can accept.

**Not established:**
- success rates or population statistics;
- robustness to other layouts, wordings, models or builds;
- reliability on real robots;
- held-out generalisation of any language result.

## 6. Limitations and open engineering issues

- **Consistency, not determinism.** Repeated temperature-zero outputs show observed consistency. Ollama
  tool-call ids differ from call to call, so requests differ after turn 0.
- **The comparison is not like-for-like.** PlanExecutor receives trusted gold goals, while reactive Qwen3 had
  to interpret language.
- **Mock only.** Planning results do not establish real-world reliability. The Robosuite backend supports one
  cube and one region; its reach and barrier are not modelled as planner bounds.
- **Small development sets.** F012a, F013 and F014 are small, and the F013/F014 items and gold labels were
  written by the same author as the checks. F014's labels were frozen before inference but were not
  independently annotated.
- **Verifier limits.**
  - Evidence spans alone cannot prove semantic authorisation.
  - F014's verifier largely decides what can be accepted.
  - The modal-verb defect is open.
- **Executor and fault-injection gaps:**
  - Fault rejections are not transactional: a refused fault has already advanced the mock's step and clock.
    `InvalidFault` aborts the episode with no `episode_end` event.
  - The executor's missing/stale-feedback paths and its place/finish prechecks can't be reached through
    `AgentLoop` on the mock; they are tested with synthetic messages.
  - A stop issued on the final allowed turn is reported as `max_steps`.
- **Still-open backlog items:**
  - B004: distinguish harness/policy validation failures from environment refusals;
  - B005: Robosuite failure paths not covered through `AgentLoop`;
  - F009: memory, still partial.

## 7. Future research directions (not selected)

1. **Language grounding:** LLM-assisted goal interpretation with independent confirmation, such as user
   clarification or confirmation, instead of lexical verification alone. Evaluate on an independently
   annotated held-out set.
2. **Model learning:** stronger reasoning models, or post-training for planning and grounding, measured
   against the F011 and F012 baselines.
3. **Embodied execution:** more realistic multi-object manipulation (Robosuite with several objects,
   backend-aware bounds) and sim-to-real evaluation of the planner and executor.
