# ForgeCode Agent Refactor Progress

This document tracks the staged refactor from a rule/tool-driven loop toward a
user-goal-driven coding agent. Each milestone is committed and pushed after its
required checks pass.

## Milestone 1 - Task Envelope and Control-State Tool Surface

Status: complete.

Completed changes:
- Expanded `TaskContract` into a backward-compatible Task Envelope with kind,
  goal, deliverables, acceptance criteria, context hints, allowed and forbidden
  paths, verification policy, model budget, tool budget, and confidence.
- Preserved deterministic routing for obvious requests and added an optional
  semantic classifier hook that is used only for low-confidence contracts.
- Made `AgentControlState` the source of truth for planning recovery state;
  `AgentPhase` remains UI-facing only.
- Updated `RequestBuilder` so tool selection is driven first by
  `AgentControlState` and the Task Envelope, with legacy booleans kept only as
  compatibility inputs during this migration.
- Replaced the broad `TodoPlanningHook` keyword rules with the shared Task
  Envelope plan requirement.
- Added regression tests for knowledge answers, file analysis, negated edit
  requests, single-file fixes, multi-module planning, planning completion,
  verification failure state, and contradictory tool-surface booleans.

Validation:
- `uv lock --check`
- `uv run python -m compileall -q forge tests`
- `uv run pytest -q`
- `git diff --check`

Remaining duplicated state after this milestone:
- `action_recovery`
- `verification_recovery`
- `verification_fix_recovery`
- `verification_fix_required`
- `mutation_recovery_read_used`
- `force_synthesis`
- `finalization_recovery`
- `stagnation_final_recovery`
- `token_limit_recovery`

## Milestone 2 - Repair Target Recovery

Status: complete.

Completed changes:
- Generate targeted repair guidance from verification diagnostics and mutation
  failures, including likely files, line numbers, symbols, expected action, and
  failure signature.
- Added `RepairTarget` in `RecoveryManager` as a focused recovery data object
  instead of scattering ad hoc diagnostic text through the loop.
- Added extraction from failed mutation tools and failed verification output.
- Rendered repair targets into mutation recovery and verification recovery
  prompts so the next model call starts from the relevant file, line, symbol,
  and expected repair action before broad discovery.
- Preserved existing tool surfaces, read quotas, repeated verification guards,
  MCP, permission, session, sub-agent, and trajectory behavior.

Validation:
- `uv lock --check`
- `uv run python -m compileall -q forge tests`
- `uv run pytest -q`
- `git diff --check`

## Milestone 3 - Relevant Progress Evaluation

Status: complete.

Completed changes:
- Count progress only when evidence, verification, plan updates, or diffs are
  relevant to the Task Envelope and the current blocker.
- Extended `ProgressEvaluator` with optional task scope, changed paths, new
  evidence paths, diff review paths, repair target paths, and change-required
  context while keeping existing callers compatible.
- Unrelated workspace revisions, repository evidence, and diff reviews no
  longer reset stagnation progress when a task or repair target has a
  constrained scope.
- Repair-target evidence is accepted as relevant progress for verification
  recovery.
- Added parameterized regression coverage for unrelated workspace changes,
  unrelated evidence, and repair-target evidence.

Validation:
- `uv lock --check`
- `uv run python -m compileall -q forge tests`
- `uv run pytest -q`
- `git diff --check`

## Milestone 4 - Earlier Completion Relevance Guards

Status: complete.

Completed changes:
- Prevent unrelated workspace modifications before late completion checks.
- Added an early mutation relevance guard before tool execution for statically
  targetable workspace-write tools.
- Reused existing task-scope relevance checks instead of creating a parallel
  policy path.
- Blocked clearly off-scope edits with `irrelevant_mutation_target`, feeding
  the result into existing mutation recovery instead of allowing unrelated
  files to be changed and rejected only at completion.
- Allowed scoped directory setup such as creating `game` when the task scope is
  `game/**`.
- Added unit and loop regression coverage proving off-scope writes are blocked
  before execution and relevant writes still complete.

Validation:
- `uv lock --check`
- `uv run python -m compileall -q forge tests`
- `uv run pytest -q`
- `git diff --check`

## Milestone 5 - Context Replay After Compaction

Status: complete.

Completed changes:
- Ensure source content that was read before context compaction can be replayed
  or reacquired, rather than reduced to unrecoverable short references.
- Changed `WorkingState` read preflight from a short "already covered" marker
  to an actual cached source replay for covered `read_file` ranges.
- Preserved cache metadata so replayed reads remain visible as cache hits and
  do not execute redundant filesystem reads.
- Updated working-evidence system context to state that covered read ranges are
  replayable after conversation compaction.
- Added regression coverage for exact, subset, overlapping, and
  compaction-reference read replay.

Validation:
- `uv lock --check`
- `uv run python -m compileall -q forge tests`
- `uv run pytest -q`
- `git diff --check`

## Milestone 6 - Loop State Consolidation

Status: complete for this refactor pass.

Completed changes:
- Move remaining duplicated recovery booleans and counters out of the main loop
  into explicit control state or narrow state objects.
- Removed the main loop's local `action_recovery` boolean. Action Recovery is
  now derived from `AgentControlState.TARGETED_ANALYSIS` through
  `AgentController.action_recovery`.
- Kept `action_recovery_calls` and `action_read_used` as budget/accounting
  fields rather than flow-state facts.
- Added controller regression coverage proving Action Recovery is derived from
  control state.

Validation:
- `uv lock --check`
- `uv run python -m compileall -q forge tests`
- `uv run pytest -q`
- `git diff --check`

Remaining state debt:
- `verification_recovery`, `verification_fix_recovery`, and
  `verification_fix_required` should become a verification recovery state
  object keyed by revision and failure signature.
- `mutation_recovery_read_used` and `mutation_recovery_context` should move
  into an edit recovery state object that owns repair targets and read budgets.
- `force_synthesis`, `finalization_recovery`, `stagnation_final_recovery`, and
  `token_limit_recovery` should become a synthesis/finalization mode instead
  of separate booleans.

## Milestone 7 - Unified State Control and Goal-Driven Execution

Status: complete for this vertical slice.

Completed changes:
- Added controller-owned `TurnRuntimeState` and `BudgetLedger` snapshots so
  request construction can consume one state object instead of independent
  recovery booleans.
- Changed planning-required tasks to start in `PLANNING`, exposing read-only
  planning tools and `todo_write` before any write tool can be attempted.
- Changed `TodoPlanningHook` so it executes the controller's `todo_required`
  decision instead of independently classifying task complexity.
- Added a production `ModelSemanticTaskClassifier` path for low-confidence
  Task Envelopes while preserving deterministic fast paths and fake-test
  behavior.
- Made `RequestBuilder` prefer `TurnRuntimeState` over legacy `RequestState`
  booleans for planning, action recovery, verification recovery, finalization,
  mutation recovery, and tool-free synthesis.
- Reworked verification recovery around `RepairTarget` diagnostics, including
  TS2305 missing exports, modules, direct dependencies, and target-sized read
  budgets instead of a fixed one-read rule.
- Prevented repeated verify before a relevant RepairTarget mutation and kept
  unrelated changes from reopening verification.
- Tightened progress and completion relevance so temporary files, generated
  `src/**/*.js` outputs, and mixed unrelated changes do not count as source
  progress or valid completion.
- Bound budget accounting for model calls and tool calls to the controller
  runtime snapshot.
- Preserved read evidence replay after context compaction.

Validation:
- `uv lock --check`
- `uv run python -m compileall -q forge tests`
- `uv run pytest -q`
- `git diff --check`

Regression coverage added or updated:
- Complex planning tasks do not get `write_file`/write tools on the first
  request.
- Single-file fixes are not forced into todo planning.
- TS2305 diagnostics build RepairTargets and allow importer/exporter reads.
- Repeated verify is blocked after failed verification until relevant repair.
- Unrelated changes after failed verification do not enable verify.
- New development tasks without a concrete path do not enter Action Recovery.
- Mixed relevant and temporary-file changes are rejected by completion.
- Tool/model budget exhaustion stops the trajectory.
- Runtime snapshots override contradictory legacy `RequestState` booleans.

Duplicate control logic removed or deprecated in this slice:
- `TodoPlanningHook` no longer owns task complexity classification.
- RequestBuilder no longer treats legacy recovery booleans as authoritative
  when a `TurnRuntimeState` is present.
- Verification read gating is now target-budget based; the old boolean
  `verification_read_used` remains only for compatibility callers.
- Tool execution now enforces the current state-selected tool set as a hard
  boundary while preserving `unknown_tool` behavior for registry misses.

Remaining old state sources:
- `agent_loop.py` still carries local migration variables for
  `verification_recovery`, `verification_fix_recovery`,
  `verification_fix_required`, `finalization_recovery`,
  `stagnation_final_recovery`, `token_limit_recovery`, `force_synthesis`,
  mutation recovery counters, and action recovery counters.
- `RequestState` still exposes legacy boolean fields for compatibility tests
  and callers that have not yet been moved to `TurnRuntimeState`.
- `ToolRunPolicy` still transports phase booleans at the execution boundary;
  these should become a single control-state value plus read-budget decisions.

Next agent_loop.py migration areas:
- Replace local verification booleans with a revision-bound verification state
  object owned by `TurnRuntimeState`.
- Move finalization/stagnation/token-limit synthesis into a single controller
  mode and remove the local booleans.
- Move mutation recovery context, failures, and read counters into a dedicated
  controller-owned edit recovery state.
- Route progress acceptance criteria and plan-step completion directly from
  Task Envelope state instead of passing booleans into `evaluate_progress`.

## Milestone 8 - agent_loop.py State Source Migration

Status: complete for this vertical slice.

Completed changes:
- Removed the remaining local `agent_loop.py` flow-state variables for
  verification recovery, edit recovery, action recovery counters, and
  synthesis/finalization recovery.
- Added controller-owned nested runtime state objects:
  `VerificationRecoveryState`, `EditRecoveryState`,
  `ActionRecoveryRuntimeState`, and `SynthesisRuntimeState`.
- Replaced synthesis booleans with `SynthesisMode` enum values for checkpoint,
  finalization, stagnation-final, and token-limit paths.
- Changed the main request construction path so `agent_loop.py` passes one
  `TurnRuntimeState` snapshot instead of copying recovery booleans into
  `RequestState`.
- Changed RequestBuilder internals to read nested runtime state directly when
  a runtime snapshot is present; scalar `RequestState` fields now act only as
  legacy fallback inputs.
- Changed ToolRunner transaction phase detection to prefer `TurnRuntimeState`
  and `AgentControlState`, and removed duplicated per-recovery tool-availability
  gates.
- Added regression coverage proving runtime snapshots override contradictory
  legacy `RequestState` and `ToolRunPolicy` booleans.

Validation:
- `uv lock --check`
- `uv run python -m compileall -q forge tests`
- `uv run pytest -q`
- `git diff --check`

Duplicate control logic removed or deprecated in this slice:
- Removed `agent_loop.py` locals:
  `verification_recovery`, `verification_fix_recovery`,
  `verification_fix_required`, `verification_failed_revision`,
  `verification_read_count`, `verification_recovery_calls`,
  `last_verification_failure_signature`, `mutation_recovery_context`,
  `mutation_failures`, `mutation_recovery_read_used`,
  `mutation_failure_count`, `action_recovery_calls`, `action_read_used`,
  `action_block_events`, `force_synthesis`, `finalization_recovery`,
  `stagnation_final_recovery`, and `token_limit_recovery`.
- Removed the request-time copy-back from local recovery variables into
  `TurnRuntimeState`; the controller snapshot is now the live state object.
- Removed duplicate ToolRunner checks that reinterpreted planning/action/edit/
  verification booleans after the state-selected tool set had already been
  enforced.

Remaining old state sources:
- `RequestState` still exposes scalar recovery fields for compatibility with
  older tests and non-runtime callers; runtime snapshots override them.
- `ToolRunPolicy` still exposes legacy recovery booleans for compatibility
  callers; runtime/control-state values override them.
- `TurnRuntimeState` still exposes compatibility properties such as
  `latest_verification`, `repair_target`, `mutation_failures`,
  `action_recovery_calls`, and `synthesis_mode`; production code should prefer
  the nested state objects.
- `AgentController.planning_recovery` and `AgentController.action_recovery`
  remain compatibility views over `AgentControlState`.

Next agent_loop.py migration areas:
- Encapsulate repeated recovery reset blocks behind controller/runtime methods
  so `agent_loop.py` mutates less nested state directly.
- Move completion block counters and current acceptance-step tracking into the
  controller runtime snapshot.
- Retire `RequestState` scalar recovery fields after downstream tests/callers
  construct runtime snapshots.
- Retire `ToolRunPolicy` scalar recovery fields after all direct tests use
  runtime/control-state inputs.
