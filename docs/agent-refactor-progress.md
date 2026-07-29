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

Status: pending.

Goal:
- Move remaining duplicated recovery booleans and counters out of the main loop
  into explicit control state or narrow state objects.
