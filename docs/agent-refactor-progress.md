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

Status: pending.

Goal:
- Generate targeted repair guidance from verification diagnostics and mutation
  failures, including likely files, line numbers, symbols, expected action, and
  failure signature.

## Milestone 3 - Relevant Progress Evaluation

Status: pending.

Goal:
- Count progress only when evidence, verification, plan updates, or diffs are
  relevant to the Task Envelope and the current blocker.

## Milestone 4 - Earlier Completion Relevance Guards

Status: pending.

Goal:
- Prevent unrelated workspace modifications before late completion checks.

## Milestone 5 - Context Replay After Compaction

Status: pending.

Goal:
- Ensure source content that was read before context compaction can be replayed
  or reacquired, rather than reduced to unrecoverable short references.

## Milestone 6 - Loop State Consolidation

Status: pending.

Goal:
- Move remaining duplicated recovery booleans and counters out of the main loop
  into explicit control state or narrow state objects.
