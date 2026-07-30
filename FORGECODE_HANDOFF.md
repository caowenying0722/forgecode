# ForgeCode Handoff

## Current Acceptance Evidence Slice

- Completion checking no longer contains product, framework, fixture, or game-feature-specific acceptance regexes.
- `AcceptanceEvidence` and `AcceptanceLedger` track each `TaskContract.acceptance_criteria` item by stable `criterion-N` IDs.
- Evidence is append-only and typed: source changes, tests, typecheck, build, lint, smoke, symbol evidence, runtime integration, configuration, review, and manual limitations.
- `task_plan` can associate steps with deliverables and criterion IDs. `task_update` can attach structured evidence and only counts as progress when that evidence is valid.
- `ProgressEvaluator` now receives completed criteria, completed plan steps, verification error counts, failure signature changes, verification reuse, source changes, and repair-target resolution.
- `finish_task` gap reports are generated from the task contract, plan state, acceptance ledger, verification state, and workspace change classification.
- Acceptance ledger updates are persisted as `acceptance_ledger_updated` rollout events with full ledger snapshots so sessions can replay satisfied, partial, and pending criteria.

## Remaining Migration Work

See `PLANS.md` for the next milestone covering `RequestState` compatibility fields, `TurnRuntimeState` counter migration, unified recovery scopes, edit revision checks, and further `agent_loop.py` decomposition.
