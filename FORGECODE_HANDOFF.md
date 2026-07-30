# ForgeCode Handoff

## Current Acceptance Evidence Slice

- Completion checking no longer contains product, framework, fixture, or game-feature-specific acceptance regexes.
- `AcceptanceEvidence` and `AcceptanceLedger` track each `TaskContract.acceptance_criteria` item by stable `criterion-N` IDs.
- Evidence is append-only and typed: source changes, tests, typecheck, build, lint, smoke, symbol evidence, runtime integration, configuration, review, and manual limitations.
- `task_plan` can associate steps with deliverables and criterion IDs. `task_update` can attach structured evidence and only counts as progress when that evidence is valid.
- `ProgressEvaluator` now receives completed criteria, completed plan steps, verification error counts, failure signature changes, verification reuse, source changes, and repair-target resolution.
- `finish_task` gap reports are generated from the task contract, plan state, acceptance ledger, verification state, and workspace change classification.
- Acceptance ledger updates are persisted as `acceptance_ledger_updated` rollout events with full ledger snapshots so sessions can replay satisfied, partial, and pending criteria.

## Runtime State And Recovery Cleanup

The current `PLANS.md` milestone is complete:

- `RequestState` recovery/synthesis compatibility scalars were removed; recovery request construction now reads those fields from `TurnRuntimeState`.
- Action, mutation, and verification recovery tool selection use `RecoveryScope` instead of bare `read_available` booleans.
- Remaining Agent Loop counters moved into `TurnRuntimeState` sub-states for loop progress, completion gate state, synthesis retries, token-limit recovery, and model-failure recovery.
- Edit tools now support stale-write guards: `write_file` and `replace_text` accept `expected_sha256`, `write_file_chunk` accepts `expected_current_sha256`, and `apply_patch` accepts `expected_sha256_by_path`. `read_file` returns whole-file `sha256` metadata for this flow.
- Tool-call signature normalization and early mutation relevance checks moved from `agent_loop.py` to `agent_tool_calls.py`.
