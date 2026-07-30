# ForgeCode Plans

## Next Milestone: Runtime State And Recovery Cleanup

- Remove scalar compatibility recovery fields from `RequestState`.
- Move the remaining local counters in `agent_loop.py` into `TurnRuntimeState`.
- Replace action/mutation recovery `read_available` booleans with a unified `RecoveryScope`.
- Add expected file revision or content hash checks to edit operations.
- Further split the large `agent_loop.py` orchestration module.
