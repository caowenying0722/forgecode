# ForgeCode runtime boundaries

`Conversation` is the composition root and public compatibility facade. New
behavior should be placed in the narrowest owner below instead of adding a new
branch directly to `agent_loop.py`.

| Module | Owns | Must not own |
| --- | --- | --- |
| `agent_loop` | phase transitions and orchestration order | provider parsing, tool serialization, recovery text |
| `agent_state` / `state` | turn phases, value objects, emitted events | execution policy or side effects |
| `model_client` | Anthropic provider transport and stream parsing | turn-level usage or completion policy |
| `model_runner` | one model run and cumulative usage | retries requiring workspace/context decisions |
| `model_failure` | classify model exceptions into recovery actions | conversation or context mutation |
| `tool_executor` | hooks, permissions, logging and registry execution | Agent phase or repetition policy |
| `tool_runner` | Agent-level tool guards and one batch's observations | filesystem/process implementations |
| `tool_targets` | canonical mutation target extraction | recovery decisions |
| `completion` | deterministic workspace policy evaluation | model-visible feedback |
| `completion_checker` | Agent finish declarations and completion feedback | tool execution |
| `recovery_manager` | recovery-phase tool selection | recovery messages or counters |
| `recovery_feedback` | action/edit/stagnation feedback | tool selection |
| `protocol_recovery` | malformed model/tool protocol feedback | task completion policy |
| `request_builder` | request tool visibility and system context | model transport |
| `agent_messages` | model-visible message serialization | verification or recovery policy |
| `process` | subprocess execution value objects and helpers | shell-tool validation |
| `background` | background process lifecycle and notifications | subprocess implementation |
| `session_manager` | runtime session save/load/history facade | session file format implementation |
| `workspace` | task-local workspace snapshots and revisions | completion decisions |
| `team` | durable team messages and request lifecycle | task graph ownership |

Dependency direction should generally be:

```text
agent_loop -> runners/managers -> contracts and primitives
tools      -> runtime primitives (process, workspace, state)
```

Runtime primitives must not import concrete tool implementations. The
composition root may import tools to wire the application together.
