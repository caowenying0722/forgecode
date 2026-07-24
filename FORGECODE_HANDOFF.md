# ForgeCode Handoff Summary

This document summarizes the recent ForgeCode design and implementation work so a new conversation can resume without re-reading the full chat history.

## Current Goal

ForgeCode is being evolved toward a Claude Code / Codex style terminal coding agent with:

- dynamic context assembly and compaction
- session resume
- interaction modes
- unified tool execution
- permission hooks
- MCP tools
- task/subagent delegation
- durable memory
- safety recovery instead of raw internal errors

## Implemented Features

### Interaction Modes

ForgeCode now has three modes:

- `auto`: default; infers whether a workspace diff is required.
- `plan`: read-only planning mode; does not require a workspace diff.
- `code`: execution mode; requires real task-local workspace changes when the user authorized edits.

`/edit` was removed in favor of keeping `/code` as the execution mode.

Plan mode allows read/context tools such as:

- `list_directory`
- `find_files`
- `read_file`
- `grep`
- `git_status`
- `memory_list`
- `memory_read`
- `task`
- `explore_subagent`
- `todo_write`

### Context Management

System prompt is not injected once into history. It is dynamically assembled for every model request.

Each request can include:

- base prompt from `forge/prompts/system.md`
- active task context
- working-state evidence
- tool availability context
- interaction mode context
- repository instructions and relevant memory
- recovery-specific context

Repository context includes:

- `~/.forge/AGENTS.md`
- `AGENTS.md` / `AGENTS.override.md` from repo root to current cwd
- root `FORGE.md`
- `.forge/rules/*.md`
- query-relevant `.forge/memory/*.md`

Context overflow protection:

- cheap compaction on request copies
- large tool results saved under `.forge/context/tool-results/`
- old tool results shortened
- middle messages cropped
- full structured compaction at 80% of configured context window
- fallback threshold of 120,000 history characters if no model window is configured
- provider `context_overflow` triggers forced compaction and one retry

Manual compaction is available via `/compact`.

### Session Resume

ForgeCode has session save/resume support:

- saves messages
- saves active task
- saves interaction mode
- saves permission mode

Useful commands:

- `/resume`
- `/resume <session-id>`
- `/sessions`

### Completion And Recovery

The completion gate distinguishes:

- answer
- inspection
- change

It uses task-local workspace tracking so pre-existing user diffs are not treated as work completed by the agent.

Several raw stuck behaviors were improved:

- stagnation limit raised from 8 to 16
- stagnation can enter final recovery instead of immediately stopping
- finalization recovery can summarize already changed/verified work
- cumulative input-token safety limit enters token-limit recovery instead of returning only a raw threshold error

Current per-turn token safety behavior:

- default `max_turn_input_tokens = 500_000`
- this is per user turn, not whole session
- when reached, ForgeCode makes one no-tool recovery request
- the final response summarizes progress, remaining work, verification, and next step
- turn status remains `stuck` because a safety threshold stopped the turn

### ToolExecutor And Hooks

All tool calls now go through a centralized `ToolExecutor`.

Cross-cutting behavior is handled by hooks instead of being scattered inside the loop:

- `PermissionHook`
- `ToolLoggingHook`
- `TodoPlanningHook`

Hook events include:

- `user_prompt_submit`
- `pre_tool_use`
- `permission_denied`
- `post_tool_use`
- `stop`

Tool logs are written to:

```text
.forge/logs/tools.jsonl
```

### Permissions

Permission modes:

- `trusted`: allow read/write/process tools
- `strict`: ask before workspace writes and process execution
- `readonly`: only allow read-only tools

Commands:

- `/permission`
- `/permission trusted`
- `/permission strict`
- `/permission readonly`

Strict mode uses an interactive confirmation flow. Denials are treated as permission outcomes, not edit failures.

### Todo Planning

`todo_write` was added as a planning tool.

For complex tasks, `TodoPlanningHook` can require a short plan before write/process tools.

Important fix:

- `todo_required` is classified as a protocol/planning gate failure.
- It no longer counts as a workspace-write failure, avoiding misleading edit recovery loops.

### MCP

ForgeCode has real MCP support:

- config file: `.forge/mcp.json`
- stdio JSON-RPC with `Content-Length` framing
- supports `initialize`, `tools/list`, `tools/call`
- remote tools are exposed as `mcp_{server}_{tool}`
- MCP tools are converted into normal `ToolResult`
- MCP tools go through normal tool execution, permission and logging

Example server exists:

```text
examples/mcp_web_fetch_server.py
```

Command:

```text
/mcp
```

### Subagents

ForgeCode now has a minimal supervised worker subagent model.

Main tools:

- `task`: preferred delegation tool
- `explore_subagent`: compatibility alias

Subagent behavior:

- clean isolated model context
- bounded rounds
- returns a structured report to the main agent via `ToolResult.content`
- does not directly own the main task outcome

Subagent tool policy:

It has normal repository tools, including:

- read/search/list tools
- write tools
- patch tools
- command execution
- verify
- git tools
- MCP tools
- memory tools

It does not have recursive/control tools:

- `task`
- `explore_subagent`
- `task_get`
- `task_plan`
- `task_update`
- `todo_write`
- `finish_task`

Subagent calls still go through:

- `ToolExecutor`
- shared `PermissionHook`
- tool logging
- shared `WorkspaceTracker`

This means subagent writes are visible to the main agent completion gate.

### Memory System

ForgeCode memory is repository-scoped durable memory.

Storage:

```text
.forge/memory/*.md
.forge/memory/MEMORY.md
```

Commands:

- `/remember name | content`
- `/memory list`
- `/memory show name`
- `/memory forget name`
- `/memory rebuild`
- `/memory consolidate`

Model tools:

- `memory_list` read-only
- `memory_read` read-only
- `memory_write` workspace-write
- `memory_update` workspace-write
- `memory_delete` workspace-write

Memory writes, updates, and deletes go through:

- `ToolExecutor`
- `PermissionHook`
- `ToolLoggingHook`

Frontmatter now includes:

- `name`
- `description`
- `type`
- `source`
- `created_at`
- `updated_at`

Source values currently used:

- `/remember`: `manual`
- explicit natural language `记住：...` / `remember: ...`: `explicit_user_prompt`
- model memory tool: `model_memory_tool`
- old memory files without metadata: `legacy`

The system currently does not do Codex-style automatic memory candidate extraction. The user specifically said automatic candidate prompts did not feel necessary.

Memory retrieval:

- deterministic keyword matching
- max 5 records
- max 4 KB per record
- max 20 KB total
- only relevant memory is injected into the request

Secrets are rejected with a simple pattern for API keys, tokens, passwords and private keys.

## Important Design Decisions

### System Prompt Strategy

ForgeCode uses dynamic request assembly, not one-time history injection.

The fixed base prompt remains in `forge/prompts/system.md`, but task state, repository rules, memory, working evidence and recovery instructions are rebuilt per model call.

Known limitation:

- repository memory/rules are selected once at the beginning of a user turn based on the user prompt
- later model calls in the same turn reuse that repository context
- future improvement could re-query memory/rules based on evolving tool evidence

### Token Limit Strategy

The 500,000 input-token limit is a per-turn safety fuse.

It is not:

- a single request context limit
- a whole-session limit

It is:

- cumulative input tokens across all model calls in one user turn

When hit, ForgeCode now enters token-limit recovery.

### Memory Strategy

Current memory is explicit or tool-driven.

It is not fully automatic:

- `/remember` writes memory
- `记住：...` writes memory
- model can call memory tools
- no background automatic extraction from completed conversations

The latest request intentionally implemented metadata only, not automatic candidates.

## Verification State

Recent full test runs passed after the major changes:

```text
331 passed
ForgeCode 0.1.0
```

If resuming after a fresh checkout or new changes, rerun:

```powershell
.\.venv\Scripts\python.exe -m pytest --basetemp=.pytest_tmp_verify
.\.venv\Scripts\forge.exe --version
```

## Current Workspace Note

At the time this handoff was written, `git status --short --branch` showed:

```text
## main...target/main
?? game/
```

The `game/` directory was not touched by this handoff. Treat it as user or unrelated workspace state unless the user explicitly asks to modify it.

## Remaining Gaps

High priority:

- Memory write/update/delete probably needs a memory-specific confirmation policy. Even in `trusted`, long-term memory mutation can be more sensitive than normal file edits.
- `/context` or another diagnostic command should show exactly which memory records were injected for the current request.
- Memory records have metadata, but no `last_used_at` or selection audit yet.
- ToolExecutor and hook architecture exists, but auto-commit remains intentionally not enabled.

Medium priority:

- Memory retrieval is keyword-based; BM25 or better ranking would be stronger without needing embeddings.
- Repository context is selected once per user turn rather than per intermediate model call.
- Memory consolidation only removes exact duplicates. It does not merge semantically overlapping or conflicting records.
- No global `~/.forge/memory` store yet; only global `~/.forge/AGENTS.md` instructions.

Lower priority:

- Better UI for permission prompts and memory diffs.
- Better docs for subagent worker semantics.
- Benchmarks comparing token use with and without subagents/memory/context compaction.

## Suggested Next Work

Best next implementation tasks:

1. Add a memory-specific permission mode or hook option so `memory_write/update/delete` default to ask.
2. Add injected-memory diagnostics to `/context`.
3. Add `last_used_at` and selection tracking for memory records.
4. Improve memory ranking from simple keyword scoring to BM25-style scoring.
5. Re-run full tests and push once the user asks.
