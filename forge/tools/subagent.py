'''Bounded supervised subagents exposed as ForgeCode tools.'''

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from pydantic import Field

from forge.tools.base import Tool, ToolInput, ToolRegistry, ToolResult
from forge.tools.filesystem import ListDirectoryTool, ReadFileTool
from forge.tools.git import GitStatusTool
from forge.tools.search import FindFilesTool, GrepTool

if TYPE_CHECKING:
    from forge.hooks.builtin import PermissionHook
    from forge.hooks.registry import HookRegistry
    from forge.runtime.team import MessageBus
    from forge.runtime.workspace import WorkspaceTracker
    from forge.runtime.worktree import SubagentWorktreeManager


SUBAGENT_EXCLUDED_TOOLS = frozenset(
    {
        'task',
        'task_create',
        'task_claim',
        'task_get',
        'task_plan',
        'task_update',
        'todo_write',
        'finish_task',
    }
)


SUBAGENT_SYSTEM = '''You are a ForgeCode Task Subagent.
You perform bounded repository work for the main agent in an isolated Git
worktree and context. Other agents cannot see your unintegrated file edits.
Use the provided tools to inspect, edit, run commands, and verify when needed.
You cannot spawn recursive subagents or manage the main task plan. Do not claim
completion of the user's whole task. Return a concise structured report with:
- relevant_files
- evidence
- changes_made
- verification
- remaining_risks
- open_questions
Ground every claim in observed repository evidence.'''


class SubagentModelClient(Protocol):
    def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
    ) -> Any:
        ...


class TaskSubagentInput(ToolInput):
    task: str = Field(min_length=1, max_length=2_000)
    focus_paths: list[str] = Field(default_factory=list, max_length=10)
    max_rounds: int = Field(default=4, ge=1, le=6)


class TaskSubagentTool(Tool[TaskSubagentInput]):
    name = 'task'
    effect = 'process'
    description = (
        'Delegate bounded repository work to a subagent with an isolated Git '
        'worktree and context. Use this to locate relevant files, gather '
        'evidence, and '
        'make scoped edits when isolation or parallel investigation helps. '
        'Do not use for simple local reads or small focused edits. The '
        'subagent cannot spawn recursive agents or manage task-plan tools; '
        'its calls still pass through permissions, hooks, and logging. '
        'Changes are integrated only when the lead files still match the '
        'subagent start baseline; conflicts preserve the worktree. It returns '
        'a structured report for the main agent.'
    )
    input_model = TaskSubagentInput

    def __init__(
        self,
        root: Path,
        *,
        client: SubagentModelClient | None = None,
        permission: 'PermissionHook | None' = None,
        workspace_tracker: 'WorkspaceTracker | None' = None,
        team_bus: 'MessageBus | None' = None,
        worktree_manager: 'SubagentWorktreeManager | None' = None,
    ) -> None:
        super().__init__(root)
        self.client = client
        self.permission = permission
        self.workspace_tracker = workspace_tracker
        self.team_bus = team_bus
        self.worktree_manager = worktree_manager

    async def execute(self, arguments: TaskSubagentInput) -> ToolResult:
        from forge.runtime.model_client import AnthropicModelClient
        from forge.runtime.worktree import (
            SubagentWorktreeManager,
            WorktreeError,
        )

        client = self.client or AnthropicModelClient.from_config()
        manager = self.worktree_manager or SubagentWorktreeManager(self.root)
        try:
            lease = manager.create(self.name)
        except WorktreeError as error:
            return ToolResult.fail(
                error.code,
                str(error),
                details={'isolation': 'worktree'},
            )
        subagent = TaskSubagent(
            lease.path,
            client,
            permission=self.permission,
            team_bus=self.team_bus,
            control_root=self.root,
            agent_id=lease.id,
        )
        result = await subagent.run(arguments)
        try:
            integration = manager.integrate(
                lease,
                apply=result.success,
            )
        except WorktreeError as error:
            return ToolResult.fail(
                error.code,
                str(error),
                content=result.content,
                details={
                    'isolation': 'worktree',
                    'worktree_path': str(lease.path),
                },
                metadata={**result.metadata, 'worktree_path': str(lease.path)},
            )
        integration_metadata = {
            'isolation': 'worktree',
            'worktree_id': lease.id,
            'base_head': lease.base_head,
            'changed_paths': list(integration.changed_paths),
            'integrated_paths': list(integration.integrated_paths),
            'conflicts': list(integration.conflicts),
            'worktree_path': integration.worktree_path,
            'worktree_cleaned_up': integration.cleaned_up,
        }
        if integration.conflicts:
            paths = ', '.join(integration.conflicts)
            return ToolResult.fail(
                'subagent_merge_conflict',
                'Subagent changes were not integrated because the lead '
                f'workspace also changed: {paths}. The isolated worktree '
                'was preserved for review.',
                content=result.content,
                details=integration_metadata,
                metadata={**result.metadata, **integration_metadata},
            )
        if not result.success:
            return ToolResult(
                success=False,
                summary=result.summary,
                content=result.content,
                error=result.error,
                metadata={**result.metadata, **integration_metadata},
            )
        return ToolResult.ok(
            result.summary,
            content=result.content,
            metadata={**result.metadata, **integration_metadata},
        )


class TaskSubagent:
    '''A small isolated model loop without recursive task/subagent tools.'''

    def __init__(
        self,
        root: Path,
        client: SubagentModelClient,
        *,
        permission: 'PermissionHook | None' = None,
        hooks: 'HookRegistry | None' = None,
        workspace_tracker: 'WorkspaceTracker | None' = None,
        team_bus: 'MessageBus | None' = None,
        control_root: Path | None = None,
        agent_id: str = 'task_subagent',
    ) -> None:
        from forge.hooks.builtin import PermissionHook, ToolLoggingHook
        from forge.hooks.registry import HookRegistry
        from forge.runtime.tool_executor import ToolExecutor

        self.root = root
        self.control_root = (control_root or root).resolve()
        self.agent_id = agent_id
        self.client = client
        self.registry = create_subagent_registry(
            root,
            workspace_tracker,
            team_bus=team_bus,
            agent_id=agent_id,
            control_root=self.control_root,
        )
        self.permission = permission or PermissionHook('trusted')
        self.logger = ToolLoggingHook(self.control_root, agent=agent_id)
        self.hooks = hooks or HookRegistry([self.permission, self.logger])
        self.executor = ToolExecutor(
            self.registry,
            root=root,
            workspace_tracker=workspace_tracker,
            permission=self.permission,
            logger=self.logger,
            hooks=self.hooks,
        )

    async def run(self, arguments: TaskSubagentInput) -> ToolResult:
        from forge.runtime.agent_messages import (
            build_assistant_message,
            build_tool_result_message,
        )
        from forge.runtime.model_runner import add_token_usage
        from forge.runtime.state import (
            ModelTextDelta,
            ModelToolCallCompleted,
            ModelUsageUpdate,
            TokenUsage,
        )

        messages: list[dict[str, Any]] = [
            {
                'role': 'user',
                'content': render_subagent_task(arguments),
            }
        ]
        total_usage = TokenUsage(input_tokens=0, output_tokens=0)
        tool_calls: list[str] = []
        final_text = ''

        for round_index in range(1, arguments.max_rounds + 1):
            text_parts: list[str] = []
            requested: list[Any] = []
            request_usage: TokenUsage | None = None
            async for event in self.client.stream(
                messages,
                tools=self.registry.definitions,
                system=SUBAGENT_SYSTEM,
            ):
                if isinstance(event, ModelTextDelta):
                    text_parts.append(event.text)
                elif isinstance(event, ModelToolCallCompleted):
                    requested.append(event.tool_call)
                elif isinstance(event, ModelUsageUpdate):
                    request_usage = event.usage
            if request_usage is not None:
                total_usage = add_token_usage(total_usage, request_usage)
            text = ''.join(text_parts).strip()
            if not requested:
                final_text = text
                break
            messages.append(build_assistant_message(text, requested))
            results: list[tuple[ToolCall, ToolResult]] = []
            for tool_call in requested:
                execution = await self.executor.execute(tool_call)
                result = execution.result
                results.append((tool_call, result))
                tool_calls.append(tool_call.name)
            messages.append(build_tool_result_message(results))
            if round_index == arguments.max_rounds:
                messages.append(
                    {
                        'role': 'user',
                        'content': (
                            'Round limit reached. Return the structured '
                            'work report now using existing evidence.'
                        ),
                    }
                )

        if not final_text:
            final_text = 'Task subagent reached its round limit without a report.'
            return ToolResult.fail(
                'subagent_no_report',
                final_text,
                metadata=metadata(total_usage, tool_calls),
            )
        return ToolResult.ok(
            'Task subagent returned a structured report.',
            content=final_text,
            metadata=metadata(total_usage, tool_calls),
        )


def create_subagent_registry(
    root: Path,
    workspace_tracker: 'WorkspaceTracker | None' = None,
    *,
    team_bus: 'MessageBus | None' = None,
    agent_id: str = 'task_subagent',
    control_root: Path | None = None,
) -> ToolRegistry:
    from forge.mcp import MCPClientManager
    from forge.tools.filesystem import (
        CreateDirectoryTool,
        ReplaceTextTool,
        WriteFileChunkTool,
        WriteFileTool,
    )
    from forge.tools.git import GitDiffTool
    from forge.tools.memory import create_memory_tools
    from forge.tools.mcp import MCPTool
    from forge.tools.patch import ApplyPatchTool
    from forge.tools.shell import RunCommandTool
    from forge.tools.task_graph import create_task_graph_tools
    from forge.tools.team import create_team_tools
    from forge.tools.verify import VerifyTool
    from forge.runtime.workspace import WorkspaceTracker

    tracker = workspace_tracker or WorkspaceTracker(root)
    state_root = (control_root or root).resolve()
    bus = team_bus
    mcp_manager = MCPClientManager.from_config_file(state_root)
    tools = [
        ListDirectoryTool(root),
        FindFilesTool(root),
        ReadFileTool(root),
        GrepTool(root),
        CreateDirectoryTool(root),
        WriteFileTool(root),
        WriteFileChunkTool(root),
        ReplaceTextTool(root),
        ApplyPatchTool(root),
        RunCommandTool(root),
        VerifyTool(root, tracker),
        GitStatusTool(root),
        GitDiffTool(root),
        *create_task_graph_tools(state_root),
        *create_memory_tools(state_root),
        *create_team_tools(state_root, bus=bus, agent_id=agent_id),
        *[
            MCPTool(root, remote_tool)
            for remote_tool in mcp_manager.list_tools()
        ],
    ]
    return ToolRegistry(
        [
            tool
            for tool in tools
            if tool.name not in SUBAGENT_EXCLUDED_TOOLS
        ],
        workspace_tracker=tracker,
    )


def render_subagent_task(arguments: TaskSubagentInput) -> str:
    focus = (
        '\nFocus paths:\n' + '\n'.join(f'- {path}' for path in arguments.focus_paths)
        if arguments.focus_paths
        else ''
    )
    return f'Task:\n{arguments.task}{focus}'


def metadata(usage: Any, tool_calls: list[str]) -> dict[str, Any]:
    return {
        'subagent': 'task',
        'input_tokens': usage.total_input_tokens,
        'output_tokens': usage.output_tokens,
        'tool_calls': tool_calls,
    }
