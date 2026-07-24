'''Model-visible tools for the persistent task graph.'''

from __future__ import annotations

import json
from pathlib import Path

from pydantic import Field

from forge.tasks.graph import GraphTask, TaskGraphStore
from forge.tools.base import Tool, ToolExecutionError, ToolInput, ToolResult


class TaskCreateInput(ToolInput):
    subject: str = Field(min_length=1, max_length=500)
    description: str = Field(default='', max_length=10_000)
    blocked_by: list[str] = Field(default_factory=list, max_length=50)


class TaskCreateTool(Tool[TaskCreateInput]):
    name = 'task_create'
    description = (
        'Create one persistent task-graph item with optional blocked_by '
        'dependencies. Use only for durable multi-step work queues when the '
        'user explicitly asks to split work into persistent tasks, track '
        'dependencies, resume later, or coordinate multiple agents. Do not '
        'use for ordinary bug fixes, single focused edits, one-turn '
        'investigations, the current turn todo list, or the active-goal '
        'linear plan.'
    )
    input_model = TaskCreateInput
    effect = 'workspace_write'

    def __init__(self, root: Path, store: TaskGraphStore | None = None) -> None:
        super().__init__(root)
        self.store = store or TaskGraphStore(root)

    async def execute(self, arguments: TaskCreateInput) -> ToolResult:
        try:
            task = self.store.create(
                arguments.subject,
                description=arguments.description,
                blocked_by=arguments.blocked_by,
            )
        except ValueError as error:
            raise ToolExecutionError('task_create_rejected', str(error)) from error
        return ToolResult.ok(
            f'Created task {task.id}.',
            content=render_task_json(task),
            metadata={'task_id': task.id},
        )


class TaskListInput(ToolInput):
    include_completed: bool = True


class TaskListTool(Tool[TaskListInput]):
    name = 'task_list'
    description = (
        'List persistent task-graph items with status, owner, and blocked_by '
        'dependencies. Use when inspecting an existing durable task graph, '
        'not as a default first step for ordinary repository work.'
    )
    input_model = TaskListInput

    def __init__(self, root: Path, store: TaskGraphStore | None = None) -> None:
        super().__init__(root)
        self.store = store or TaskGraphStore(root)

    async def execute(self, arguments: TaskListInput) -> ToolResult:
        tasks = [
            task
            for task in self.store.list()
            if arguments.include_completed or task.status != 'completed'
        ]
        if not tasks:
            return ToolResult.ok('No task-graph items found.')
        return ToolResult.ok(
            f'Listed {len(tasks)} task-graph item(s).',
            content='\n'.join(render_task_line(task) for task in tasks),
            metadata={'task_count': len(tasks)},
        )


class TaskGraphGetInput(ToolInput):
    task_id: str = Field(min_length=1)


class TaskGraphGetTool(Tool[TaskGraphGetInput]):
    name = 'task_graph_get'
    description = (
        'Get full JSON details for one persistent task-graph item. Use only '
        'when operating on an existing durable task graph.'
    )
    input_model = TaskGraphGetInput

    def __init__(self, root: Path, store: TaskGraphStore | None = None) -> None:
        super().__init__(root)
        self.store = store or TaskGraphStore(root)

    async def execute(self, arguments: TaskGraphGetInput) -> ToolResult:
        try:
            task = self.store.load(arguments.task_id)
        except (FileNotFoundError, ValueError) as error:
            raise ToolExecutionError('task_not_found', str(error)) from error
        return ToolResult.ok(
            f'Read task {task.id}.',
            content=render_task_json(task),
            metadata={'task_id': task.id},
        )


class TaskClaimInput(ToolInput):
    task_id: str = Field(min_length=1)
    owner: str = Field(default='agent', min_length=1, max_length=200)


class TaskClaimTool(Tool[TaskClaimInput]):
    name = 'task_claim'
    description = (
        'Claim one pending unblocked task-graph item. This sets owner and '
        'moves status from pending to in_progress. Use only for durable '
        'task-graph workflows, not for the current active-goal plan.'
    )
    input_model = TaskClaimInput
    effect = 'workspace_write'

    def __init__(self, root: Path, store: TaskGraphStore | None = None) -> None:
        super().__init__(root)
        self.store = store or TaskGraphStore(root)

    async def execute(self, arguments: TaskClaimInput) -> ToolResult:
        try:
            task = self.store.claim(arguments.task_id, owner=arguments.owner)
        except (FileNotFoundError, ValueError) as error:
            raise ToolExecutionError('task_claim_rejected', str(error)) from error
        return ToolResult.ok(
            f'Claimed task {task.id}.',
            content=render_task_json(task),
            metadata={'task_id': task.id, 'owner': task.owner},
        )


class TaskCompleteInput(ToolInput):
    task_id: str = Field(min_length=1)
    evidence: list[str] = Field(default_factory=list, max_length=20)


class TaskCompleteTool(Tool[TaskCompleteInput]):
    name = 'task_complete'
    description = (
        'Complete one in-progress task-graph item and report downstream '
        'pending tasks that became unblocked. Use only for durable task-graph '
        'workflows after the claimed graph task is actually done.'
    )
    input_model = TaskCompleteInput
    effect = 'workspace_write'

    def __init__(self, root: Path, store: TaskGraphStore | None = None) -> None:
        super().__init__(root)
        self.store = store or TaskGraphStore(root)

    async def execute(self, arguments: TaskCompleteInput) -> ToolResult:
        try:
            task, unblocked = self.store.complete(
                arguments.task_id,
                evidence=arguments.evidence,
            )
        except (FileNotFoundError, ValueError) as error:
            raise ToolExecutionError('task_complete_rejected', str(error)) from error
        lines = [render_task_json(task)]
        if unblocked:
            lines.extend(
                [
                    '',
                    'Unblocked downstream tasks:',
                    *[render_task_line(item) for item in unblocked],
                ]
            )
        return ToolResult.ok(
            f'Completed task {task.id}.',
            content='\n'.join(lines),
            metadata={
                'task_id': task.id,
                'unblocked_task_ids': [item.id for item in unblocked],
            },
        )


def create_task_graph_tools(
    root: Path,
    store: TaskGraphStore | None = None,
) -> tuple[
    TaskCreateTool,
    TaskListTool,
    TaskGraphGetTool,
    TaskClaimTool,
    TaskCompleteTool,
]:
    shared = store or TaskGraphStore(root)
    return (
        TaskCreateTool(root, shared),
        TaskListTool(root, shared),
        TaskGraphGetTool(root, shared),
        TaskClaimTool(root, shared),
        TaskCompleteTool(root, shared),
    )


def render_task_line(task: GraphTask) -> str:
    owner = f' owner={task.owner}' if task.owner else ''
    dependencies = (
        f' blocked_by=[{", ".join(task.blocked_by)}]'
        if task.blocked_by
        else ''
    )
    return f'- {task.id} [{task.status}]{owner}{dependencies}: {task.subject}'


def render_task_json(task: GraphTask) -> str:
    return json.dumps(task.as_dict(), ensure_ascii=False, indent=2)
