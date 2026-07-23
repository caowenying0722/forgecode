'''Lightweight per-turn TODO planning tool.'''

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from forge.tools.base import Tool, ToolInput, ToolResult


TodoStatus = Literal['pending', 'in_progress', 'completed']
TodoPriority = Literal['high', 'medium', 'low']


@dataclass(frozen=True, slots=True)
class TodoItem:
    content: str
    status: TodoStatus
    priority: TodoPriority
    id: str


class TodoWriteItem(ToolInput):
    content: str = Field(min_length=1, max_length=500)
    status: TodoStatus
    priority: TodoPriority = 'medium'
    id: str = Field(min_length=1, max_length=80)


class TodoWriteInput(ToolInput):
    todos: list[TodoWriteItem] = Field(min_length=1, max_length=20)

    @model_validator(mode='after')
    def validate_todos(self) -> TodoWriteInput:
        ids = [todo.id for todo in self.todos]
        if len(ids) != len(set(ids)):
            raise ValueError('todo ids must be unique')
        in_progress = [todo for todo in self.todos if todo.status == 'in_progress']
        if len(in_progress) > 1:
            raise ValueError('at most one todo may be in_progress')
        if all(todo.status == 'completed' for todo in self.todos):
            raise ValueError('at least one todo must remain pending or in_progress')
        return self


class TodoList:
    '''In-memory TODO state scoped to one Conversation.'''

    def __init__(self) -> None:
        self.items: tuple[TodoItem, ...] = ()
        self.updated = False

    def replace(self, todos: list[TodoWriteItem]) -> tuple[TodoItem, ...]:
        self.items = tuple(
            TodoItem(
                content=todo.content.strip(),
                status=todo.status,
                priority=todo.priority,
                id=todo.id.strip(),
            )
            for todo in todos
        )
        self.updated = True
        return self.items

    def reset_turn(self) -> None:
        self.updated = False

    def render(self) -> str:
        if not self.items:
            return 'No TODOs.'
        return '\n'.join(
            f'- [{item.status}] ({item.priority}) {item.id}: {item.content}'
            for item in self.items
        )


class TodoWriteTool(Tool[TodoWriteInput]):
    name = 'todo_write'
    description = (
        'Create or update the short working TODO list before complex work. '
        'Use this before workspace writes or process commands when the task '
        'has multiple steps, multiple files, architectural changes, or '
        'explicit priority/planning language. Keep exactly one item '
        'in_progress while work is underway and update statuses as progress '
        'is made. This tool records planning state only; it does not modify '
        'repository files.'
    )
    input_model = TodoWriteInput

    def __init__(self, root: Path, todo_list: TodoList) -> None:
        super().__init__(root)
        self.todo_list = todo_list

    async def execute(self, arguments: TodoWriteInput) -> ToolResult:
        items = self.todo_list.replace(arguments.todos)
        return ToolResult.ok(
            f'Updated {len(items)} TODO item(s).',
            content=self.todo_list.render(),
            metadata={
                'todo_write': True,
                'todo_count': len(items),
                'in_progress': sum(
                    item.status == 'in_progress' for item in items
                ),
            },
        )
