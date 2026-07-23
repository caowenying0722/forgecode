'''Tests for todo_write planning tool.'''

from __future__ import annotations

import asyncio
from pathlib import Path

from forge.tools.todo import TodoList, TodoWriteTool


def test_todo_write_updates_in_memory_plan(tmp_path: Path) -> None:
    todos = TodoList()
    tool = TodoWriteTool(tmp_path, todos)

    result = asyncio.run(
        tool.run(
            {
                'todos': [
                    {
                        'id': 'inspect',
                        'content': 'Inspect relevant files',
                        'status': 'completed',
                        'priority': 'high',
                    },
                    {
                        'id': 'implement',
                        'content': 'Implement the change',
                        'status': 'in_progress',
                        'priority': 'high',
                    },
                ]
            }
        )
    )

    assert result.success is True
    assert result.metadata['todo_write'] is True
    assert '[completed] (high) inspect' in todos.render()
    assert '[in_progress] (high) implement' in todos.render()


def test_todo_write_rejects_multiple_in_progress_items(tmp_path: Path) -> None:
    tool = TodoWriteTool(tmp_path, TodoList())

    result = asyncio.run(
        tool.run(
            {
                'todos': [
                    {
                        'id': 'one',
                        'content': 'One',
                        'status': 'in_progress',
                    },
                    {
                        'id': 'two',
                        'content': 'Two',
                        'status': 'in_progress',
                    },
                ]
            }
        )
    )

    assert result.success is False
    assert result.error is not None
    assert result.error.code == 'invalid_arguments'
