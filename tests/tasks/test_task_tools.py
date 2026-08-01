'''Tests for model-visible optional task planning tools.'''

import asyncio
from pathlib import Path

from forge.tasks.manager import TaskManager
from forge.tools.base import ToolRegistry
from forge.tools.task import create_task_tools


def test_task_tools_create_and_advance_persistent_plan(
    tmp_path: Path,
) -> None:
    manager = TaskManager(tmp_path)
    manager.start('Implement and verify the feature')
    registry = ToolRegistry(create_task_tools(tmp_path, manager))

    planned = asyncio.run(
        registry.execute(
            'task_plan',
            {
                'steps': ['Inspect implementation', 'Implement fix', 'Test'],
                'scope_hints': ['forge/'],
            },
        )
    )
    updated = asyncio.run(
        registry.execute(
            'task_update',
            {
                'step_id': 'step-1',
                'status': 'completed',
                'evidence': ['Read the relevant runtime files.'],
            },
        )
    )

    assert planned.success is True
    assert updated.success is True
    assert manager.active is not None
    assert manager.active.current_step_id == 'step-2'
    assert manager.store.current_path.exists()


def test_task_plan_ignores_prose_scope_hints(tmp_path: Path) -> None:
    manager = TaskManager(tmp_path)
    manager.start('开始落地实现')
    registry = ToolRegistry(create_task_tools(tmp_path, manager))

    result = asyncio.run(
        registry.execute(
            'task_plan',
            {
                'steps': ['初始化项目', '实现功能'],
                'scope_hints': [
                    '当前目录为空项目，仅有 task.md',
                    '需要从零创建完整项目',
                    '主要修改 src/game/** 和 tests/**',
                ],
            },
        )
    )

    assert result.success is True
    assert manager.active is not None
    assert manager.active.scope_hints == ('src/game/**', 'tests/**')
    assert result.metadata['ignored_scope_hints'] == [
        '当前目录为空项目，仅有 task.md',
        '需要从零创建完整项目',
    ]


def test_task_plan_rejects_simple_one_step_plan(tmp_path: Path) -> None:
    manager = TaskManager(tmp_path)
    manager.start('Read one file')
    registry = ToolRegistry(create_task_tools(tmp_path, manager))

    result = asyncio.run(
        registry.execute('task_plan', {'steps': ['Read README']})
    )

    assert result.success is False
    assert result.error is not None
    assert result.error.code == 'invalid_arguments'
    assert manager.active is not None
    assert manager.active.planned is False
    assert not manager.store.current_path.exists()


def test_existing_plan_error_explains_supported_update_operations(
    tmp_path: Path,
) -> None:
    manager = TaskManager(tmp_path)
    manager.start('Implement and verify')
    registry = ToolRegistry(create_task_tools(tmp_path, manager))
    asyncio.run(
        registry.execute('task_plan', {'steps': ['Inspect', 'Implement']})
    )

    result = asyncio.run(
        registry.execute('task_plan', {'steps': ['Re-read', 'Rebuild']})
    )

    assert result.success is False
    assert result.error is not None
    assert result.error.code == 'task_plan_rejected'
    assert result.error.details['task_id'] == manager.active.id
    assert result.error.details['active_step_id'] == 'step-1'
    assert 'append' in result.error.details['allowed_operations']
    assert result.error.details['example']['operation'] == 'append'


def test_task_plan_can_append_without_replacing_existing_steps(
    tmp_path: Path,
) -> None:
    manager = TaskManager(tmp_path)
    manager.start('Implement and verify')
    registry = ToolRegistry(create_task_tools(tmp_path, manager))
    asyncio.run(
        registry.execute('task_plan', {'steps': ['Inspect', 'Implement']})
    )

    result = asyncio.run(
        registry.execute(
            'task_plan',
            {'operation': 'append', 'steps': ['Verify acceptance']},
        )
    )

    assert result.success is True
    assert manager.active is not None
    assert [step.title for step in manager.active.steps] == [
        'Inspect',
        'Implement',
        'Verify acceptance',
    ]
