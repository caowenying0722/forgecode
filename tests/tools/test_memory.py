'''Tests for model-managed repository memory tools.'''

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from forge.tools.base import ToolRegistry
from forge.tools.memory import create_memory_tools


def run(coroutine: object):
    return asyncio.run(coroutine)  # type: ignore[arg-type]


def test_memory_tools_create_read_update_delete(tmp_path: Path) -> None:
    registry = ToolRegistry(create_memory_tools(tmp_path))

    created = run(
        registry.execute(
            'memory_write',
            {
                'name': 'testing',
                'content': 'Run uv run pytest.',
                'description': 'Test command',
            },
        )
    )
    listed = run(registry.execute('memory_list', {}))
    read = run(registry.execute('memory_read', {'name': 'testing'}))
    updated = run(
        registry.execute(
            'memory_update',
            {
                'name': 'testing',
                'content': 'Run uv run pytest -q.',
            },
        )
    )
    deleted = run(registry.execute('memory_delete', {'name': 'testing'}))

    assert created.success is True
    assert listed.success is True
    listed_payload = json.loads(listed.content)
    assert listed_payload[0]['name'] == 'testing'
    assert listed_payload[0]['source'] == 'model_memory_tool'
    assert listed_payload[0]['created_at']
    assert listed_payload[0]['updated_at']
    assert read.success is True
    read_payload = json.loads(read.content)
    assert read_payload['content'] == 'Run uv run pytest.'
    assert read_payload['source'] == 'model_memory_tool'
    assert updated.success is True
    assert deleted.success is True
    assert not (tmp_path / '.forge' / 'memory' / 'testing.md').exists()


def test_memory_write_rejects_duplicates_and_secrets(tmp_path: Path) -> None:
    registry = ToolRegistry(create_memory_tools(tmp_path))
    run(
        registry.execute(
            'memory_write',
            {'name': 'testing', 'content': 'Use pytest.'},
        )
    )

    duplicate = run(
        registry.execute(
            'memory_write',
            {'name': 'testing', 'content': 'Use pytest again.'},
        )
    )
    secret = run(
        registry.execute(
            'memory_write',
            {'name': 'credentials', 'content': 'API_KEY=sk-secret1234'},
        )
    )

    assert duplicate.success is False
    assert duplicate.error is not None
    assert duplicate.error.code == 'memory_write_rejected'
    assert secret.success is False
    assert secret.error is not None
    assert secret.error.code == 'memory_write_rejected'


def test_memory_tool_effects_are_permission_aware(tmp_path: Path) -> None:
    registry = ToolRegistry(create_memory_tools(tmp_path))

    assert registry.effect('memory_list') == 'read_only'
    assert registry.effect('memory_read') == 'read_only'
    assert registry.effect('memory_write') == 'workspace_write'
    assert registry.effect('memory_update') == 'workspace_write'
    assert registry.effect('memory_delete') == 'workspace_write'
