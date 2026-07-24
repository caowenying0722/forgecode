'''Tests for persistent dependency-aware task graph tools.'''

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from forge.tasks.graph import TaskGraphStore
from forge.tools.base import ToolRegistry
from forge.tools.task_graph import create_task_graph_tools


def run_tool(registry: ToolRegistry, name: str, arguments: dict):
    return asyncio.run(registry.execute(name, arguments))


def test_task_graph_claim_blocks_until_dependencies_complete(
    tmp_path: Path,
) -> None:
    store = TaskGraphStore(tmp_path)
    registry = ToolRegistry(create_task_graph_tools(tmp_path, store))

    schema = run_tool(
        registry,
        'task_create',
        {'subject': 'setup database schema'},
    )
    schema_id = schema.metadata['task_id']
    endpoint = run_tool(
        registry,
        'task_create',
        {
            'subject': 'create API endpoints',
            'blocked_by': [schema_id],
        },
    )
    endpoint_id = endpoint.metadata['task_id']

    blocked = run_tool(
        registry,
        'task_claim',
        {'task_id': endpoint_id, 'owner': 'agent-a'},
    )

    assert blocked.success is False
    assert blocked.error is not None
    assert blocked.error.code == 'task_claim_rejected'
    assert schema_id in blocked.error.message

    claimed = run_tool(
        registry,
        'task_claim',
        {'task_id': schema_id, 'owner': 'agent-a'},
    )
    completed = run_tool(
        registry,
        'task_complete',
        {
            'task_id': schema_id,
            'evidence': ['created schema migration'],
        },
    )
    unblocked_claim = run_tool(
        registry,
        'task_claim',
        {'task_id': endpoint_id, 'owner': 'agent-b'},
    )

    assert claimed.success is True
    assert completed.success is True
    assert endpoint_id in completed.metadata['unblocked_task_ids']
    assert unblocked_claim.success is True
    assert json.loads(unblocked_claim.content)['owner'] == 'agent-b'
    assert (
        tmp_path / '.forge' / 'task-graph' / f'{schema_id}.json'
    ).exists()


def test_task_graph_read_tools_list_and_get_details(tmp_path: Path) -> None:
    store = TaskGraphStore(tmp_path)
    registry = ToolRegistry(create_task_graph_tools(tmp_path, store))
    created = run_tool(
        registry,
        'task_create',
        {
            'subject': 'write docs',
            'description': 'Document the task graph behavior.',
        },
    )

    listed = run_tool(registry, 'task_list', {})
    details = run_tool(
        registry,
        'task_graph_get',
        {'task_id': created.metadata['task_id']},
    )

    assert listed.success is True
    assert created.metadata['task_id'] in listed.content
    assert details.success is True
    assert json.loads(details.content)['description'] == (
        'Document the task graph behavior.'
    )
