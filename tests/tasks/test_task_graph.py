'''Tests for persistent dependency-aware task graph tools.'''

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
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


def test_task_graph_records_executable_contract_and_ready_wave(
    tmp_path: Path,
) -> None:
    store = TaskGraphStore(tmp_path)
    registry = ToolRegistry(create_task_graph_tools(tmp_path, store))
    first = run_tool(
        registry,
        'task_create',
        {
            'subject': 'implement parser',
            'acceptance_criteria': ['invalid input is rejected'],
            'read_scope': [{'path': 'forge/parser/**'}],
            'write_scope': [
                {
                    'path': 'forge/parser/core.py',
                    'symbols': ['parse'],
                    'logical_area': 'parser-core',
                }
            ],
            'verification': ['pytest tests/parser -q'],
        },
    )
    second = run_tool(
        registry,
        'task_create',
        {
            'subject': 'write parser docs',
            'read_scope': [{'path': 'forge/parser/types.py'}],
            'write_scope': [{'path': 'docs/parser.md'}],
        },
    )

    plan = run_tool(registry, 'task_graph_plan', {})
    details = json.loads(first.content)

    assert details['acceptance_criteria'] == ['invalid input is rejected']
    assert details['write_scope'][0]['symbols'] == ['parse']
    assert details['verification'] == ['pytest tests/parser -q']
    assert plan.success is True
    assert plan.metadata['ready_task_ids'] == [
        first.metadata['task_id'],
        second.metadata['task_id'],
    ]


def test_write_overlap_is_serialized_and_blocks_claim(tmp_path: Path) -> None:
    store = TaskGraphStore(tmp_path)
    first = store.create(
        'change authentication',
        write_scope=[{'path': 'forge/auth.py', 'symbols': ['login']}],
    )
    second = store.create(
        'change authentication errors',
        write_scope=[{'path': 'forge/auth.py', 'symbols': ['login']}],
    )

    assert [task.id for task in store.ready_wave()] == [first.id]
    store.claim(first.id, owner='agent-a')

    assert store.can_start(second.id) is False
    try:
        store.claim(second.id, owner='agent-b')
    except ValueError as error:
        assert first.id in str(error)
    else:
        raise AssertionError('overlapping write task should not be claimable')


def test_different_symbols_require_cautious_mode_for_parallel_work(
    tmp_path: Path,
) -> None:
    store = TaskGraphStore(tmp_path)
    first = store.create(
        'change parser',
        execution='cautious',
        write_scope=[{'path': 'forge/core.py', 'symbols': ['parse']}],
    )
    second = store.create(
        'change renderer',
        execution='cautious',
        write_scope=[{'path': 'forge/core.py', 'symbols': ['render']}],
    )

    conflict = store.conflict(first, second)

    assert conflict is not None
    assert conflict.kind == 'same_file_different_symbols'
    assert conflict.blocks_parallel is False
    assert [task.id for task in store.ready_wave()] == [first.id, second.id]


def test_old_task_json_loads_with_new_contract_defaults(tmp_path: Path) -> None:
    directory = tmp_path / '.forge' / 'task-graph'
    directory.mkdir(parents=True)
    task_id = 'graph-task-0123456789ab'
    (directory / f'{task_id}.json').write_text(
        json.dumps({'id': task_id, 'subject': 'legacy task'}),
        encoding='utf-8',
    )

    task = TaskGraphStore(tmp_path).load(task_id)

    assert task.acceptance_criteria == ()
    assert task.read_scope == ()
    assert task.write_scope == ()
    assert task.execution == 'parallel'


def test_declared_verification_requires_completion_evidence(
    tmp_path: Path,
) -> None:
    store = TaskGraphStore(tmp_path)
    task = store.create('verified change', verification=['pytest -q'])
    store.claim(task.id, owner='agent-a')

    try:
        store.complete(task.id)
    except ValueError as error:
        assert 'completion evidence is required' in str(error)
    else:
        raise AssertionError('verification requirements should require evidence')

    completed, _ = store.complete(task.id, evidence=['pytest -q: 12 passed'])
    assert completed.status == 'completed'


def test_concurrent_claim_allows_only_one_owner(tmp_path: Path) -> None:
    store = TaskGraphStore(tmp_path)
    task = store.create('single-owner task')

    def claim(owner: str) -> str:
        try:
            return store.claim(task.id, owner=owner).owner or ''
        except ValueError:
            return 'rejected'

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(claim, ['agent-a', 'agent-b']))

    assert outcomes.count('rejected') == 1
    assert store.load(task.id).owner in {'agent-a', 'agent-b'}
