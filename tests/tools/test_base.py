'''Tests for the shared M1.3 tool contract.'''

import asyncio
from pathlib import Path

from forge.tools import create_default_registry
from forge.tools.base import ToolResult
from forge.tasks.manager import TaskManager
from forge.tools.task import create_task_tools


def run(coroutine: object) -> ToolResult:
    return asyncio.run(coroutine)  # type: ignore[arg-type]


def test_default_registry_exposes_all_tool_schemas(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        'forge.mcp.client.forge_app_root',
        lambda *, app_root=None: tmp_path / 'empty-app',
    )
    registry = create_default_registry(tmp_path)

    assert registry.names == (
        'list_directory',
        'find_files',
        'read_file',
        'grep',
        'create_directory',
        'write_file',
        'write_file_chunk',
        'replace_text',
        'apply_patch',
        'run_command',
        'verify',
        'cleanup_verification_artifacts',
        'git_status',
        'git_diff',
        'task_create',
        'task_list',
        'task_graph_get',
        'task_graph_plan',
        'task_claim',
        'task_complete',
        'memory_list',
        'memory_read',
        'memory_write',
        'memory_update',
        'memory_delete',
        'send_message',
        'check_inbox',
        'request_team_action',
        'respond_team_request',
        'list_team_requests',
        'claim_next_task',
        'task',
        'finish_task',
    )
    assert [definition['name'] for definition in registry.definitions] == list(
        registry.names
    )
    assert all(
        definition['input_schema']['type'] == 'object'
        for definition in registry.definitions
    )
    assert registry.effect('read_file') == 'read_only'
    assert registry.effect('create_directory') == 'workspace_write'
    assert registry.effect('write_file') == 'workspace_write'
    assert registry.effect('write_file_chunk') == 'workspace_write'
    assert registry.effect('replace_text') == 'workspace_write'
    assert registry.effect('apply_patch') == 'workspace_write'
    assert registry.effect('run_command') == 'process'
    assert registry.effect('cleanup_verification_artifacts') == 'process'
    assert registry.effect('task') == 'process'
    assert registry.effect('task_create') == 'workspace_write'
    assert registry.effect('task_list') == 'read_only'
    assert registry.effect('task_graph_get') == 'read_only'
    assert registry.effect('task_graph_plan') == 'read_only'
    assert registry.effect('task_claim') == 'workspace_write'
    assert registry.effect('task_complete') == 'workspace_write'
    assert registry.effect('send_message') == 'read_only'
    assert registry.effect('check_inbox') == 'read_only'
    assert registry.effect('request_team_action') == 'read_only'
    assert registry.effect('respond_team_request') == 'read_only'
    assert registry.effect('list_team_requests') == 'read_only'
    assert registry.effect('claim_next_task') == 'workspace_write'
    assert registry.effect('finish_task') == 'read_only'
    assert registry.effect('missing') is None


def test_tool_descriptions_define_task_boundaries(tmp_path: Path) -> None:
    definitions = {
        definition['name']: definition['description']
        for definition in create_default_registry(tmp_path).definitions
    }
    definitions.update(
        {
            tool.definition['name']: tool.definition['description']
            for tool in create_task_tools(
                tmp_path,
                TaskManager(tmp_path),
            )
        }
    )

    assert 'active-goal linear plan' in definitions['task_plan']
    assert 'durable project task queues' in definitions['task_plan']
    assert 'ordinary bug fixes' in definitions['task_create']
    assert 'existing durable task graph' in definitions['task_graph_get']
    assert 'not for the current active-goal plan' in definitions['task_claim']
    assert 'simple local reads or small focused edits' in definitions['task']
    assert 'run_in_background=true only for slow commands' in (
        definitions['run_command']
    )
    assert 'call create_directory first' in definitions['write_file']
    assert 'durable team message' in definitions['send_message']
    assert 'Collect and consume durable team messages' in definitions['check_inbox']
    assert 'pending/approved/rejected' in definitions['request_team_action']
    assert 'correlated by request_id' in definitions['respond_team_request']
    assert 'idle teammate should self-claim' in definitions['claim_next_task']


def test_registry_returns_structured_unknown_tool_error(
    tmp_path: Path,
) -> None:
    registry = create_default_registry(tmp_path)

    result = run(registry.execute('missing', {}))

    assert result.success is False
    assert result.error is not None
    assert result.error.code == 'unknown_tool'
    assert result.error.details['available_tools'] == list(registry.names)


def test_tool_validation_failure_does_not_raise(tmp_path: Path) -> None:
    registry = create_default_registry(tmp_path)

    result = run(
        registry.execute(
            'read_file',
            {'path': 'README.md', 'unexpected': True},
        )
    )

    assert result.success is False
    assert result.error is not None
    assert result.error.code == 'invalid_arguments'
    assert 'Allowed arguments: end_line, path, start_line' in (
        result.error.message
    )
    assert result.error.details['allowed_arguments'] == [
        'end_line',
        'path',
        'start_line',
    ]
    assert result.error.details['required_arguments'] == ['path']
    assert result.error.details['unknown_arguments'] == ['unexpected']
    assert result.error.details['validation_errors'][0]['type'] == (
        'extra_forbidden'
    )


def test_tool_validation_failure_explains_oversized_content(
    tmp_path: Path,
) -> None:
    registry = create_default_registry(tmp_path)

    result = run(
        registry.execute(
            'write_file',
            {'path': 'large.js', 'content': 'x' * 30_001},
        )
    )

    assert result.success is False
    assert result.error is not None
    assert result.error.code == 'invalid_arguments'
    assert '`content` has 30001 characters; maximum is 30000' in (
        result.error.message
    )
    assert 'Use write_file_chunk' in (
        result.error.message
    )
    assert result.error.details['recovery_hint'].startswith(
        'Use write_file_chunk'
    )


def test_repository_escape_returns_structured_error(tmp_path: Path) -> None:
    registry = create_default_registry(tmp_path)

    result = run(registry.execute('read_file', {'path': '../secret.txt'}))

    assert result.success is False
    assert result.error is not None
    assert result.error.code == 'path_outside_repository'
