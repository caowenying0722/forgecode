'''Integration tests for the M2 model-tool-verification loop.'''

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
import subprocess
from typing import Any

from forge.runtime.agent_loop import Conversation
from forge.runtime.completion import TaskPolicy
from forge.runtime.state import (
    CompletionBlocked,
    ConversationEvent,
    ModelStreamEvent,
    ModelTextDelta,
    ModelToolCallCompleted,
    ModelUsageUpdate,
    TokenUsage,
    ToolCall,
    ToolExecutionCompleted,
    TurnCompleted,
    VerificationCompleted,
    WorkspaceChanged,
)
from forge.tools import create_default_registry
from forge.tools.base import Tool, ToolInput, ToolResult


def initialize_git_repository(root: Path) -> None:
    subprocess.run(['git', 'init', '--quiet'], cwd=root, check=True)
    subprocess.run(
        ['git', 'config', 'user.email', 'forge@example.test'],
        cwd=root,
        check=True,
    )
    subprocess.run(
        ['git', 'config', 'user.name', 'ForgeCode Tests'],
        cwd=root,
        check=True,
    )
    (root / 'sample.txt').write_text('old\n', encoding='utf-8')
    subprocess.run(['git', 'add', '.'], cwd=root, check=True)
    subprocess.run(
        ['git', 'commit', '--quiet', '-m', 'baseline'],
        cwd=root,
        check=True,
    )


class FakeModelClient:
    provider = 'fake'

    def __init__(self, *responses: list[ModelStreamEvent]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
    ) -> AsyncIterator[ModelStreamEvent]:
        self.calls.append(
            {'messages': messages, 'tools': tools, 'system': system}
        )
        for event in self.responses.pop(0):
            yield event


class EmptyProcessInput(ToolInput):
    pass


class ProcessModifyTool(Tool[EmptyProcessInput]):
    name = 'process_modify'
    description = 'Modify sample.txt from a process-like test tool.'
    input_model = EmptyProcessInput
    effect = 'process'

    async def execute(self, arguments: EmptyProcessInput) -> ToolResult:
        del arguments
        (self.root / 'sample.txt').write_text('temporary\n', encoding='utf-8')
        return ToolResult.ok('Temporarily changed sample.txt.')


class ProcessRevertTool(Tool[EmptyProcessInput]):
    name = 'process_revert'
    description = 'Revert sample.txt from a process-like test tool.'
    input_model = EmptyProcessInput
    effect = 'process'

    async def execute(self, arguments: EmptyProcessInput) -> ToolResult:
        del arguments
        (self.root / 'sample.txt').write_text('old\n', encoding='utf-8')
        return ToolResult.ok('Reverted sample.txt to the turn baseline.')


def response_with_tool(call: ToolCall) -> list[ModelStreamEvent]:
    return [
        ModelUsageUpdate(usage=TokenUsage(10, 0)),
        ModelToolCallCompleted(tool_call=call),
        ModelUsageUpdate(usage=TokenUsage(10, 2)),
    ]


def response_with_tools(*calls: ToolCall) -> list[ModelStreamEvent]:
    return [
        ModelUsageUpdate(usage=TokenUsage(10, 0)),
        *(ModelToolCallCompleted(tool_call=call) for call in calls),
        ModelUsageUpdate(usage=TokenUsage(10, 2)),
    ]


def text_response(text: str) -> list[ModelStreamEvent]:
    return [
        ModelUsageUpdate(usage=TokenUsage(10, 0)),
        ModelTextDelta(text=text),
        ModelUsageUpdate(usage=TokenUsage(10, 2)),
    ]


def finish_response(
    call_id: str,
    *,
    task_kind: str,
    status: str = 'completed',
    summary: str = 'Finished.',
    blocked_reasons: list[str] | None = None,
) -> list[ModelStreamEvent]:
    return response_with_tool(
        ToolCall(
            0,
            call_id,
            'finish_task',
            {
                'task_kind': task_kind,
                'status': status,
                'summary': summary,
                'blocked_reasons': blocked_reasons or [],
            },
        )
    )


def todo_response(call_id: str = 'toolu_todo') -> list[ModelStreamEvent]:
    return response_with_tool(
        ToolCall(
            0,
            call_id,
            'todo_write',
            {
                'todos': [
                    {
                        'content': 'Update the sample file',
                        'status': 'in_progress',
                        'priority': 'high',
                        'id': 'edit-sample',
                    },
                    {
                        'content': 'Verify and finish',
                        'status': 'pending',
                        'priority': 'medium',
                        'id': 'verify-finish',
                    },
                ]
            },
        )
    )


def collect_turn(
    conversation: Conversation,
    prompt: str,
) -> list[ConversationEvent]:
    async def collect() -> list[ConversationEvent]:
        return [event async for event in conversation.stream(prompt)]

    return asyncio.run(collect())


def read_only_stagnation_calls(prefix: str) -> list[ToolCall]:
    '''Build one evidence read followed by eight read-only no-progress calls.'''
    specifications = [
        ('read_file', {'path': 'sample.txt'}),
        ('grep', {'path': 'sample.txt', 'pattern': 'old'}),
        ('run_command', {'command': 'git status --short'}),
        ('read_file', {'path': 'sample.txt'}),
        ('grep', {'path': 'sample.txt', 'pattern': '^old$'}),
        ('run_command', {'command': 'git diff --check'}),
        ('read_file', {'path': 'sample.txt'}),
        ('grep', {'path': 'sample.txt', 'pattern': 'o.d'}),
        ('run_command', {'command': 'git status --porcelain=v1'}),
    ]
    return [
        ToolCall(0, f'{prefix}-{index}', name, arguments)
        for index, (name, arguments) in enumerate(specifications, start=1)
    ]


def test_todo_required_enters_planning_recovery_before_write(
    tmp_path: Path,
) -> None:
    initialize_git_repository(tmp_path)
    edit = ToolCall(
        0,
        'toolu_edit',
        'replace_text',
        {
            'path': 'sample.txt',
            'old_text': 'old\n',
            'new_text': 'new\n',
        },
    )
    verify = ToolCall(
        0, 'toolu_verify', 'verify', {'command': 'git diff --check'}
    )
    client = FakeModelClient(
        response_with_tool(edit),
        todo_response(),
        response_with_tool(edit),
        response_with_tool(verify),
        finish_response(
            'finish_after_todo',
            task_kind='change',
            summary='Updated sample after planning.',
        ),
    )
    conversation = Conversation(
        client=client,
        registry=create_default_registry(tmp_path),
    )

    events = collect_turn(
        conversation,
        '帮我完整实现权限 hook 系统并修改 sample.txt',
    )

    completed = events[-1]
    second_tools = {
        str(tool.get('name')) for tool in client.calls[1]['tools'] or []
    }
    assert second_tools == {'todo_write'}
    assert '[ForgeCode Planning Recovery]' in client.calls[1]['system']
    assert (tmp_path / 'sample.txt').read_text(encoding='utf-8') == 'new\n'
    assert any(
        isinstance(event, ToolExecutionCompleted)
        and event.tool_call.name == 'todo_write'
        and event.result.success
        for event in events
    )
    assert isinstance(completed, TurnCompleted)
    assert completed.result.status == 'completed'
    assert completed.result.changed_paths == ('sample.txt',)


def test_agent_loop_rejects_early_answer_then_accepts_verify_evidence(
    tmp_path: Path,
) -> None:
    initialize_git_repository(tmp_path)
    edit = ToolCall(
        0,
        'toolu_edit',
        'replace_text',
        {
            'path': 'sample.txt',
            'old_text': 'old\n',
            'new_text': 'new\n',
        },
    )
    verify = ToolCall(
        0, 'toolu_verify', 'verify', {'command': 'git diff --check'}
    )
    client = FakeModelClient(
        response_with_tool(edit),
        finish_response('finish_early', task_kind='change'),
        response_with_tool(verify),
        finish_response(
            'finish_verified',
            task_kind='change',
            summary='Implemented and verified.',
        ),
    )
    conversation = Conversation(
        client=client,
        registry=create_default_registry(tmp_path),
        task_policy=TaskPolicy(require_verification=True),
    )
    events = collect_turn(conversation, 'Change and verify sample.txt')
    completed = events[-1]

    assert any(isinstance(item, WorkspaceChanged) for item in events)
    assert any(isinstance(item, CompletionBlocked) for item in events)
    assert any(isinstance(item, VerificationCompleted) for item in events)
    assert isinstance(completed, TurnCompleted)
    assert completed.result.changed_paths == ('sample.txt',)
    assert completed.result.verification is not None
    assert completed.result.verification.success is True
    feedback = str(client.calls[2]['messages'][-1]['content'])
    assert 'has not been verified' in feedback


def test_default_policy_rejects_unverified_change_then_accepts_verify(
    tmp_path: Path,
) -> None:
    initialize_git_repository(tmp_path)
    edit = ToolCall(
        0,
        'toolu_default_edit',
        'replace_text',
        {
            'path': 'sample.txt',
            'old_text': 'old\n',
            'new_text': 'new\n',
        },
    )
    verify = ToolCall(
        0,
        'toolu_default_verify',
        'verify',
        {'command': 'git diff --check'},
    )
    client = FakeModelClient(
        response_with_tool(edit),
        finish_response(
            'finish_without_verify',
            task_kind='change',
            summary='Implemented the requested change.',
        ),
        response_with_tool(verify),
        finish_response(
            'finish_with_verify',
            task_kind='change',
            summary='Implemented and verified the requested change.',
        ),
    )
    conversation = Conversation(
        client=client,
        registry=create_default_registry(tmp_path),
    )

    events = collect_turn(conversation, 'Change sample.txt')

    completed = events[-1]
    assert isinstance(completed, TurnCompleted)
    assert completed.result.status == 'completed'
    assert completed.result.changed_paths == ('sample.txt',)
    assert completed.result.verification is not None
    assert any(isinstance(item, CompletionBlocked) for item in events)


def test_replayed_game_evidence_can_progress_to_edit_and_verification(
    tmp_path: Path,
) -> None:
    game_files = {
        'play/js/world.js': 'const faceMode = buggy;\n',
        'play/js/game.js': 'export const game = true;\n',
        'play/js/player.js': 'export const player = true;\n',
        'play/js/constants.js': 'export const BLOCK = 1;\n',
        'play/index.html': '<main>game</main>\n',
    }
    for path, content in game_files.items():
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding='utf-8')
    initialize_git_repository(tmp_path)

    initial_reads = tuple(
        ToolCall(0, f'initial-{index}', 'read_file', {'path': path})
        for index, path in enumerate(game_files)
    )
    replay_reads = tuple(
        ToolCall(
            0,
            f'replay-{index}',
            'read_file',
            {'path': path, 'start_line': 1, 'end_line': 500},
        )
        for index, path in enumerate(game_files)
    )
    edit = ToolCall(
        0,
        'edit-world',
        'replace_text',
        {
            'path': 'play/js/world.js',
            'old_text': 'const faceMode = buggy;\n',
            'new_text': 'const faceMode = six-sided;\n',
        },
    )
    verify = ToolCall(
        0,
        'verify-game',
        'verify',
        {'command': 'git diff --check'},
    )
    client = FakeModelClient(
        response_with_tools(*initial_reads),
        response_with_tools(*replay_reads),
        response_with_tool(edit),
        response_with_tool(verify),
        finish_response(
            'finish-game',
            task_kind='change',
            summary='Fixed and verified the block rendering code.',
        ),
    )
    conversation = Conversation(
        client=client,
        registry=create_default_registry(tmp_path),
        task_policy=TaskPolicy(
            require_changes=True,
            require_verification=True,
        ),
    )

    events = collect_turn(conversation, '修复方块材质渲染')

    replay_results = [
        event.result
        for event in events
        if isinstance(event, ToolExecutionCompleted)
        and event.tool_call.id.startswith('replay-')
    ]
    assert len(replay_results) == len(game_files)
    assert all(result.metadata['evidence_replayed'] for result in replay_results)
    completed = events[-1]
    assert isinstance(completed, TurnCompleted)
    assert completed.result.status == 'completed'
    assert completed.result.verification is not None
    assert completed.result.verification.success is True
    assert 'six-sided' in (
        tmp_path / 'play/js/world.js'
    ).read_text(encoding='utf-8')


def test_failed_patch_recovers_to_valid_begin_patch_and_completion(
    tmp_path: Path,
) -> None:
    initialize_git_repository(tmp_path)
    invalid_patch = (
        '*** Begin ' 'Patch\n'
        '*** Update File: sample.txt\n'
        '@@\n'
        '-not-current\n'
        '+new\n'
        '*** End ' 'Patch'
    )
    valid_patch = (
        '*** Begin ' 'Patch\n'
        '*** Update File: sample.txt\n'
        '@@\n'
        '-old\n'
        '+new\n'
        '*** End ' 'Patch'
    )
    client = FakeModelClient(
        response_with_tool(
            ToolCall(
                0,
                'patch-failed',
                'apply_patch',
                {'patch': invalid_patch},
            )
        ),
        response_with_tool(
            ToolCall(
                0,
                'read-target',
                'read_file',
                {'path': 'sample.txt'},
            )
        ),
        response_with_tool(
            ToolCall(
                0,
                'patch-retried',
                'apply_patch',
                {'patch': valid_patch},
            )
        ),
        response_with_tool(
            ToolCall(
                0,
                'verify-recovery',
                'verify',
                {'command': 'git diff --check'},
            )
        ),
        finish_response(
            'finish-recovery',
            task_kind='change',
            summary='Recovered, changed, and verified sample.txt.',
        ),
    )
    conversation = Conversation(
        client=client,
        registry=create_default_registry(tmp_path),
        task_policy=TaskPolicy(
            require_changes=True,
            require_verification=True,
        ),
    )

    events = collect_turn(conversation, 'Fix and verify sample.txt')

    completed = events[-1]
    assert isinstance(completed, TurnCompleted)
    assert completed.result.status == 'completed'
    assert completed.result.model_calls == 5
    assert completed.result.changed_paths == ('sample.txt',)
    assert (tmp_path / 'sample.txt').read_text(encoding='utf-8') == 'new\n'
    assert '[Failed Mutation Recovery]' in client.calls[1]['system']
    assert 'patch_context_not_found' in client.calls[1]['system']
    assert '[Failed Mutation Recovery]' not in client.calls[3]['system']


def test_write_then_revert_to_baseline_enters_edit_recovery(
    tmp_path: Path,
) -> None:
    initialize_git_repository(tmp_path)
    (tmp_path / 'sample.txt').write_bytes(b'old\n')
    client = FakeModelClient(
        response_with_tools(
            ToolCall(
                0,
                'write-new',
                'replace_text',
                {
                    'path': 'sample.txt',
                    'old_text': 'old\n',
                    'new_text': 'new\n',
                },
            ),
            ToolCall(
                1,
                'restore-old',
                'replace_text',
                {
                    'path': 'sample.txt',
                    'old_text': 'new\n',
                    'new_text': 'old\n',
                },
            ),
        ),
        text_response('Done.'),
    )
    conversation = Conversation(
        client=client,
        registry=create_default_registry(tmp_path),
        mutation_recovery_limit=2,
    )

    events = collect_turn(conversation, 'Change sample.txt')

    completed = events[-1]
    assert isinstance(completed, TurnCompleted)
    assert completed.result.status == 'stuck'
    assert completed.result.changed_paths == ()
    assert completed.result.model_calls == 2
    assert '[Failed Mutation Recovery]' in client.calls[1]['system']
    assert 'no_workspace_change' in client.calls[1]['system']
    assert (tmp_path / 'sample.txt').read_text(encoding='utf-8') == 'old\n'


def test_later_write_failure_in_same_response_remains_in_recovery(
    tmp_path: Path,
) -> None:
    initialize_git_repository(tmp_path)
    successful_edit = ToolCall(
        0,
        'successful-edit',
        'replace_text',
        {
            'path': 'sample.txt',
            'old_text': 'old\n',
            'new_text': 'new\n',
        },
    )
    failed_edit = ToolCall(
        1,
        'later-failed-edit',
        'replace_text',
        {
            'path': 'sample.txt',
            'old_text': 'missing\n',
            'new_text': 'extra\n',
        },
    )
    client = FakeModelClient(
        response_with_tools(successful_edit, failed_edit),
        text_response('Done after only the first edit.'),
    )
    conversation = Conversation(
        client=client,
        registry=create_default_registry(tmp_path),
        mutation_recovery_limit=2,
    )

    events = collect_turn(conversation, 'Apply both required edits')

    failed_result = next(
        event.result
        for event in events
        if isinstance(event, ToolExecutionCompleted)
        and event.tool_call.id == 'later-failed-edit'
    )
    assert failed_result.error is not None
    assert failed_result.error.code == 'text_not_found'
    completed = events[-1]
    assert isinstance(completed, TurnCompleted)
    assert completed.result.status == 'stuck'
    assert completed.result.model_calls == 2
    assert completed.result.changed_paths == ('sample.txt',)
    assert '[Failed Mutation Recovery]' in client.calls[1]['system']
    assert 'text_not_found' in client.calls[1]['system']
    assert (tmp_path / 'sample.txt').read_text(encoding='utf-8') == 'new\n'


def test_pending_write_failure_hides_finish_and_bounds_invalid_attempts(
    tmp_path: Path,
) -> None:
    initialize_git_repository(tmp_path)
    edit = ToolCall(
        0,
        'initial-edit',
        'replace_text',
        {
            'path': 'sample.txt',
            'old_text': 'old\n',
            'new_text': 'new\n',
        },
    )
    verify = ToolCall(
        0,
        'verify-initial-edit',
        'verify',
        {'command': 'git diff --check'},
    )
    failed_edit = ToolCall(
        0,
        'unresolved-edit',
        'replace_text',
        {
            'path': 'sample.txt',
            'old_text': 'missing\n',
            'new_text': 'extra\n',
        },
    )
    client = FakeModelClient(
        response_with_tool(edit),
        response_with_tool(failed_edit),
        *(
            finish_response(
                f'premature-finish-{index}',
                task_kind='change',
                summary='Finished despite the unresolved edit.',
            )
            for index in range(1, 4)
        ),
    )
    conversation = Conversation(
        client=client,
        registry=create_default_registry(tmp_path),
    )

    events = collect_turn(conversation, 'Apply and verify all required edits')

    finish_result = next(
        event.result
        for event in events
        if isinstance(event, ToolExecutionCompleted)
        and event.tool_call.id == 'premature-finish-1'
    )
    assert finish_result.error is not None
    assert finish_result.error.code == 'tool_not_available_in_phase'
    assert 'finish_task' not in {
        definition['name'] for definition in client.calls[2]['tools'] or ()
    }
    completed = events[-1]
    assert isinstance(completed, TurnCompleted)
    assert completed.result.status == 'stuck'
    assert completed.result.model_calls == 5
    assert completed.result.verification is None
    assert 'malformed or schema-invalid tool requests' in completed.result.text
    assert 'Finished despite the unresolved edit.' not in completed.result.text


def test_parent_not_found_recovery_exposes_create_directory_then_write(
    tmp_path: Path,
) -> None:
    initialize_git_repository(tmp_path)
    failed_write = ToolCall(
        0,
        'write-missing-parent',
        'write_file',
        {
            'path': 'game/index.html',
            'content': '<!doctype html>\n<html></html>\n',
        },
    )
    todo = ToolCall(
        0,
        'plan-game',
        'todo_write',
        {
            'todos': [
                {
                    'id': 'plan',
                    'content': '规划金铲铲风格 Web UI 结构和交互',
                    'status': 'completed',
                    'priority': 'high',
                },
                {
                    'id': 'implement',
                    'content': '创建 game 目录并写入页面文件',
                    'status': 'in_progress',
                    'priority': 'high',
                },
                {
                    'id': 'verify',
                    'content': '运行验证命令',
                    'status': 'pending',
                    'priority': 'medium',
                },
            ]
        },
    )
    create_directory = ToolCall(
        0,
        'create-game-directory',
        'create_directory',
        {'path': 'game'},
    )
    write_file = ToolCall(
        1,
        'write-game-index',
        'write_file',
        {
            'path': 'game/index.html',
            'content': '<!doctype html>\n<html></html>\n',
        },
    )
    verify = ToolCall(
        0,
        'verify-game',
        'verify',
        {'command': 'git diff --check'},
    )
    client = FakeModelClient(
        response_with_tool(todo),
        response_with_tool(failed_write),
        response_with_tools(create_directory, write_file),
        response_with_tool(verify),
        finish_response(
            'finish-game',
            task_kind='change',
            summary='Created game web UI skeleton.',
        ),
    )
    conversation = Conversation(
        client=client,
        registry=create_default_registry(tmp_path),
    )

    events = collect_turn(
        conversation,
        '帮我新建一个game/目录，然后写一个高还原度的金铲铲游戏的web界面，先规划再行动',
    )

    first_failure = next(
        event.result
        for event in events
        if isinstance(event, ToolExecutionCompleted)
        and event.tool_call.id == 'write-missing-parent'
    )
    assert first_failure.error is not None
    assert first_failure.error.code == 'parent_not_found'
    recovery_tool_names = {
        definition['name'] for definition in client.calls[2]['tools'] or ()
    }
    assert recovery_tool_names == {
        'create_directory',
        'list_directory',
        'write_file',
    }
    recovery_request = (
        (client.calls[2]['system'] or '')
        + str(client.calls[2]['messages'])
    )
    assert 'Call create_directory for game' in recovery_request
    assert (tmp_path / 'game' / 'index.html').read_text(
        encoding='utf-8'
    ) == '<!doctype html>\n<html></html>\n'
    completed = events[-1]
    assert isinstance(completed, TurnCompleted)
    assert completed.result.status == 'completed'
    assert 'game/index.html' in completed.result.changed_paths


def test_empty_directory_scaffold_does_not_trigger_edit_recovery(
    tmp_path: Path,
) -> None:
    initialize_git_repository(tmp_path)
    directories = tuple(
        ToolCall(
            index,
            f'create-directory-{index}',
            'create_directory',
            {'path': path},
        )
        for index, path in enumerate(
            (
                'src',
                'src/game',
                'src/game/scenes',
                'src/game/entities',
                'src/game/systems',
                'src/game/configs',
                'src/game/tests',
                'public',
            )
        )
    )
    write = ToolCall(
        0,
        'write-after-scaffold',
        'write_file',
        {'path': 'src/game/README.txt', 'content': 'game scaffold\n'},
    )
    verify = ToolCall(
        0,
        'verify-scaffold',
        'verify',
        {'command': 'git diff --check'},
    )
    client = FakeModelClient(
        response_with_tools(*directories),
        response_with_tool(write),
        response_with_tool(verify),
        finish_response(
            'finish-scaffold',
            task_kind='change',
            summary='Created and verified the scaffold.',
        ),
    )
    conversation = Conversation(
        client=client,
        registry=create_default_registry(tmp_path),
    )

    events = collect_turn(conversation, 'Create the requested scaffold')

    completed = events[-1]
    assert isinstance(completed, TurnCompleted)
    assert completed.result.status == 'completed'
    assert (tmp_path / 'src' / 'game' / 'README.txt').is_file()
    assert completed.result.changed_paths == ('src/game/README.txt',)
    failures = [
        event.result.error.code
        for event in events
        if isinstance(event, ToolExecutionCompleted)
        and event.result.error is not None
    ]
    assert 'no_workspace_change' not in failures


def test_existing_directory_scaffold_is_safe_to_resume(
    tmp_path: Path,
) -> None:
    initialize_git_repository(tmp_path)
    for index in range(8):
        (tmp_path / f'existing-{index}').mkdir()
    calls = tuple(
        ToolCall(
            index,
            f'existing-directory-{index}',
            'create_directory',
            {'path': f'existing-{index}'},
        )
        for index in range(8)
    )
    write = ToolCall(
        0,
        'write-after-existing-directories',
        'write_file',
        {'path': 'existing-0/app.txt', 'content': 'resumed\n'},
    )
    verify = ToolCall(
        0,
        'verify-existing-directories',
        'verify',
        {'command': 'git diff --check'},
    )
    conversation = Conversation(
        client=FakeModelClient(
            response_with_tools(*calls),
            response_with_tool(write),
            response_with_tool(verify),
            finish_response(
                'finish-existing-directories',
                task_kind='change',
                summary='Resumed and verified the scaffold.',
            ),
        ),
        registry=create_default_registry(tmp_path),
    )

    events = collect_turn(conversation, 'Create a real project change')

    completed = events[-1]
    assert isinstance(completed, TurnCompleted)
    assert completed.result.status == 'completed'
    assert (tmp_path / 'existing-0' / 'app.txt').is_file()


def test_required_change_stagnation_enters_action_recovery_and_can_finish(
    tmp_path: Path,
) -> None:
    initialize_git_repository(tmp_path)
    investigation = read_only_stagnation_calls('action-success')
    edit = ToolCall(
        0,
        'action-success-edit',
        'replace_text',
        {
            'path': 'sample.txt',
            'old_text': 'old\n',
            'new_text': 'new\n',
        },
    )
    verify = ToolCall(
        0,
        'action-success-verify',
        'verify',
        {'command': 'git diff --check'},
    )
    summary = 'Changed sample.txt after Action Recovery and verified it.'
    client = FakeModelClient(
        *(response_with_tool(call) for call in investigation),
        response_with_tool(edit),
        response_with_tool(verify),
        finish_response(
            'action-success-finish',
            task_kind='change',
            summary=summary,
        ),
    )
    conversation = Conversation(
        client=client,
        registry=create_default_registry(tmp_path),
        task_policy=TaskPolicy(
            require_changes=True,
            require_verification=True,
        ),
    )

    events = collect_turn(conversation, 'Change and verify sample.txt')

    completed = events[-1]
    assert isinstance(completed, TurnCompleted)
    assert completed.result.status == 'completed'
    assert completed.result.model_calls == 12
    assert completed.result.text == summary
    assert completed.result.changed_paths == ('sample.txt',)
    assert completed.result.verification is not None
    assert completed.result.verification.success is True
    assert (tmp_path / 'sample.txt').read_text(encoding='utf-8') == 'new\n'
    assert client.responses == []
    recovery_request = (
        (client.calls[9]['system'] or '')
        + str(client.calls[9]['messages'])
    )
    assert '[ForgeCode Action Recovery]' in recovery_request
    assert client.calls[9]['tools'] is not None
    assert any(
        isinstance(event, CompletionBlocked)
        and any('task-local workspace change' in reason for reason in event.reasons)
        for event in events
    )
    executed_names = [
        event.tool_call.name
        for event in events
        if isinstance(event, ToolExecutionCompleted)
    ]
    assert executed_names[-3:] == ['replace_text', 'verify', 'finish_task']


def test_cli_fix_intent_enters_action_recovery_despite_novel_reads(
    tmp_path: Path,
) -> None:
    initialize_git_repository(tmp_path)
    targets = [
        tmp_path / 'play' / 'js' / f'file-{index}.js'
        for index in range(1, 4)
    ]
    for index, target in enumerate(targets, start=1):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f'old-{index}\n', encoding='utf-8')
    subprocess.run(['git', 'add', '.'], cwd=tmp_path, check=True)
    subprocess.run(
        ['git', 'commit', '--quiet', '-m', 'game baseline'],
        cwd=tmp_path,
        check=True,
    )
    reads = [
        ToolCall(
            0,
            f'novel-read-{index}',
            'read_file',
            {'path': f'play/js/file-{index}.js'},
        )
        for index in range(1, 4)
    ]
    edit = ToolCall(
        0,
        'novel-read-edit',
        'replace_text',
        {
            'path': 'play/js/file-1.js',
            'old_text': 'old-1\n',
            'new_text': 'fixed-1\n',
        },
    )
    verify = ToolCall(
        0,
        'novel-read-verify',
        'verify',
        {'command': 'git diff --check'},
    )
    client = FakeModelClient(
        *(response_with_tool(call) for call in reads),
        response_with_tool(edit),
        response_with_tool(verify),
        finish_response(
            'novel-read-finish',
            task_kind='change',
            summary='Fixed the rendering code after bounded discovery.',
        ),
    )
    conversation = Conversation(
        client=client,
        registry=create_default_registry(tmp_path),
        pre_mutation_limit=3,
        stagnation_warning=20,
        stagnation_limit=30,
    )

    events = collect_turn(
        conversation,
        '当前游戏很多方块只有一两面材质，帮我修复一下',
    )

    completed = events[-1]
    assert isinstance(completed, TurnCompleted)
    assert completed.result.status == 'completed'
    assert completed.result.model_calls == 6
    assert completed.result.changed_paths == ('play/js/file-1.js',)
    assert all(
        any(
            isinstance(event, ToolExecutionCompleted)
            and event.tool_call.id == read.id
            and event.result.success
            for event in events
        )
        for read in reads
    )
    assert '[ForgeCode Action Recovery]' in (
        client.calls[3]['system'] or ''
    )
    recovery_names = {
        str(definition['name'])
        for definition in client.calls[3]['tools'] or ()
    }
    assert 'replace_text' in recovery_names
    assert 'find_files' not in recovery_names
    assert 'list_directory' not in recovery_names


def test_action_recovery_failed_edit_transfers_to_mutation_recovery(
    tmp_path: Path,
) -> None:
    initialize_git_repository(tmp_path)
    read = ToolCall(
        0,
        'action-transfer-read',
        'read_file',
        {'path': 'sample.txt'},
    )
    failed_edit = ToolCall(
        0,
        'action-transfer-failed-edit',
        'replace_text',
        {
            'path': 'sample.txt',
            'old_text': 'not-present\n',
            'new_text': 'new\n',
        },
    )
    valid_edit = ToolCall(
        0,
        'action-transfer-valid-edit',
        'replace_text',
        {
            'path': 'sample.txt',
            'old_text': 'old\n',
            'new_text': 'new\n',
        },
    )
    verify = ToolCall(
        0,
        'action-transfer-verify',
        'verify',
        {'command': 'git diff --check'},
    )
    client = FakeModelClient(
        response_with_tool(read),
        response_with_tool(failed_edit),
        response_with_tool(valid_edit),
        response_with_tool(verify),
        finish_response(
            'action-transfer-finish',
            task_kind='change',
            summary='Recovered from the failed edit and verified.',
        ),
    )
    conversation = Conversation(
        client=client,
        registry=create_default_registry(tmp_path),
        pre_mutation_limit=1,
    )

    events = collect_turn(conversation, 'Fix sample.txt')

    completed = events[-1]
    assert isinstance(completed, TurnCompleted)
    assert completed.result.status == 'completed'
    assert completed.result.changed_paths == ('sample.txt',)
    assert '[ForgeCode Action Recovery]' in (
        client.calls[1]['system'] or ''
    )
    assert '[Failed Mutation Recovery]' in (
        client.calls[2]['system'] or ''
    )
    mutation_tool_names = {
        str(definition['name'])
        for definition in client.calls[2]['tools'] or ()
    }
    assert {'read_file', 'grep', 'replace_text', 'apply_patch'} <= (
        mutation_tool_names
    )
    assert 'verify' not in mutation_tool_names
    assert 'run_command' not in mutation_tool_names
    assert 'finish_task' not in mutation_tool_names
    assert 'verify' in {
        str(definition['name'])
        for definition in client.calls[3]['tools'] or ()
    }


def test_process_modify_then_revert_does_not_reset_pre_mutation_budget(
    tmp_path: Path,
) -> None:
    initialize_git_repository(tmp_path)
    registry = create_default_registry(tmp_path)
    registry.register(ProcessModifyTool(tmp_path))
    registry.register(ProcessRevertTool(tmp_path))
    transient_batch = response_with_tools(
        ToolCall(0, 'process-modify', 'process_modify', {}),
        ToolCall(1, 'process-revert', 'process_revert', {}),
    )
    edit = ToolCall(
        0,
        'process-revert-real-edit',
        'replace_text',
        {
            'path': 'sample.txt',
            'old_text': 'old\n',
            'new_text': 'new\n',
        },
    )
    verify = ToolCall(
        0,
        'process-revert-verify',
        'verify',
        {'command': 'git diff --check'},
    )
    client = FakeModelClient(
        transient_batch,
        response_with_tool(edit),
        response_with_tool(verify),
        finish_response(
            'process-revert-finish',
            task_kind='change',
            summary='Created a persistent change after the reverted batch.',
        ),
    )
    conversation = Conversation(
        client=client,
        registry=registry,
        pre_mutation_limit=1,
    )

    events = collect_turn(conversation, 'Fix sample.txt')

    completed = events[-1]
    assert isinstance(completed, TurnCompleted)
    assert completed.result.status == 'completed'
    assert completed.result.changed_paths == ('sample.txt',)
    revisions = [
        event.revision
        for event in events
        if isinstance(event, WorkspaceChanged)
    ]
    assert revisions[:2] == [1, 2]
    assert '[ForgeCode Action Recovery]' in (
        client.calls[1]['system'] or ''
    )


def test_action_recovery_executes_at_most_one_read_from_a_tool_batch(
    tmp_path: Path,
) -> None:
    initialize_git_repository(tmp_path)
    initial = ToolCall(
        0,
        'read-limit-initial',
        'read_file',
        {'path': 'sample.txt'},
    )
    first_recovery_read = ToolCall(
        0,
        'read-limit-first',
        'read_file',
        {'path': 'sample.txt', 'start_line': 1, 'end_line': 1},
    )
    second_recovery_read = ToolCall(
        1,
        'read-limit-second',
        'grep',
        {'path': 'sample.txt', 'pattern': 'old'},
    )
    edit = ToolCall(
        0,
        'read-limit-edit',
        'replace_text',
        {
            'path': 'sample.txt',
            'old_text': 'old\n',
            'new_text': 'new\n',
        },
    )
    verify = ToolCall(
        0,
        'read-limit-verify',
        'verify',
        {'command': 'git diff --check'},
    )
    client = FakeModelClient(
        response_with_tool(initial),
        response_with_tools(first_recovery_read, second_recovery_read),
        response_with_tool(edit),
        response_with_tool(verify),
        finish_response(
            'read-limit-finish',
            task_kind='change',
            summary='Edited after one bounded recovery read.',
        ),
    )
    conversation = Conversation(
        client=client,
        registry=create_default_registry(tmp_path),
        pre_mutation_limit=1,
    )

    events = collect_turn(conversation, 'Fix sample.txt')

    second_result = next(
        event.result
        for event in events
        if isinstance(event, ToolExecutionCompleted)
        and event.tool_call.id == second_recovery_read.id
    )
    assert second_result.success is False
    assert second_result.error is not None
    assert second_result.error.code == 'action_read_limit_reached'
    post_read_names = {
        str(definition['name'])
        for definition in client.calls[2]['tools'] or ()
    }
    assert 'read_file' not in post_read_names
    assert 'grep' not in post_read_names
    assert isinstance(events[-1], TurnCompleted)
    assert events[-1].result.status == 'completed'


def test_required_change_action_recovery_read_only_call_stops_specifically(
    tmp_path: Path,
) -> None:
    initialize_git_repository(tmp_path)
    investigation = read_only_stagnation_calls('action-read-only')
    recovery_read = ToolCall(
        0,
        'action-read-only-recovery-read',
        'read_file',
        {'path': 'sample.txt', 'start_line': 1, 'end_line': 1},
    )
    client = FakeModelClient(
        *(response_with_tool(call) for call in investigation),
        response_with_tool(recovery_read),
        finish_response(
            'action-read-only-answer-one',
            task_kind='answer',
        ),
        finish_response(
            'action-read-only-answer-two',
            task_kind='answer',
        ),
    )
    conversation = Conversation(
        client=client,
        registry=create_default_registry(tmp_path),
        task_policy=TaskPolicy(
            require_changes=True,
            require_verification=True,
        ),
    )

    events = collect_turn(conversation, 'Change and verify sample.txt')

    completed = events[-1]
    assert isinstance(completed, TurnCompleted)
    assert completed.result.status == 'stuck'
    assert completed.result.model_calls == 12
    assert completed.result.changed_paths == ()
    assert client.responses == []
    assert 'action recovery' in completed.result.text.casefold()
    assert (
        'without new workspace, plan, or repository evidence'
        not in completed.result.text
    )
    recovery_request = (
        (client.calls[9]['system'] or '')
        + str(client.calls[9]['messages'])
    )
    assert '[ForgeCode Action Recovery]' in recovery_request
    assert client.calls[9]['tools'] is not None
    recovery_tool_names = {
        str(definition['name'])
        for definition in client.calls[9]['tools'] or ()
    }
    assert 'replace_text' in recovery_tool_names
    assert 'find_files' not in recovery_tool_names
    assert 'list_directory' not in recovery_tool_names
    assert 'run_command' not in recovery_tool_names
    assert 'verify' not in recovery_tool_names
    recovery_events = [
        event
        for event in events
        if isinstance(event, ToolExecutionCompleted)
        and event.tool_call.id == recovery_read.id
    ]
    assert len(recovery_events) == 1
    assert all(event.result.success for event in recovery_events)
    post_read_tool_names = {
        str(definition['name'])
        for definition in client.calls[10]['tools'] or ()
    }
    assert 'read_file' not in post_read_tool_names
    assert 'grep' not in post_read_tool_names


def test_cli_intent_requires_task_local_edit_for_preexisting_untracked_file(
    tmp_path: Path,
) -> None:
    initialize_git_repository(tmp_path)
    world = tmp_path / 'play' / 'js' / 'world.js'
    world.parent.mkdir(parents=True)
    world.write_text('const faceMode = buggy;\n', encoding='utf-8')
    inspect = ToolCall(
        0,
        'untracked-inspect',
        'git_diff',
        {'path': 'play/js/world.js'},
    )
    edit = ToolCall(
        0,
        'untracked-edit',
        'replace_text',
        {
            'path': 'play/js/world.js',
            'old_text': 'const faceMode = buggy;\n',
            'new_text': 'const faceMode = sixSided;\n',
        },
    )
    verify = ToolCall(
        0,
        'untracked-verify',
        'verify',
        {'command': 'git diff --check'},
    )
    summary = 'Fixed and verified the preexisting untracked game file.'
    client = FakeModelClient(
        response_with_tool(inspect),
        finish_response('untracked-early-finish', task_kind='change'),
        response_with_tool(edit),
        response_with_tool(verify),
        finish_response(
            'untracked-finish',
            task_kind='change',
            summary=summary,
        ),
    )
    conversation = Conversation(
        client=client,
        registry=create_default_registry(tmp_path),
    )

    events = collect_turn(
        conversation,
        '当前游戏很多方块只有一两面材质，帮我修复一下',
    )

    completed = events[-1]
    assert isinstance(completed, TurnCompleted)
    assert completed.result.status == 'completed'
    assert completed.result.text == summary
    assert completed.result.changed_paths == ('play/js/world.js',)
    assert completed.result.verification is not None
    assert completed.result.verification.success is True
    assert world.read_text(encoding='utf-8') == (
        'const faceMode = sixSided;\n'
    )
    inspect_event = next(
        event
        for event in events
        if isinstance(event, ToolExecutionCompleted)
        and event.tool_call.id == inspect.id
    )
    assert inspect_event.result.metadata['synthetic_diff'] is True
    early_finish = next(
        event
        for event in events
        if isinstance(event, ToolExecutionCompleted)
        and event.tool_call.id == 'untracked-early-finish'
    )
    assert early_finish.result.success is False
    assert early_finish.result.error is not None
    assert early_finish.result.error.code == 'finish_rejected'
    assert '[ForgeCode Action Recovery]' in (
        client.calls[2]['system'] or ''
    )


def test_inspection_stagnation_does_not_enter_action_recovery(
    tmp_path: Path,
) -> None:
    initialize_git_repository(tmp_path)
    investigation = read_only_stagnation_calls('inspection')
    summary = 'sample.txt contains the old baseline value.'
    client = FakeModelClient(
        *(response_with_tool(call) for call in investigation),
        text_response(summary),
    )
    conversation = Conversation(
        client=client,
        registry=create_default_registry(tmp_path),
        stagnation_warning=4,
        stagnation_limit=8,
    )

    events = collect_turn(conversation, 'Inspect and explain sample.txt')

    completed = events[-1]
    assert isinstance(completed, TurnCompleted)
    assert completed.result.status == 'completed'
    assert completed.result.text == summary
    assert completed.result.model_calls == 10
    assert completed.result.changed_paths == ()
    assert client.responses == []
    assert client.calls[-1]['tools'] is None
    assert '[ForgeCode Stagnation Final Recovery]' in (
        client.calls[-1]['system'] or ''
    )
    assert all(
        '[ForgeCode Action Recovery]' not in (
            (call['system'] or '') + str(call['messages'])
        )
        for call in client.calls
    )


def test_tool_enabled_checkpoint_recovers_unverified_final_answer(
    tmp_path: Path,
) -> None:
    initialize_git_repository(tmp_path)
    edit = ToolCall(
        0,
        'checkpoint-edit',
        'replace_text',
        {
            'path': 'sample.txt',
            'old_text': 'old\n',
            'new_text': 'new\n',
        },
    )
    first_status = ToolCall(0, 'checkpoint-status-1', 'git_status', {})
    cached_status = ToolCall(0, 'checkpoint-status-2', 'git_status', {})
    verify = ToolCall(
        0,
        'checkpoint-verify',
        'verify',
        {'command': 'git diff --check'},
    )
    summary = 'Updated and verified sample.txt.'
    client = FakeModelClient(
        response_with_tool(edit),
        response_with_tool(first_status),
        response_with_tool(cached_status),
        text_response('I cannot continue because tools are unavailable.'),
        response_with_tool(verify),
        finish_response(
            'checkpoint-finish',
            task_kind='change',
            summary=summary,
        ),
    )
    conversation = Conversation(
        client=client,
        registry=create_default_registry(tmp_path),
        task_policy=TaskPolicy(require_verification=True),
        stagnation_warning=1,
        stagnation_limit=4,
    )

    events = collect_turn(conversation, 'Change and verify sample.txt')

    completed = events[-1]
    assert isinstance(completed, TurnCompleted)
    assert completed.result.status == 'completed'
    assert completed.result.text == summary
    assert completed.result.verification is not None
    assert completed.result.verification.success is True
    assert client.calls[3]['tools'] is not None
    assert client.calls[4]['tools'] is not None
    assert '[ForgeCode Stagnation Final Recovery]' not in (
        client.calls[4]['system'] or ''
    )
    assert any(
        isinstance(event, CompletionBlocked)
        and any('verify tool' in reason for reason in event.reasons)
        for event in events
    )


def test_stagnation_limit_keeps_tools_for_incomplete_change_recovery(
    tmp_path: Path,
) -> None:
    initialize_git_repository(tmp_path)
    edit = ToolCall(
        0,
        'limit-edit',
        'replace_text',
        {
            'path': 'sample.txt',
            'old_text': 'old\n',
            'new_text': 'new\n',
        },
    )
    statuses = [
        ToolCall(0, f'limit-status-{index}', 'git_status', {})
        for index in range(1, 4)
    ]
    verify = ToolCall(
        0,
        'limit-verify',
        'verify',
        {'command': 'git diff --check'},
    )
    client = FakeModelClient(
        response_with_tool(edit),
        *(response_with_tool(call) for call in statuses),
        response_with_tool(verify),
        finish_response(
            'limit-finish',
            task_kind='change',
            summary='Recovered, verified, and finished the change.',
        ),
    )
    conversation = Conversation(
        client=client,
        registry=create_default_registry(tmp_path),
        task_policy=TaskPolicy(require_verification=True),
        stagnation_warning=1,
        stagnation_limit=2,
    )

    events = collect_turn(conversation, 'Change and verify sample.txt')

    completed = events[-1]
    assert isinstance(completed, TurnCompleted)
    assert completed.result.status == 'completed'
    assert completed.result.verification is not None
    assert completed.result.verification.success is True
    assert client.calls[4]['tools'] is not None
    assert '[ForgeCode Stagnation Final Recovery]' not in (
        client.calls[4]['system'] or ''
    )


def test_missing_verification_recovery_focuses_verify_after_package_change(
    tmp_path: Path,
) -> None:
    initialize_git_repository(tmp_path)
    edit = ToolCall(
        0,
        'package-edit',
        'write_file',
        {
            'path': 'package.json',
            'content': '{"scripts":{"build":"echo ok"}}\n',
        },
    )
    empty_src_search = ToolCall(
        0,
        'empty-src-search',
        'find_files',
        {'path': '.', 'pattern': 'src/**/*.ts', 'max_results': 200},
    )
    verify = ToolCall(
        0,
        'package-verify',
        'verify',
        {'command': 'git diff --check'},
    )
    summary = 'Created package.json and verified the diff.'
    client = FakeModelClient(
        response_with_tool(edit),
        text_response('Created package.json.'),
        response_with_tool(empty_src_search),
        text_response('No source files found.'),
        response_with_tool(verify),
        finish_response(
            'package-finish',
            task_kind='change',
            summary=summary,
        ),
    )
    conversation = Conversation(
        client=client,
        registry=create_default_registry(tmp_path),
        max_completion_blocks=2,
        stagnation_warning=1,
        stagnation_limit=2,
    )

    events = collect_turn(conversation, 'Create the project baseline')

    completed = events[-1]
    assert isinstance(completed, TurnCompleted)
    assert completed.result.status == 'completed'
    assert completed.result.text == summary
    assert completed.result.changed_paths == ('package.json',)
    assert completed.result.verification is not None
    assert completed.result.verification.success is True
    verify_recovery_request = client.calls[2]
    verify_recovery_tool_names = {
        definition['name'] for definition in verify_recovery_request['tools']
    }
    assert verify_recovery_tool_names == {'verify'}
    assert '[ForgeCode Verification Recovery]' in (
        verify_recovery_request['system'] or ''
    )


def test_failed_verification_recovery_allows_fix_before_verify(
    tmp_path: Path,
) -> None:
    initialize_git_repository(tmp_path)
    edit = ToolCall(
        0,
        'broken-edit',
        'write_file',
        {
            'path': 'package.json',
            'content': '{"scripts":{"build":"missing-command"}}\n',
        },
    )
    failed_verify = ToolCall(
        0,
        'broken-verify',
        'verify',
        {'command': 'git diff --check --definitely-invalid'},
    )
    fix = ToolCall(
        0,
        'fix-package-script',
        'write_file',
        {
            'path': 'package.json',
            'content': '{"scripts":{"build":"echo ok"}}\n',
        },
    )
    passed_verify = ToolCall(
        0,
        'fixed-verify',
        'verify',
        {'command': 'git diff --check'},
    )
    summary = 'Fixed package.json and verified git diff --check.'
    client = FakeModelClient(
        response_with_tool(edit),
        response_with_tool(failed_verify),
        text_response('Build failed.'),
        response_with_tool(fix),
        response_with_tool(passed_verify),
        finish_response(
            'fixed-finish',
            task_kind='change',
            summary=summary,
        ),
    )
    conversation = Conversation(
        client=client,
        registry=create_default_registry(tmp_path),
        max_completion_blocks=2,
        stagnation_warning=1,
        stagnation_limit=2,
    )

    events = collect_turn(conversation, 'Create and verify the project')

    completed = events[-1]
    assert isinstance(completed, TurnCompleted)
    assert completed.result.status == 'completed'
    assert completed.result.text == summary
    assert completed.result.verification is not None
    assert completed.result.verification.success is True
    failed_recovery_request = client.calls[2]
    failed_recovery_tool_names = {
        definition['name'] for definition in failed_recovery_request['tools']
    }
    assert 'verify' in failed_recovery_tool_names
    assert 'run_command' in failed_recovery_tool_names
    assert 'write_file' in failed_recovery_tool_names
    assert 'list_directory' not in failed_recovery_tool_names
    assert '[ForgeCode Verification Recovery]' in (
        failed_recovery_request['system'] or ''
    )


def test_failed_verification_recovery_limits_reads_then_forces_repair(
    tmp_path: Path,
) -> None:
    initialize_git_repository(tmp_path)
    edit = ToolCall(
        0,
        'complex-package-edit',
        'write_file',
        {
            'path': 'package.json',
            'content': '{"scripts":{"build":"echo ok"}}\n',
        },
    )
    failed_verify = ToolCall(
        0,
        'missing-tsconfig-verify',
        'verify',
        {'command': 'git diff --check --definitely-invalid'},
    )
    one_read = ToolCall(
        0,
        'look-for-tsconfig',
        'find_files',
        {'path': '.', 'pattern': 'tsconfig*.json', 'max_results': 20},
    )
    rejected_second_read = ToolCall(
        0,
        'repeat-tsconfig-search',
        'find_files',
        {'path': '.', 'pattern': 'tsconfig*.json', 'max_results': 20},
    )
    repair = ToolCall(
        0,
        'create-tsconfig',
        'write_file',
        {
            'path': 'tsconfig.json',
            'content': '{"compilerOptions":{"strict":true}}\n',
        },
    )
    passed_verify = ToolCall(
        0,
        'complex-pass-verify',
        'verify',
        {'command': 'git diff --check'},
    )
    summary = 'Created project config and verified the complex baseline.'
    client = FakeModelClient(
        response_with_tool(edit),
        response_with_tool(failed_verify),
        response_with_tool(one_read),
        response_with_tools(rejected_second_read, repair),
        response_with_tool(passed_verify),
        finish_response(
            'complex-finish',
            task_kind='change',
            summary=summary,
        ),
    )
    conversation = Conversation(
        client=client,
        registry=create_default_registry(tmp_path),
        max_completion_blocks=2,
        stagnation_warning=1,
        stagnation_limit=2,
    )

    events = collect_turn(
        conversation,
        '阅读当前目录下的任务文件task.md，明确任务后开始工作',
    )

    completed = events[-1]
    assert isinstance(completed, TurnCompleted)
    assert completed.result.status == 'completed'
    assert completed.result.text == summary
    assert completed.result.changed_paths == (
        'package.json',
        'tsconfig.json',
    )
    assert completed.result.verification is not None
    assert completed.result.verification.success is True

    first_repair_tools = {
        definition['name'] for definition in client.calls[2]['tools']
    }
    assert {'find_files', 'read_file', 'grep'} <= first_repair_tools
    second_repair_tools = {
        definition['name'] for definition in client.calls[3]['tools']
    }
    assert 'find_files' not in second_repair_tools
    assert 'read_file' not in second_repair_tools
    assert 'grep' not in second_repair_tools
    assert {'write_file', 'run_command', 'verify'} <= second_repair_tools
    rejected_results = [
        event.result
        for event in events
        if isinstance(event, ToolExecutionCompleted)
        and event.tool_call.id == 'repeat-tsconfig-search'
    ]
    assert rejected_results
    assert rejected_results[0].success is False
    assert rejected_results[0].error is not None
    assert rejected_results[0].error.code == 'verification_read_limit_reached'


def test_repeated_verification_failure_stops_with_specific_report(
    tmp_path: Path,
) -> None:
    initialize_git_repository(tmp_path)
    edit = ToolCall(
        0,
        'whitespace-edit',
        'write_file',
        {'path': 'sample.txt', 'content': 'bad whitespace  \n'},
    )
    verify = ToolCall(0, 'first-failing-verify', 'verify', {'target': 'diff'})
    repeat = ToolCall(
        0,
        'second-failing-verify',
        'verify',
        {'target': 'diff'},
    )
    client = FakeModelClient(
        response_with_tool(edit),
        response_with_tool(verify),
        response_with_tool(repeat),
    )
    conversation = Conversation(
        client=client,
        registry=create_default_registry(tmp_path),
        max_completion_blocks=2,
        stagnation_warning=1,
        stagnation_limit=2,
    )

    events = collect_turn(conversation, 'Create and verify package metadata')

    completed = events[-1]
    assert isinstance(completed, TurnCompleted)
    assert completed.result.status == 'stuck'
    assert completed.result.text == (
        'Verification Recovery stopped after the same verification failure '
        'repeated.'
    )
    assert completed.result.verification is not None
    assert completed.result.verification.status == 'failed'


def test_verified_change_stagnation_allows_final_summary_recovery(
    tmp_path: Path,
) -> None:
    initialize_git_repository(tmp_path)
    edit = ToolCall(
        0,
        'convergence-edit',
        'replace_text',
        {
            'path': 'sample.txt',
            'old_text': 'old\n',
            'new_text': 'new\n',
        },
    )
    verify = ToolCall(
        0,
        'convergence-verify',
        'verify',
        {'command': 'git diff --check'},
    )
    initial_diff = ToolCall(0, 'initial-diff', 'git_diff', {})
    redundant_diffs = [
        ToolCall(0, f'redundant-diff-{index}', 'git_diff', {})
        for index in range(1, 9)
    ]
    summary = 'Updated sample.txt and verified it with git diff --check.'
    client = FakeModelClient(
        response_with_tool(edit),
        response_with_tool(verify),
        text_response(summary),
    )
    conversation = Conversation(
        client=client,
        registry=create_default_registry(tmp_path),
        task_policy=TaskPolicy(require_verification=True),
    )

    events = collect_turn(conversation, 'Change and verify sample.txt')

    completed = events[-1]
    assert isinstance(completed, TurnCompleted)
    assert completed.result.status == 'completed'
    assert completed.result.model_calls == 3
    assert completed.result.text == summary
    assert completed.result.changed_paths == ('sample.txt',)
    assert completed.result.verification is not None
    assert completed.result.verification.success is True
    assert len(client.calls) == 3
    final_request = (
        (client.calls[-1]['system'] or '')
        + str(client.calls[-1]['messages'])
    )
    assert '[ForgeCode Finalization Recovery]' in final_request
    assert {tool['name'] for tool in client.calls[-1]['tools']} == {
        'finish_task'
    }
    assert 'Runtime Tool Availability' not in (
        client.calls[-1]['system'] or ''
    )


def test_unverified_change_stagnation_recovers_before_completion(
    tmp_path: Path,
) -> None:
    initialize_git_repository(tmp_path)
    edit = ToolCall(
        0,
        'unverified-convergence-edit',
        'replace_text',
        {
            'path': 'sample.txt',
            'old_text': 'old\n',
            'new_text': 'new\n',
        },
    )
    initial_diff = ToolCall(0, 'unverified-initial-diff', 'git_diff', {})
    redundant_diffs = [
        ToolCall(0, f'unverified-redundant-diff-{index}', 'git_diff', {})
        for index in range(1, 9)
    ]
    verify = ToolCall(
        0,
        'unverified-recovery-verify',
        'verify',
        {'command': 'git diff --check'},
    )
    premature = 'Updated sample.txt; no verification was required or run.'
    summary = 'Updated and verified sample.txt after recovery.'
    client = FakeModelClient(
        response_with_tool(edit),
        response_with_tool(initial_diff),
        *(response_with_tool(call) for call in redundant_diffs),
        text_response(premature),
        response_with_tool(verify),
        finish_response(
            'unverified-recovery-finish',
            task_kind='change',
            summary=summary,
        ),
    )
    conversation = Conversation(
        client=client,
        registry=create_default_registry(tmp_path),
    )

    events = collect_turn(conversation, 'Change sample.txt')

    completed = events[-1]
    assert isinstance(completed, TurnCompleted)
    assert completed.result.status == 'completed'
    assert completed.result.text == summary
    assert completed.result.changed_paths == ('sample.txt',)
    assert completed.result.verification is not None
    assert completed.result.verification.success is True
    assert any(
        'has not been verified' in reason
        for reason in completed.result.completion_reasons
    ) is False
    assert any(
        isinstance(event, CompletionBlocked)
        and any('verify tool' in reason for reason in event.reasons)
        for event in events
    )
    assert client.calls[-2]['tools'] is not None


def test_novel_repository_evidence_cannot_extend_completion_ready_loop(
    tmp_path: Path,
) -> None:
    for index in range(1, 9):
        path = tmp_path / 'notes' / f'evidence-{index}.txt'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f'evidence {index}\n', encoding='utf-8')
    initialize_git_repository(tmp_path)
    edit = ToolCall(
        0,
        'novel-ready-edit',
        'replace_text',
        {
            'path': 'sample.txt',
            'old_text': 'old\n',
            'new_text': 'new\n',
        },
    )
    verify = ToolCall(
        0,
        'novel-ready-verify',
        'verify',
        {'command': 'git diff --check'},
    )
    diff = ToolCall(0, 'novel-ready-diff', 'git_diff', {})
    novel_reads = [
        ToolCall(
            0,
            f'novel-ready-read-{index}',
            'read_file',
            {'path': f'notes/evidence-{index}.txt'},
        )
        for index in range(1, 9)
    ]
    summary = 'Updated and verified sample.txt.'
    client = FakeModelClient(
        response_with_tool(edit),
        response_with_tool(verify),
        text_response(summary),
    )
    conversation = Conversation(
        client=client,
        registry=create_default_registry(tmp_path),
        task_policy=TaskPolicy(require_verification=True),
    )

    events = collect_turn(conversation, 'Change and verify sample.txt')

    completed = events[-1]
    assert isinstance(completed, TurnCompleted)
    assert completed.result.status == 'completed'
    assert completed.result.model_calls == 3
    assert completed.result.text == summary
    assert {tool['name'] for tool in client.calls[-1]['tools']} == {
        'finish_task'
    }
    assert '[ForgeCode Finalization Recovery]' in (
        client.calls[-1]['system'] or ''
    )


def test_finalization_recovery_stops_after_one_more_redundant_diff(
    tmp_path: Path,
) -> None:
    initialize_git_repository(tmp_path)
    edit = ToolCall(
        0,
        'finalization-edit',
        'replace_text',
        {
            'path': 'sample.txt',
            'old_text': 'old\n',
            'new_text': 'new\n',
        },
    )
    verify = ToolCall(
        0,
        'finalization-verify',
        'verify',
        {'command': 'git diff --check'},
    )
    initial_diff = ToolCall(0, 'finalization-diff', 'git_diff', {})
    redundant_diffs = [
        ToolCall(0, f'finalization-repeat-{index}', 'git_diff', {})
        for index in range(1, 10)
    ]
    client = FakeModelClient(
        response_with_tool(edit),
        response_with_tool(verify),
        response_with_tool(initial_diff),
        *(response_with_tool(call) for call in redundant_diffs),
    )
    conversation = Conversation(
        client=client,
        registry=create_default_registry(tmp_path),
        task_policy=TaskPolicy(require_verification=True),
    )

    events = collect_turn(conversation, 'Change and verify sample.txt')

    completed = events[-1]
    assert isinstance(completed, TurnCompleted)
    assert completed.result.status == 'stuck'
    assert completed.result.model_calls == 3
    assert 'finalization recovery' in completed.result.text.casefold()
    assert completed.result.changed_paths == ('sample.txt',)
    assert completed.result.verification is not None
    assert completed.result.verification.success is True
    assert len(client.calls) == 3
    recovery_request = (
        (client.calls[-1]['system'] or '')
        + str(client.calls[-1]['messages'])
    )
    assert '[ForgeCode Finalization Recovery]' in recovery_request
    assert {tool['name'] for tool in client.calls[-1]['tools']} == {
        'finish_task'
    }


def test_unfinished_explicit_plan_does_not_enter_finalization_recovery(
    tmp_path: Path,
) -> None:
    initialize_git_repository(tmp_path)
    plan = ToolCall(
        0,
        'unfinished-plan',
        'task_plan',
        {'steps': ['Edit sample', 'Complete remaining work']},
    )
    edit = ToolCall(
        0,
        'unfinished-plan-edit',
        'replace_text',
        {
            'path': 'sample.txt',
            'old_text': 'old\n',
            'new_text': 'new\n',
        },
    )
    verify = ToolCall(
        0,
        'unfinished-plan-verify',
        'verify',
        {'command': 'git diff --check'},
    )
    diff = ToolCall(0, 'unfinished-plan-diff', 'git_diff', {})
    redundant_diffs = [
        ToolCall(0, f'unfinished-plan-repeat-{index}', 'git_diff', {})
        for index in range(1, 9)
    ]
    summary = 'Edited and verified sample.txt, but the explicit plan remains incomplete.'
    client = FakeModelClient(
        response_with_tool(plan),
        response_with_tool(edit),
        response_with_tool(verify),
        response_with_tool(diff),
        *(response_with_tool(call) for call in redundant_diffs),
        text_response(summary),
    )
    conversation = Conversation(
        client=client,
        registry=create_default_registry(tmp_path),
    )

    events = collect_turn(conversation, 'Complete both planned steps')

    completed = events[-1]
    assert isinstance(completed, TurnCompleted)
    assert completed.result.status == 'completed'
    assert completed.result.text == summary
    assert completed.result.model_calls == 13
    assert all(call['tools'] is not None for call in client.calls)
    assert all(
        '[ForgeCode Finalization Recovery]' not in (call['system'] or '')
        for call in client.calls
    )


def test_runtime_tells_model_that_request_tools_are_available(
    tmp_path: Path,
) -> None:
    initialize_git_repository(tmp_path)
    client = FakeModelClient(
        text_response('I will decide how to proceed.'),
        finish_response(
            'finish_answer',
            task_kind='answer',
            summary='I decided to answer.',
        ),
    )
    conversation = Conversation(
        client=client,
        registry=create_default_registry(tmp_path),
    )

    events = collect_turn(conversation, 'Describe the tools in this request')

    completed = events[-1]
    assert isinstance(completed, TurnCompleted)
    assert completed.result.status == 'completed'
    assert len(client.calls) == 1
    assert 'tools included with this model request are currently available' in (
        client.calls[0]['system'] or ''
    )


def test_malformed_tool_arguments_recover_without_pausing_tools(
    tmp_path: Path,
) -> None:
    initialize_git_repository(tmp_path)
    malformed = ToolCall(
        0,
        'toolu_bad_list',
        'list_directory',
        {'path': '.', '}}{': '?'},
    )
    corrected = ToolCall(
        0,
        'toolu_good_list',
        'list_directory',
        {'path': '.'},
    )
    client = FakeModelClient(
        response_with_tool(malformed),
        response_with_tool(corrected),
        text_response('Inspected the repository root.'),
    )
    conversation = Conversation(
        client=client,
        registry=create_default_registry(tmp_path),
        stagnation_warning=1,
        stagnation_limit=3,
    )

    events = collect_turn(conversation, 'Inspect the repository')

    tool_events = [
        event for event in events
        if isinstance(event, ToolExecutionCompleted)
    ]
    assert tool_events[0].result.error is not None
    assert tool_events[0].result.error.code == 'invalid_arguments'
    assert tool_events[1].result.success is True
    assert all(call['tools'] is not None for call in client.calls)
    assert all(
        'Repository action tools are paused' not in (call['system'] or '')
        for call in client.calls
    )
    assert isinstance(events[-1], TurnCompleted)
    assert events[-1].result.status == 'completed'


def test_repeated_exact_cache_hit_is_rejected_as_non_progress(
    tmp_path: Path,
) -> None:
    initialize_git_repository(tmp_path)
    first = ToolCall(
        0,
        'list-root-first',
        'list_directory',
        {'path': '.', 'max_results': 200},
    )
    second = ToolCall(
        0,
        'list-root-second',
        'list_directory',
        {'path': '.', 'max_results': 200},
    )
    third = ToolCall(
        0,
        'list-root-third',
        'list_directory',
        {'path': '.', 'max_results': 200},
    )
    client = FakeModelClient(
        response_with_tool(first),
        response_with_tool(second),
        response_with_tool(third),
        text_response('Used existing directory evidence.'),
    )
    conversation = Conversation(
        client=client,
        registry=create_default_registry(tmp_path),
        stagnation_warning=10,
        stagnation_limit=20,
    )

    events = collect_turn(conversation, 'Inspect the repository root')

    results = [
        event
        for event in events
        if isinstance(event, ToolExecutionCompleted)
    ]
    assert results[1].result.success is True
    assert results[1].result.metadata['cache_hit'] is True
    assert results[2].result.success is False
    assert results[2].result.error is not None
    assert results[2].result.error.code == 'redundant_cached_tool_call'
    completed = events[-1]
    assert isinstance(completed, TurnCompleted)
    assert completed.result.status == 'completed'


def test_invalid_grep_regex_recovers_as_tool_protocol_failure(
    tmp_path: Path,
) -> None:
    initialize_git_repository(tmp_path)
    invalid = ToolCall(
        0,
        'toolu_invalid_regex',
        'grep',
        {'path': 'sample.txt', 'pattern': 'old('},
    )
    corrected = ToolCall(
        0,
        'toolu_literal_search',
        'grep',
        {'path': 'sample.txt', 'pattern': 'old', 'regex': False},
    )
    client = FakeModelClient(
        response_with_tool(invalid),
        response_with_tool(corrected),
        finish_response(
            'toolu_regex_finish',
            task_kind='inspection',
            summary='Found the literal text.',
        ),
    )
    conversation = Conversation(
        client=client,
        registry=create_default_registry(tmp_path),
    )

    events = collect_turn(conversation, 'Inspect sample.txt for literal text')

    tool_events = [
        event
        for event in events
        if isinstance(event, ToolExecutionCompleted)
    ]
    assert tool_events[0].result.success is False
    assert tool_events[0].result.error is not None
    assert tool_events[0].result.error.code == 'invalid_pattern'
    assert tool_events[1].result.success is True
    assert all(call['tools'] is not None for call in client.calls)
    assert isinstance(events[-1], TurnCompleted)
    assert events[-1].result.status == 'completed'


def test_inspection_finish_requires_and_accepts_repository_evidence(
    tmp_path: Path,
) -> None:
    initialize_git_repository(tmp_path)
    read = ToolCall(
        0,
        'toolu_inspect_read',
        'read_file',
        {'path': 'sample.txt'},
    )
    client = FakeModelClient(
        finish_response('finish_without_evidence', task_kind='inspection'),
        response_with_tool(read),
        finish_response(
            'finish_with_evidence',
            task_kind='inspection',
            summary='sample.txt contains the inspected value.',
        ),
    )
    conversation = Conversation(
        client=client,
        registry=create_default_registry(tmp_path),
    )

    events = collect_turn(conversation, 'Inspect sample.txt')

    blocks = [event for event in events if isinstance(event, CompletionBlocked)]
    assert len(blocks) == 1
    assert 'requires repository evidence' in blocks[0].reasons[0]
    completed = events[-1]
    assert isinstance(completed, TurnCompleted)
    assert completed.result.status == 'completed'
    assert completed.result.text == 'sample.txt contains the inspected value.'


def test_finish_task_must_be_called_alone(tmp_path: Path) -> None:
    initialize_git_repository(tmp_path)
    read = ToolCall(
        0,
        'toolu_mixed_read',
        'read_file',
        {'path': 'sample.txt'},
    )
    finish = ToolCall(
        1,
        'toolu_mixed_finish',
        'finish_task',
        {
            'task_kind': 'inspection',
            'status': 'completed',
            'summary': 'Inspected.',
            'blocked_reasons': [],
        },
    )
    mixed_response = [
        ModelUsageUpdate(usage=TokenUsage(10, 0)),
        ModelToolCallCompleted(tool_call=read),
        ModelToolCallCompleted(tool_call=finish),
        ModelUsageUpdate(usage=TokenUsage(10, 2)),
    ]
    client = FakeModelClient(
        mixed_response,
        finish_response(
            'toolu_finish_alone',
            task_kind='inspection',
            summary='Inspected sample.txt.',
        ),
    )
    conversation = Conversation(
        client=client,
        registry=create_default_registry(tmp_path),
    )

    events = collect_turn(conversation, 'Inspect sample.txt')

    tool_events = [
        event for event in events
        if isinstance(event, ToolExecutionCompleted)
    ]
    mixed_finish = next(
        event for event in tool_events
        if event.tool_call.id == 'toolu_mixed_finish'
    )
    assert mixed_finish.result.success is False
    assert mixed_finish.result.error is not None
    assert mixed_finish.result.error.code == 'finish_must_be_alone'
    assert isinstance(events[-1], TurnCompleted)
    assert events[-1].result.status == 'completed'


def test_agent_loop_stops_after_three_completion_rejections(
    tmp_path: Path,
) -> None:
    initialize_git_repository(tmp_path)
    edit = ToolCall(
        0,
        'toolu_edit',
        'replace_text',
        {
            'path': 'sample.txt',
            'old_text': 'old\n',
            'new_text': 'new\n',
        },
    )
    client = FakeModelClient(
        response_with_tool(edit),
        finish_response('finish_once', task_kind='change'),
        finish_response('finish_twice', task_kind='change'),
        finish_response('finish_three', task_kind='change'),
    )
    conversation = Conversation(
        client=client,
        registry=create_default_registry(tmp_path),
        task_policy=TaskPolicy(require_verification=True),
    )
    events = collect_turn(conversation, 'Change sample.txt')

    blocks = [item for item in events if isinstance(item, CompletionBlocked)]
    assert [item.attempt for item in blocks] == [1, 2, 3]
    completed = events[-1]
    assert isinstance(completed, TurnCompleted)
    assert completed.result.status == 'stuck'
    assert completed.result.completion_reasons
    assert conversation.task_manager.active is not None
    assert conversation.task_manager.active.status == 'stuck'


def test_false_blocker_is_rejected_and_recovery_keeps_all_tools(
    tmp_path: Path,
) -> None:
    initialize_git_repository(tmp_path)
    searches = [
        ToolCall(
            0,
            f'toolu_find_{index}',
            'find_files',
            {'path': '.', 'pattern': pattern},
        )
        for index, pattern in enumerate(('missing-a', 'missing-b'), start=1)
    ]
    recovery_searches = [
        ToolCall(
            0,
            f'toolu_recovery_{index}',
            'find_files',
            {'path': '.', 'pattern': f'still-missing-{index}'},
        )
        for index in range(1, 5)
    ]
    client = FakeModelClient(
        *(response_with_tool(call) for call in searches),
        finish_response(
            'finish_blocked',
            task_kind='change',
            status='blocked',
            summary='I could not complete the requested code change.',
            blocked_reasons=['No applicable source evidence was found.'],
        ),
        *(response_with_tool(call) for call in recovery_searches),
    )
    conversation = Conversation(
        client=client,
        registry=create_default_registry(tmp_path),
        task_policy=TaskPolicy(
            require_changes=True,
            require_verification=True,
        ),
        stagnation_warning=2,
        stagnation_limit=4,
    )

    events = collect_turn(conversation, 'Change and verify the game')

    completed = events[-1]
    assert isinstance(completed, TurnCompleted)
    assert completed.result.status == 'stuck'
    assert len(client.calls) >= 6
    assert all(call['tools'] is not None for call in client.calls)
    finish_event = next(
        event for event in events
        if isinstance(event, ToolExecutionCompleted)
        and event.tool_call.name == 'finish_task'
    )
    assert finish_event.result.error is not None
    assert finish_event.result.error.code == 'finish_rejected'


def test_empty_recovery_response_returns_stuck_turn(
    tmp_path: Path,
) -> None:
    initialize_git_repository(tmp_path)
    searches = [
        ToolCall(
            0,
            f'toolu_empty_{index}',
            'find_files',
            {'path': '.', 'pattern': pattern},
        )
        for index, pattern in enumerate(('none-a', 'none-b'), start=1)
    ]
    client = FakeModelClient(
        *(response_with_tool(call) for call in searches),
        [ModelUsageUpdate(usage=TokenUsage(10, 0))],
    )
    conversation = Conversation(
        client=client,
        registry=create_default_registry(tmp_path),
        stagnation_warning=2,
        stagnation_limit=4,
    )

    events = collect_turn(conversation, 'Inspect missing files')

    completed = events[-1]
    assert isinstance(completed, TurnCompleted)
    assert completed.result.status == 'stuck'
    assert 'no usable answer' in completed.result.text
    assert len(client.calls) == 3


def test_empty_response_after_completion_rejection_is_stuck(
    tmp_path: Path,
) -> None:
    initialize_git_repository(tmp_path)
    edit = ToolCall(
        0,
        'toolu_edit_empty',
        'replace_text',
        {
            'path': 'sample.txt',
            'old_text': 'old\n',
            'new_text': 'new\n',
        },
    )
    client = FakeModelClient(
        response_with_tool(edit),
        finish_response('finish_unverified', task_kind='change'),
        [ModelUsageUpdate(usage=TokenUsage(10, 0))],
    )
    conversation = Conversation(
        client=client,
        registry=create_default_registry(tmp_path),
        task_policy=TaskPolicy(require_verification=True),
    )

    events = collect_turn(conversation, 'Change and verify sample.txt')

    completed = events[-1]
    assert isinstance(completed, TurnCompleted)
    assert completed.result.status == 'stuck'
    assert any(
        'has not been verified' in reason
        for reason in completed.result.completion_reasons
    )
    assert len(client.calls) == 3
