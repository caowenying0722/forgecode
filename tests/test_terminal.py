'''Tests for the Rich terminal presentation.'''

from io import StringIO
from pathlib import Path

from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document
from rich.console import Console

from forge.runtime.state import (
    ContextCompacted,
    TokenUsage,
    ToolCall,
    TurnResult,
    VerificationEvidence,
)
from forge.runtime.agent_state import AgentPhase
from forge.permissions.policy import ApprovalResponse, PermissionRequest
from forge.terminal import (
    EncodingSafeTextIO,
    ForgePromptCompleter,
    SLASH_COMMAND_COMPLETER,
    TerminalUI,
    repair_input_text,
    summarize_diagnostic,
    summarize_tool_arguments,
    streaming_preview,
    token_usage_summary,
)
from forge.tools.base import ToolResult


class FakePromptSession:
    def __init__(self, response: str) -> None:
        self.response = response
        self.messages: list[object] = []
        self.kwargs: list[dict[str, object]] = []

    def prompt(self, message: object = '', **kwargs: object) -> str:
        self.messages.append(message)
        self.kwargs.append(kwargs)
        return self.response


def terminal_with_output() -> tuple[TerminalUI, StringIO]:
    output = StringIO()
    console = Console(
        file=output,
        force_terminal=False,
        width=100,
    )
    return TerminalUI(console=console), output


def test_terminal_preserves_multiline_prompt_from_interactive_session() -> None:
    output = StringIO()
    console = Console(file=output, force_terminal=True, width=100)
    prompt_session = FakePromptSession('first line\nsecond line\nthird line')
    terminal = TerminalUI(
        console=console,
        prompt_session=prompt_session,
    )

    prompt = terminal.read_prompt()

    assert prompt == 'first line\nsecond line\nthird line'
    assert len(prompt_session.messages) == 1
    assert prompt_session.kwargs[0]['completer'] is terminal.prompt_completer
    assert prompt_session.kwargs[0]['complete_while_typing'] is True


def test_terminal_repairs_invalid_surrogate_prompt_text() -> None:
    assert repair_input_text('fix\udc80bug') == 'fix?bug'


def test_terminal_repairs_windows_utf8_gbk_mojibake_prompt_text() -> None:
    original = '根据 task.md 继续完善项目功能'
    mojibake = original.encode('utf-8').decode('gb18030', errors='replace')

    assert mojibake != original
    assert repair_input_text(mojibake) == original


def test_terminal_repairs_weather_question_mojibake_prompt_text() -> None:
    original = '厦门今天天气，能联网就用 MCP 查一下，简短回答。'
    mojibake = original.encode('utf-8').decode('gb18030', errors='replace')

    assert mojibake != original
    assert repair_input_text(mojibake).startswith('厦门今天天气，能联网')


def test_terminal_keeps_normal_chinese_prompt_text() -> None:
    original = '根据 task.md 继续完善项目功能'

    assert repair_input_text(original) == original


def test_encoding_safe_text_io_replaces_unencodable_output() -> None:
    class FakeOutput:
        encoding = 'gbk'

        def __init__(self) -> None:
            self.value = ''

        def write(self, text: str) -> int:
            self.value += text
            return len(text)

        def flush(self) -> None:
            pass

    output = FakeOutput()
    stream = EncodingSafeTextIO(output)

    stream.write('step ↳ done')

    assert output.value == 'step ? done'


def test_terminal_rebinds_completion_on_every_prompt(tmp_path: Path) -> None:
    output = StringIO()
    console = Console(file=output, force_terminal=True, width=100)
    prompt_session = FakePromptSession('ok')
    terminal = TerminalUI(
        console=console,
        prompt_session=prompt_session,
        workspace_root=tmp_path,
    )

    terminal.read_prompt()
    created = tmp_path / 'later.ts'
    created.write_text('later\n', encoding='utf-8')
    terminal.read_prompt()

    assert len(prompt_session.kwargs) == 2
    assert all(
        kwargs['completer'] is terminal.prompt_completer
        for kwargs in prompt_session.kwargs
    )
    completions = list(
        terminal.prompt_completer.get_completions(
            Document(text='@lat', cursor_position=4),
            CompleteEvent(completion_requested=True),
        )
    )
    assert [completion.display_text for completion in completions] == [
        'later.ts'
    ]


def completions_for(text: str) -> list[object]:
    return list(
        SLASH_COMMAND_COMPLETER.get_completions(
            Document(text=text, cursor_position=len(text)),
            CompleteEvent(completion_requested=True),
        )
    )


def test_slash_opens_command_completion_menu() -> None:
    completions = completions_for('/')

    usages = [completion.display_text for completion in completions]
    descriptions = [completion.display_meta_text for completion in completions]

    assert '/context' in usages
    assert '/resume' in usages
    assert '/resume session-id' in usages
    assert '/sessions' in usages
    assert '/mode' in usages
    assert '/mode auto|plan|code' in usages
    assert '/plan' in usages
    assert '/code' in usages
    assert '/permissions' in usages
    assert '/permission' not in usages
    assert '/mcp' in usages
    assert '/hooks' in usages
    assert '/todo' in usages
    assert '/exit' in usages
    assert '/remember name | content' in usages
    assert '/memory consolidate' in usages
    assert '/task' in usages
    assert '/task history' in usages
    assert '/task resume task-id' in usages
    assert '查看当前上下文统计' in descriptions


def test_slash_completion_filters_and_replaces_current_input() -> None:
    completions = completions_for('/memory s')

    assert [completion.text for completion in completions] == [
        '/memory show '
    ]
    assert completions[0].start_position == -len('/memory s')


def test_normal_prompt_does_not_offer_slash_commands() -> None:
    assert completions_for('fix this bug') == []


def workspace_completions_for(
    root: Path,
    text: str,
) -> list[object]:
    completer = ForgePromptCompleter(root)
    return list(
        completer.get_completions(
            Document(text=text, cursor_position=len(text)),
            CompleteEvent(completion_requested=True),
        )
    )


def test_at_opens_protected_workspace_file_completion(tmp_path: Path) -> None:
    (tmp_path / 'README.md').write_text('readme\n', encoding='utf-8')
    source = tmp_path / 'src' / 'game'
    source.mkdir(parents=True)
    (source / 'player.ts').write_text('player\n', encoding='utf-8')
    (tmp_path / '.env').write_text('SECRET=value\n', encoding='utf-8')
    hidden = tmp_path / '.forge'
    hidden.mkdir()
    (hidden / 'session.json').write_text('{}\n', encoding='utf-8')
    dependency = tmp_path / 'node_modules' / 'pkg'
    dependency.mkdir(parents=True)
    (dependency / 'index.js').write_text('module\n', encoding='utf-8')

    completions = workspace_completions_for(tmp_path, 'Please inspect @')
    paths = [completion.display_text for completion in completions]

    assert paths == ['README.md', 'src/game/player.ts']
    assert all(
        completion.display_meta_text == 'workspace file'
        for completion in completions
    )


def test_at_completion_fuzzy_filters_and_replaces_only_active_token(
    tmp_path: Path,
) -> None:
    target = tmp_path / 'src' / 'game' / 'player_controller.ts'
    target.parent.mkdir(parents=True)
    target.write_text('player\n', encoding='utf-8')
    (tmp_path / 'README.md').write_text('readme\n', encoding='utf-8')

    completions = workspace_completions_for(
        tmp_path,
        'Compare @README.md with @sgpc',
    )

    assert [completion.text for completion in completions] == [
        '@src/game/player_controller.ts '
    ]
    assert completions[0].start_position == -len('@sgpc')


def test_at_completion_does_not_trigger_inside_email_address(
    tmp_path: Path,
) -> None:
    (tmp_path / 'mail.txt').write_text('mail\n', encoding='utf-8')

    assert workspace_completions_for(tmp_path, 'email user@ma') == []


def test_workspace_file_index_refreshes_between_prompts(tmp_path: Path) -> None:
    completer = ForgePromptCompleter(tmp_path)
    created = tmp_path / 'new-file.ts'
    created.write_text('new\n', encoding='utf-8')

    completer.refresh()
    completions = list(
        completer.get_completions(
            Document(text='@new', cursor_position=4),
            CompleteEvent(completion_requested=True),
        )
    )

    assert [completion.display_text for completion in completions] == [
        'new-file.ts'
    ]


def test_permission_picker_uses_injected_selector() -> None:
    output = StringIO()
    console = Console(file=output, force_terminal=False, width=100)
    terminal = TerminalUI(
        console=console,
        permission_selector=lambda current: (
            'auto' if current == 'supervised' else None
        ),
    )

    assert terminal.select_permission_mode('supervised') == 'auto'


def test_permission_picker_uses_inline_completion_menu() -> None:
    output = StringIO()
    console = Console(
        file=output,
        force_terminal=True,
        width=100,
    )
    terminal = TerminalUI(
        console=console,
        prompt_session=FakePromptSession('plan'),
    )

    selected = terminal.select_permission_mode('supervised')

    assert selected == 'plan'


def test_permission_request_dialog_uses_injected_selector() -> None:
    output = StringIO()
    console = Console(file=output, force_terminal=False, width=100)
    observed: list[PermissionRequest] = []

    def approve(request: PermissionRequest) -> ApprovalResponse:
        observed.append(request)
        return ApprovalResponse('allow_session')

    terminal = TerminalUI(console=console, approval_selector=approve)
    request = PermissionRequest(
        'run_command',
        'process.exec',
        'low',
        ('tests',),
        preview='pytest',
    )

    with terminal.stream_response() as response:
        decision = response.request_permission(request)

    assert decision == ApprovalResponse('allow_session')
    assert observed == [request]


def test_permission_request_dialog_renders_scoped_choices() -> None:
    output = StringIO()
    terminal = TerminalUI(
        console=Console(file=output, force_terminal=True, width=100),
        prompt_session=FakePromptSession(''),
    )

    response = terminal.select_permission(
        PermissionRequest(
            'run_command',
            'dependency.install',
            'high',
            ('package.json',),
            'The command installs dependencies.',
            'npm install',
        )
    )

    rendered = output.getvalue()
    assert response.choice == 'deny'
    assert 'Permission Required' in rendered
    assert 'Install dependencies' in rendered
    assert 'risk: High' in rendered
    assert 'package.json' in rendered
    assert 'Remember for this project' in rendered


def test_delete_permission_dialog_does_not_offer_project_memory() -> None:
    output = StringIO()
    terminal = TerminalUI(
        console=Console(file=output, force_terminal=True, width=100),
        prompt_session=FakePromptSession(''),
    )

    terminal.select_permission(
        PermissionRequest(
            'apply_patch',
            'file.delete',
            'high',
            ('src/old.py', 'src/dead.py'),
            'The patch deletes files.',
        )
    )

    rendered = output.getvalue()
    assert 'Delete files' in rendered
    assert 'Remember for this project' not in rendered


def test_terminal_renders_session_header_and_markdown_response() -> None:
    terminal, output = terminal_with_output()
    usage = TokenUsage(input_tokens=1200, output_tokens=34)

    terminal.show_welcome('test-model')
    with terminal.stream_response() as response:
        response.update_usage(
            TokenUsage(input_tokens=1200, output_tokens=0)
        )
        response.append_text('**Hello** ')
        response.append_text('from ForgeCode')
        response.update_usage(usage)
        response.complete(
            TurnResult(
                text='**Hello** from ForgeCode',
                usage=usage,
            )
        )
    terminal.show_goodbye()

    rendered = output.getvalue()
    assert 'ForgeCode v0.1.0' in rendered
    assert 'test-model' in rendered
    assert 'Esc interrupt · Ctrl+C exit' in rendered
    assert 'Hello from ForgeCode' in rendered
    assert rendered.count('Hello from ForgeCode') == 1
    assert 'input 1,200' in rendered
    assert 'output 34' in rendered
    assert 'total 1,234' in rendered
    assert 'Session ended.' in rendered


def test_terminal_renders_context_compaction_notice() -> None:
    terminal, output = terminal_with_output()
    with terminal.stream_response() as response:
        response.compact_context(
            ContextCompacted(
                before_characters=10_000,
                after_characters=1_000,
                transcript_path='.forge/context/transcripts/test.jsonl',
            )
        )
        response.complete(
            TurnResult(
                text='Done',
                usage=TokenUsage(input_tokens=10, output_tokens=2),
            )
        )

    rendered = output.getvalue()
    assert 'Context compacted' in rendered
    assert '10,000 -> 1,000 characters' in rendered
    assert '.forge/context/transcripts/test.jsonl' in rendered


def test_terminal_renders_cache_token_details() -> None:
    terminal, output = terminal_with_output()

    terminal.console.print(
        token_usage_summary(
            TokenUsage(
                input_tokens=100,
                output_tokens=20,
                cache_creation_input_tokens=30,
                cache_read_input_tokens=40,
            ),
            streaming=False,
        )
    )

    rendered = output.getvalue()
    assert 'tokens' in rendered
    assert 'input 170' in rendered
    assert 'total 190' in rendered
    assert 'cache read 40' in rendered
    assert 'cache write 30' in rendered


def test_terminal_separates_last_request_from_turn_cumulative_usage() -> None:
    terminal, output = terminal_with_output()

    terminal.console.print(
        token_usage_summary(
            TokenUsage(
                input_tokens=313_367,
                output_tokens=10_806,
                cache_read_input_tokens=50_560,
            ),
            streaming=False,
            request_usage=TokenUsage(
                input_tokens=9_000,
                output_tokens=300,
                cache_read_input_tokens=2_000,
            ),
            model_calls=26,
        )
    )

    rendered = output.getvalue()
    assert 'turn cumulative' in rendered
    assert 'input 363,927' in rendered
    assert 'last request  input 11,000' in rendered
    assert 'output 300' in rendered
    assert '26 model calls' in rendered


def test_terminal_labels_zero_stream_usage_as_waiting() -> None:
    terminal, output = terminal_with_output()

    terminal.console.print(
        token_usage_summary(
            TokenUsage(input_tokens=220_371, output_tokens=21_273),
            streaming=True,
            request_usage=TokenUsage(input_tokens=0, output_tokens=0),
            model_calls=15,
        )
    )

    rendered = output.getvalue()
    assert 'waiting for provider usage' in rendered
    assert 'last request  input 0' not in rendered
    assert '15 model calls' in rendered


def test_terminal_renders_tool_arguments_and_result_status() -> None:
    terminal, output = terminal_with_output()
    successful_call = ToolCall(
        index=0,
        id='toolu_read',
        name='read_file',
        arguments={'path': 'README.md'},
    )
    failed_call = ToolCall(
        index=1,
        id='toolu_shell',
        name='run_command',
        arguments={'command': 'pytest'},
    )

    with terminal.stream_response() as response:
        response.start_tool(successful_call)
        response.complete_tool(
            successful_call,
            ToolResult.ok('Read 20 lines.'),
        )
        response.start_tool(failed_call)
        response.complete_tool(
            failed_call,
            ToolResult.fail(
                'command_failed',
                'Command exited with code 1.',
                content='pytest: assertion failed at test_game.py:42',
            ),
        )

    rendered = output.getvalue()
    assert 'read_file {"path": "README.md"}' in rendered
    assert '✓' in rendered
    assert 'Read 20 lines.' in rendered
    assert 'run_command {"command": "pytest"}' in rendered
    assert '×' in rendered
    assert 'Command exited with code 1.' in rendered
    assert 'pytest: assertion failed at test_game.py:42' in rendered


def test_tool_argument_summary_redacts_large_edit_content() -> None:
    rendered = summarize_tool_arguments(
        {
            'path': 'src/app.py',
            'old_text': 'secret old text\n' * 20,
            'new_text': 'secret new text\n' * 20,
            'token': 'sk-test-token',
        }
    )

    assert 'src/app.py' in rendered
    assert 'secret old text' not in rendered
    assert 'secret new text' not in rendered
    assert 'sk-test-token' not in rendered
    assert '<redacted' in rendered


def test_tool_argument_summary_keeps_long_command_head_and_tail() -> None:
    command = 'pytest ' + ('very-long-option ' * 20) + 'tests/test_app.py'

    rendered = summarize_tool_arguments({'command': command})

    assert 'pytest' in rendered
    assert 'tests/test_app.py' in rendered
    assert len(rendered) <= 120


def test_diagnostic_summary_keeps_head_tail_without_flooding_terminal() -> None:
    diagnostic = '\n'.join(f'line {index}' for index in range(30))

    rendered = summarize_diagnostic(
        diagnostic,
        max_lines=6,
        max_characters=200,
    )

    assert 'line 0' in rendered
    assert 'line 29' in rendered
    assert 'lines omitted' in rendered
    assert 'diagnostic shortened' in rendered
    assert 'line 15' not in rendered


def test_terminal_renders_friendly_phase_status() -> None:
    terminal, output = terminal_with_output()

    with terminal.stream_response() as response:
        response.update_phase(
            AgentPhase.EXECUTING_TOOLS,
            'executing_tool:read_file',
        )

    rendered = output.getvalue()
    assert '正在查看项目' in rendered
    assert 'executing_tool:read_file' not in rendered


def test_terminal_renders_transaction_cache_annotation() -> None:
    terminal, output = terminal_with_output()
    tool_call = ToolCall(
        index=0,
        id='toolu_read',
        name='read_file',
        arguments={'path': 'README.md'},
    )

    with terminal.stream_response() as response:
        response.start_tool(tool_call)
        response.complete_tool(
            tool_call,
            ToolResult.ok(
                'Cache hit: Read 20 lines.',
                metadata={
                    'transaction_decision': 'cache_hit',
                    'cache_hit': True,
                },
            ),
        )

    rendered = output.getvalue()
    assert '已运行 read_file · 查看项目' in rendered
    assert '复用缓存: Cache hit: Read 20 lines.' in rendered


def test_terminal_places_tool_group_between_model_text_blocks() -> None:
    terminal, output = terminal_with_output()
    usage = TokenUsage(input_tokens=40, output_tokens=8)
    first = ToolCall(
        index=0,
        id='toolu_read',
        name='read_file',
        arguments={'path': 'README.md'},
    )
    second = ToolCall(
        index=1,
        id='toolu_grep',
        name='grep',
        arguments={'pattern': 'M1.4'},
    )

    with terminal.stream_response() as response:
        response.append_text('我先检查项目。')
        response.start_tool(first)
        response.complete_tool(first, ToolResult.ok('读取完成。'))
        response.start_tool(second)
        response.complete_tool(second, ToolResult.ok('搜索完成。'))
        response.append_text('检查完成，下面是结论。')
        response.update_usage(usage)
        response.complete(
            TurnResult(
                text='检查完成，下面是结论。',
                usage=usage,
                tool_calls=(first, second),
            )
        )

    rendered = output.getvalue()
    before = rendered.index('我先检查项目。')
    tool_group = rendered.index('已运行 2 个工具')
    after = rendered.index('检查完成，下面是结论。')
    assert before < tool_group < after
    assert rendered.index('read_file') < rendered.index('grep')


def test_terminal_does_not_repeat_continued_text_around_tools() -> None:
    terminal, output = terminal_with_output()
    usage = TokenUsage(input_tokens=40, output_tokens=8)
    tool_call = ToolCall(
        index=0,
        id='toolu_read',
        name='read_file',
        arguments={'path': 'README.md'},
    )

    with terminal.stream_response() as response:
        response.append_text('First half, ')
        response.start_tool(tool_call)
        response.complete_tool(tool_call, ToolResult.ok('Read file.'))
        response.append_text('second half.')
        response.complete(
            TurnResult(
                text='First half, second half.',
                usage=usage,
                tool_calls=(tool_call,),
            )
        )

    rendered = output.getvalue()
    assert rendered.count('First half') == 1
    assert rendered.count('second half.') == 1


def test_terminal_renders_live_usage_state() -> None:
    terminal, output = terminal_with_output()

    terminal.console.print(
        token_usage_summary(
            TokenUsage(
                input_tokens=100,
                output_tokens=0,
            ),
            streaming=True,
        )
    )

    rendered = output.getvalue()
    assert 'streaming' in rendered
    assert 'input 100' in rendered
    assert 'output 0' in rendered


def test_streaming_preview_keeps_only_a_bounded_tail() -> None:
    preview = streaming_preview(
        'first\nsecond\nthird\nfourth',
        max_lines=2,
        max_characters=100,
    )

    assert preview == '…\nthird\nfourth'
    assert 'first' not in preview
    assert 'second' not in preview


def test_streaming_preview_bounds_long_unbroken_text() -> None:
    preview = streaming_preview(
        'a' * 500,
        max_lines=10,
        max_characters=80,
    )

    assert preview == f'…\n{"a" * 80}'


def test_terminal_renders_error_text_literally() -> None:
    terminal, output = terminal_with_output()

    terminal.show_error(RuntimeError('[provider] unavailable'))

    assert 'Model request failed: [provider] unavailable' in output.getvalue()


def test_terminal_renders_final_change_and_verification_evidence() -> None:
    terminal, output = terminal_with_output()
    usage = TokenUsage(input_tokens=20, output_tokens=4)
    evidence = VerificationEvidence(
        command='pytest -q',
        cwd='.',
        exit_code=0,
        duration_seconds=1.25,
        timed_out=False,
        workspace_revision=1,
    )

    with terminal.stream_response() as response:
        response.complete(
            TurnResult(
                text='Done.',
                usage=usage,
                changed_paths=('forge/app.py',),
                verification=evidence,
            )
        )

    rendered = output.getvalue()
    assert 'task completed' in rendered
    assert 'forge/app.py' in rendered
    assert 'pytest -q' in rendered
    assert 'exit 0' in rendered
