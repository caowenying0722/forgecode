'''Tests for the Rich terminal presentation.'''

from io import StringIO

from rich.console import Console

from forge.runtime.state import TokenUsage, ToolCall, TurnResult
from forge.terminal import (
    TerminalUI,
    streaming_preview,
    token_usage_summary,
)
from forge.tools.base import ToolResult


def terminal_with_output() -> tuple[TerminalUI, StringIO]:
    output = StringIO()
    console = Console(
        file=output,
        force_terminal=False,
        width=100,
    )
    return TerminalUI(console=console), output


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
    assert 'Ctrl+C to exit' in rendered
    assert 'Hello from ForgeCode' in rendered
    assert 'input 1,200' in rendered
    assert 'output 34' in rendered
    assert 'total 1,234' in rendered
    assert 'Session ended.' in rendered


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
            ),
        )

    rendered = output.getvalue()
    assert 'read_file {"path": "README.md"}' in rendered
    assert '✓' in rendered
    assert 'Read 20 lines.' in rendered
    assert 'run_command {"command": "pytest"}' in rendered
    assert '×' in rendered
    assert 'Command exited with code 1.' in rendered


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
