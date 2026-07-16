'''Tests for the Rich terminal presentation.'''

from io import StringIO

from rich.console import Console

from forge.runtime.state import TokenUsage, TurnResult
from forge.terminal import TerminalUI, token_usage_summary


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


def test_terminal_renders_error_text_literally() -> None:
    terminal, output = terminal_with_output()

    terminal.show_error(RuntimeError('[provider] unavailable'))

    assert 'Model request failed: [provider] unavailable' in output.getvalue()
