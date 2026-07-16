'''Tests for the Rich terminal presentation.'''

from io import StringIO

from rich.console import Console

from forge.terminal import TerminalUI


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

    terminal.show_welcome('test-model')
    terminal.show_response('**Hello** from ForgeCode')
    terminal.show_goodbye()

    rendered = output.getvalue()
    assert 'ForgeCode v0.1.0' in rendered
    assert 'test-model' in rendered
    assert 'Ctrl+C to exit' in rendered
    assert 'Hello from ForgeCode' in rendered
    assert 'Session ended.' in rendered


def test_terminal_renders_error_text_literally() -> None:
    terminal, output = terminal_with_output()

    terminal.show_error(RuntimeError('[provider] unavailable'))

    assert 'Model request failed: [provider] unavailable' in output.getvalue()
