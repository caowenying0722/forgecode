'''Rich terminal presentation for ForgeCode.'''

from __future__ import annotations

from pathlib import Path

from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from forge import __version__
from forge.runtime.state import TokenUsage, TurnResult


class TerminalUI:
    '''Render the interactive ForgeCode conversation.'''

    def __init__(self, console: Console | None = None) -> None:
        self.console = console if console is not None else Console()

    def show_welcome(self, model: str) -> None:
        '''Show a compact session header inspired by modern coding agents.'''
        title = Text.assemble(
            ('\u25c6 ', 'bold bright_cyan'),
            ('ForgeCode', 'bold white'),
            (f' v{__version__}', 'dim'),
        )
        details = Table.grid(padding=(0, 2))
        details.add_column(style='dim', no_wrap=True)
        details.add_column()
        details.add_row('model', Text(model, style='bright_white'))
        details.add_row('cwd', Text(str(Path.cwd()), style='bright_white'))

        self.console.print(
            Panel.fit(
                details,
                title=title,
                subtitle=Text('Ctrl+C to exit', style='dim'),
                border_style='bright_cyan',
                padding=(1, 2),
            )
        )
        self.console.print(
            '[dim]Ask a question or describe a coding task.[/]'
        )
        self.console.print()

    def read_prompt(self) -> str:
        '''Read one user message with a compact agent-style prompt.'''
        return self.console.input('[bold bright_cyan]\u276f[/] ')

    def stream_response(self) -> StreamingResponseView:
        '''Create a live view for one streaming model response.'''
        return StreamingResponseView(self.console)

    def show_error(self, error: Exception) -> None:
        '''Render a recoverable request error without interpreting its markup.'''
        self.console.print(
            f'[bold red]Error[/] [dim]Model request failed:[/] '
            f'{escape(str(error))}'
        )

    def show_goodbye(self) -> None:
        '''Render the session exit message.'''
        self.console.print()
        self.console.print('[dim]Session ended.[/]')


class StreamingResponseView:
    '''Update streamed Markdown and exact usage in place.'''

    def __init__(self, console: Console) -> None:
        self.console = console
        self.text = ''
        self.usage: TokenUsage | None = None
        self.completed = False
        self.live = Live(
            self._render(),
            console=console,
            refresh_per_second=16,
            vertical_overflow='visible',
        )

    def __enter__(self) -> StreamingResponseView:
        self.console.print()
        self.console.print(
            Text.assemble(
                ('\u25cf ', 'bold bright_cyan'),
                ('ForgeCode', 'bold bright_white'),
            )
        )
        self.live.start(refresh=True)
        return self

    def __exit__(self, *_: object) -> None:
        self.live.stop()
        self.console.print()

    def append_text(self, text: str) -> None:
        '''Append one provider text delta and refresh immediately.'''
        self.text += text
        self.live.update(self._render(), refresh=True)

    def update_usage(self, usage: TokenUsage) -> None:
        '''Refresh the exact usage snapshot reported by the provider.'''
        self.usage = usage
        self.live.update(self._render(), refresh=True)

    def complete(self, result: TurnResult) -> None:
        '''Finalize the view with validated text and exact final usage.'''
        self.text = result.text
        self.usage = result.usage
        self.completed = True
        self.live.update(self._render(), refresh=True)

    def _render(self) -> Group:
        content = (
            Markdown(self.text)
            if self.text
            else Spinner(
                'dots',
                Text('Waiting for model...', style='bright_cyan'),
            )
        )
        return Group(
            content,
            token_usage_summary(
                self.usage,
                streaming=not self.completed,
            ),
        )


def token_usage_summary(
    usage: TokenUsage | None,
    *,
    streaming: bool,
) -> Text:
    '''Build the live or final token usage line.'''
    prefix = '\u21b3 streaming' if streaming else '\u21b3 tokens'
    if usage is None:
        return Text.assemble(
            (prefix, 'dim'),
            ('  input ...  output ...  total ...', 'dim'),
        )

    summary = Text.assemble(
        (prefix, 'dim'),
        ('  input ', 'dim'),
        (f'{usage.total_input_tokens:,}', 'bright_cyan'),
        ('  output ', 'dim'),
        (f'{usage.output_tokens:,}', 'bright_cyan'),
        ('  total ', 'dim'),
        (f'{usage.total_tokens:,}', 'bold bright_cyan'),
    )
    if usage.cache_read_input_tokens:
        summary.append('  cache read ', style='dim')
        summary.append(
            f'{usage.cache_read_input_tokens:,}',
            style='bright_cyan',
        )
    if usage.cache_creation_input_tokens:
        summary.append('  cache write ', style='dim')
        summary.append(
            f'{usage.cache_creation_input_tokens:,}',
            style='bright_cyan',
        )
    return summary
