'''Rich terminal presentation for ForgeCode.'''

from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.status import Status
from rich.table import Table
from rich.text import Text

from forge import __version__


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

    def thinking(self) -> Status:
        '''Show a transient status while the model request is running.'''
        return self.console.status(
            '[bright_cyan]Thinking...[/]',
            spinner='dots',
        )

    def show_response(self, response: str) -> None:
        '''Render model text as terminal Markdown.'''
        self.console.print()
        self.console.print(
            Text.assemble(
                ('\u25cf ', 'bold bright_cyan'),
                ('ForgeCode', 'bold bright_white'),
            )
        )
        self.console.print(Markdown(response))
        self.console.print()

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
