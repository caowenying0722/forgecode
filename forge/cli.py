'''Command-line entry point for ForgeCode.'''

import asyncio
from typing import Annotated, Optional

import typer

from forge import __version__
from forge.config import ConfigurationError, ForgeConfig
from forge.runtime.agent_loop import Conversation, run_single_turn
from forge.terminal import TerminalUI


app = typer.Typer(
    name='forge',
    help='ForgeCode terminal Agent Harness.',
    add_completion=False,
    invoke_without_command=True,
    no_args_is_help=False,
)


def version_callback(value: bool) -> None:
    '''Print the installed ForgeCode version and exit.'''
    if value:
        typer.echo(f'ForgeCode {__version__}')
        raise typer.Exit()


@app.callback()
def main(
    ctx: typer.Context,
    version: Annotated[
        bool,
        typer.Option(
            '--version',
            '-V',
            callback=version_callback,
            is_eager=True,
            help='Show the ForgeCode version and exit.',
        ),
    ] = False,
    prompt: Annotated[
        Optional[str],
        typer.Option(
            '--prompt',
            '-p',
            help='Send one prompt to the configured model.',
        ),
    ] = None,
) -> None:
    '''Start the ForgeCode command-line interface.'''
    if ctx.invoked_subcommand is None:
        if prompt is not None:
            if not prompt.strip():
                typer.echo('Prompt must not be empty.', err=True)
                raise typer.Exit(code=2)

            try:
                response = asyncio.run(run_single_turn(prompt))
            except ConfigurationError as error:
                print_configuration_error(error)
                raise typer.Exit(code=1) from error
            except KeyboardInterrupt as error:
                typer.echo('Model request interrupted.', err=True)
                raise typer.Exit(code=130) from error
            except Exception as error:
                typer.echo(f'Model request failed: {error}', err=True)
                raise typer.Exit(code=1) from error

            typer.echo(response)
            return

        try:
            run_interactive_chat()
        except ConfigurationError as error:
            print_configuration_error(error)
            raise typer.Exit(code=1) from error


def print_configuration_error(error: ConfigurationError) -> None:
    '''Print actionable model configuration guidance.'''
    typer.echo('Model configuration is incomplete.', err=True)
    typer.echo(str(error), err=True)
    typer.echo(
        'Set ANTHROPIC_API_KEY and MODEL_ID before starting ForgeCode.',
        err=True,
    )
    typer.echo(
        'ANTHROPIC_BASE_URL is optional and defaults to the official API.',
        err=True,
    )


def run_interactive_chat(
    session: Conversation | None = None,
    terminal: TerminalUI | None = None,
) -> None:
    '''Run a local chat session until the user interrupts it.'''
    resolved_session = session if session is not None else Conversation()
    resolved_terminal = terminal if terminal is not None else TerminalUI()
    client = getattr(resolved_session, 'client', None)
    model = getattr(client, 'model', 'configured model')
    resolved_terminal.show_welcome(model)

    while True:
        try:
            prompt = resolved_terminal.read_prompt()
        except (KeyboardInterrupt, EOFError, typer.Abort):
            resolved_terminal.show_goodbye()
            return

        if not prompt.strip():
            continue

        try:
            with resolved_terminal.thinking():
                response = asyncio.run(resolved_session.send(prompt))
        except (KeyboardInterrupt, typer.Abort):
            resolved_terminal.show_goodbye()
            return
        except Exception as error:
            resolved_terminal.show_error(error)
            continue

        resolved_terminal.show_response(response)


@app.command('config')
def show_config() -> None:
    '''Check the Anthropic-compatible model configuration.'''
    try:
        config = ForgeConfig.from_env()
    except ConfigurationError as error:
        print_configuration_error(error)
        raise typer.Exit(code=1) from error

    typer.echo('Anthropic configuration is ready.')
    typer.echo(f'Model ID: {config.model_id}')
    typer.echo(f'Base URL: {config.base_url}')
    typer.echo('API key: configured')


if __name__ == '__main__':
    app()
