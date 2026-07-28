'''Command-line entry point for ForgeCode.'''

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import suppress
import os
from pathlib import Path
import shutil
from typing import Annotated

import typer

from forge import __version__
from forge.config import (
    ConfigurationError,
    ForgeConfig,
    forge_home,
    initialize_user_config,
    write_user_config,
)
from forge.runtime.agent_loop import Conversation
from forge.runtime.state import (
    AgentPhaseChanged,
    CompletionBlocked,
    ContextCompacted,
    ModelTextDelta,
    ModelUsageUpdate,
    ToolExecutionCompleted,
    ToolExecutionStarted,
    TurnCompleted,
)
from forge.sessions.trajectory import TrajectoryRecorder
from forge.terminal import StreamingResponseView, TerminalUI
from forge.tools import create_default_registry
from forge.workspace_root import WorkspaceLocation, resolve_workspace


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
    cwd: Annotated[
        Path | None,
        typer.Option(
            '--cwd', '--cd', '-C',
            help='Start in this directory before resolving the workspace.',
        ),
    ] = None,
    root: Annotated[
        Path | None,
        typer.Option(
            '--root',
            help='Use this directory as the workspace boundary.',
        ),
    ] = None,
    no_git_root: Annotated[
        bool,
        typer.Option(
            '--no-git-root',
            help='Use cwd directly instead of discovering a parent Git root.',
        ),
    ] = False,
) -> None:
    '''Start the ForgeCode command-line interface.'''
    del version
    try:
        location = resolve_workspace(
            cwd=cwd,
            root=root,
            discover_git=not no_git_root,
        )
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    ctx.ensure_object(dict)
    ctx.obj['workspace'] = location
    previous_cwd = Path.cwd()
    os.chdir(location.cwd)
    ctx.call_on_close(lambda: os.chdir(previous_cwd))
    if ctx.invoked_subcommand is None:
        try:
            run_interactive_chat(
                workspace_root=location.root,
                startup_cwd=location.cwd,
            )
        except ConfigurationError as error:
            print_configuration_error(error)
            raise typer.Exit(code=1) from error


def print_configuration_error(error: ConfigurationError) -> None:
    '''Print actionable model configuration guidance.'''
    typer.echo('Model configuration is incomplete.', err=True)
    typer.echo(str(error), err=True)
    typer.echo(
        'Run `forge config --init`, then set your model and API key.',
        err=True,
    )
    typer.echo(
        'ANTHROPIC_BASE_URL is optional and defaults to the official API.',
        err=True,
    )


def run_interactive_chat(
    session: Conversation | None = None,
    terminal: TerminalUI | None = None,
    recorder: TrajectoryRecorder | None = None,
    workspace_root: Path | None = None,
    startup_cwd: Path | None = None,
) -> None:
    '''Run a local chat session until the user interrupts it.'''
    root = (workspace_root or Path.cwd()).resolve()
    cwd = (startup_cwd or Path.cwd()).resolve()
    resolved_session = (
        session
        if session is not None
        else Conversation(
            registry=create_default_registry(root),
            context_root=root,
        )
    )
    resolved_terminal = (
        terminal
        if terminal is not None
        else TerminalUI(workspace_root=root)
    )
    resolved_recorder = (
        recorder
        if recorder is not None
        else create_trajectory_recorder(root)
    )
    enable_rollout = getattr(
        resolved_session,
        'enable_rollout_persistence',
        None,
    )
    if enable_rollout is not None:
        enable_rollout()
    client = getattr(resolved_session, 'client', None)
    model = getattr(client, 'model', 'configured model')
    resolved_terminal.show_welcome(model, workspace_root=root, cwd=cwd)

    while True:
        try:
            prompt = resolved_terminal.read_prompt()
        except (KeyboardInterrupt, EOFError, typer.Abort):
            resolved_terminal.show_goodbye()
            return

        if not prompt.strip():
            continue

        if prompt.strip() == '/exit':
            resolved_terminal.show_goodbye()
            return

        if prompt.strip() == '/context':
            stats = getattr(resolved_session, 'context_stats', None)
            if stats is None:
                resolved_terminal.show_error(
                    RuntimeError('Context statistics are unavailable.')
                )
            else:
                resolved_terminal.show_context(stats)
            continue

        if prompt.strip() == '/compact':
            compact = getattr(resolved_session, 'compact', None)
            if compact is None:
                resolved_terminal.show_error(
                    RuntimeError('Context compaction is unavailable.')
                )
            else:
                resolved_terminal.show_compaction(asyncio.run(compact()))
            continue

        if prompt.strip() == '/resume':
            try:
                resolved_terminal.show_notice(
                    'Session',
                    resolved_session.resume_session(),
                )
            except (OSError, ValueError) as error:
                resolved_terminal.show_error(error)
            continue

        if prompt.strip().startswith('/resume '):
            session_id = prompt.strip()[len('/resume '):].strip()
            if not session_id:
                resolved_terminal.show_error(
                    ValueError('Usage: /resume session-id')
                )
            else:
                try:
                    resolved_terminal.show_notice(
                        'Session',
                        resolved_session.resume_session(session_id),
                    )
                except (OSError, ValueError) as error:
                    resolved_terminal.show_error(error)
            continue

        if prompt.strip() == '/fork':
            try:
                resolved_terminal.show_notice(
                    'Session',
                    resolved_session.fork_session(),
                )
            except (OSError, ValueError) as error:
                resolved_terminal.show_error(error)
            continue

        if prompt.strip().startswith('/fork '):
            session_id = prompt.strip()[len('/fork '):].strip()
            if not session_id:
                resolved_terminal.show_error(
                    ValueError('Usage: /fork session-id')
                )
            else:
                try:
                    resolved_terminal.show_notice(
                        'Session',
                        resolved_session.fork_session(session_id),
                    )
                except (OSError, ValueError) as error:
                    resolved_terminal.show_error(error)
            continue

        if prompt.strip() == '/sessions':
            resolved_terminal.show_notice(
                'Sessions',
                resolved_session.session_history(),
            )
            continue

        if prompt.strip() == '/worktrees':
            resolved_terminal.show_notice(
                'Subagent worktrees',
                resolved_session.subagent_worktrees(),
            )
            continue

        if prompt.strip() == '/mcp':
            resolved_terminal.show_notice('MCP', resolved_session.mcp_status())
            continue

        if prompt.strip() == '/hooks':
            resolved_terminal.show_notice(
                'Hooks',
                resolved_session.hooks_status(),
            )
            continue

        if prompt.strip() == '/todo':
            resolved_terminal.show_notice(
                'TODO',
                resolved_session.todo_status(),
            )
            continue

        if prompt.strip() == '/permissions':
            current_mode = getattr(
                getattr(resolved_session, 'permission', None),
                'mode',
                'trusted',
            )
            selected = resolved_terminal.select_permission_mode(current_mode)
            if selected is None:
                resolved_terminal.show_notice(
                    'Permissions',
                    'Permission selection cancelled.',
                )
            else:
                try:
                    resolved_terminal.show_notice(
                        'Permissions',
                        resolved_session.permission_set(selected),
                    )
                except ValueError as error:
                    resolved_terminal.show_error(error)
            continue

        if prompt.strip() == '/mode':
            resolved_terminal.show_notice(
                'Mode',
                resolved_session.mode_show(),
            )
            continue

        if prompt.strip().startswith('/mode '):
            mode = prompt.strip()[len('/mode '):].strip()
            try:
                resolved_terminal.show_notice(
                    'Mode',
                    resolved_session.mode_set(mode),
                )
            except ValueError as error:
                resolved_terminal.show_error(error)
            continue

        if prompt.strip() == '/plan':
            resolved_terminal.show_notice(
                'Mode',
                resolved_session.mode_set('plan'),
            )
            continue

        if prompt.strip() == '/code':
            resolved_terminal.show_notice(
                'Mode',
                resolved_session.mode_set('code'),
            )
            continue

        if prompt.strip() == '/task':
            resolved_terminal.show_notice('Task', resolved_session.task_show())
            continue

        if prompt.strip() == '/task history':
            resolved_terminal.show_notice(
                'Task',
                resolved_session.task_history(),
            )
            continue

        if prompt.strip().startswith('/task resume '):
            task_id = prompt.strip()[len('/task resume '):].strip()
            if not task_id:
                resolved_terminal.show_error(
                    ValueError('Usage: /task resume task-id')
                )
            else:
                try:
                    notice = resolved_session.task_resume(task_id)
                    resolved_terminal.show_notice('Task', notice)
                except (OSError, ValueError) as error:
                    resolved_terminal.show_error(error)
            continue

        if prompt.startswith('/remember '):
            payload = prompt[len('/remember '):].strip()
            name, separator, content = payload.partition('|')
            if not separator:
                resolved_terminal.show_error(
                    ValueError('Usage: /remember name | content')
                )
            else:
                try:
                    notice = resolved_session.remember(name.strip(), content.strip())
                    resolved_terminal.show_notice('Memory', notice)
                except ValueError as error:
                    resolved_terminal.show_error(error)
            continue

        if prompt == '/memory list':
            resolved_terminal.show_notice(
                'Memory', resolved_session.memory_list()
            )
            continue

        if prompt.startswith('/memory show '):
            resolved_terminal.show_notice(
                'Memory',
                resolved_session.memory_show(
                    prompt[len('/memory show '):].strip()
                ),
            )
            continue

        if prompt.startswith('/memory forget '):
            resolved_terminal.show_notice(
                'Memory',
                resolved_session.memory_forget(
                    prompt[len('/memory forget '):].strip()
                ),
            )
            continue

        if prompt == '/memory rebuild':
            resolved_terminal.show_notice(
                'Memory', resolved_session.memory_rebuild()
            )
            continue

        if prompt == '/memory consolidate':
            resolved_terminal.show_notice(
                'Memory', resolved_session.memory_consolidate()
            )
            continue

        try:
            with resolved_terminal.stream_response() as response_view:
                asyncio.run(
                    render_streamed_turn_interruptibly(
                        resolved_session,
                        prompt,
                        response_view,
                        resolved_recorder,
                        wait_for_interrupt=getattr(
                            resolved_terminal,
                            'wait_for_interrupt',
                            None,
                        ),
                    )
                )
        except UserTurnInterrupted:
            resolved_terminal.show_interrupted()
            continue
        except (KeyboardInterrupt, typer.Abort):
            resolved_terminal.show_goodbye()
            return
        except Exception as error:
            resolved_terminal.show_error(error)
            continue


class UserTurnInterrupted(Exception):
    '''The user pressed Esc to stop only the active response.'''


async def render_streamed_turn_interruptibly(
    session: Conversation,
    prompt: str,
    response_view: StreamingResponseView,
    recorder: TrajectoryRecorder | None = None,
    *,
    wait_for_interrupt: Callable[[], Awaitable[None]] | None,
) -> None:
    '''Race one response against the terminal Esc watcher.'''
    if wait_for_interrupt is None:
        await render_streamed_turn(session, prompt, response_view, recorder)
        return
    response_task = asyncio.create_task(
        render_streamed_turn(session, prompt, response_view, recorder)
    )
    interrupt_task = asyncio.create_task(wait_for_interrupt())
    try:
        done, _ = await asyncio.wait(
            {response_task, interrupt_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if response_task in done:
            await response_task
            return
        response_task.cancel()
        with suppress(asyncio.CancelledError):
            await response_task
        response_view.interrupt()
        raise UserTurnInterrupted('Esc pressed')
    finally:
        interrupt_task.cancel()
        with suppress(asyncio.CancelledError):
            await interrupt_task


async def render_streamed_turn(
    session: Conversation,
    prompt: str,
    response_view: StreamingResponseView,
    recorder: TrajectoryRecorder | None = None,
) -> None:
    '''Forward conversation stream events to the live terminal view.'''
    set_permission_approver = getattr(
        session,
        'set_permission_approver',
        None,
    )
    if set_permission_approver is not None:
        set_permission_approver(response_view.request_permission)
    if recorder is not None:
        recorder.record_user_message(prompt)
    try:
        async for event in session.stream(prompt):
            if recorder is not None:
                recorder.record_event(event)
            if isinstance(event, ModelTextDelta):
                response_view.append_text(event.text)
            elif isinstance(event, AgentPhaseChanged):
                response_view.update_phase(event.phase, event.reason)
            elif isinstance(event, ModelUsageUpdate):
                response_view.update_usage(
                    event.usage,
                    request_usage=event.request_usage,
                    model_calls=event.model_calls,
                )
            elif isinstance(event, ToolExecutionStarted):
                response_view.start_tool(event.tool_call)
            elif isinstance(event, ToolExecutionCompleted):
                response_view.complete_tool(event.tool_call, event.result)
            elif isinstance(event, CompletionBlocked):
                response_view.block_completion(event.reasons)
            elif isinstance(event, ContextCompacted):
                response_view.compact_context(event)
            elif isinstance(event, TurnCompleted):
                run_stop_hooks = getattr(session, 'run_stop_hooks', None)
                if run_stop_hooks is not None:
                    await run_stop_hooks(event.result)
                response_view.complete(event.result)
        save_session = getattr(session, 'save_session', None)
        if save_session is not None:
            save_session()
    except Exception as error:
        if recorder is not None:
            recorder.record_error(error)
        raise
    finally:
        if set_permission_approver is not None:
            set_permission_approver(None)


def create_trajectory_recorder(root: Path) -> TrajectoryRecorder:
    '''Create the default append-only recorder for one CLI session.'''
    return TrajectoryRecorder.create(root)


@app.command('config')
def show_config(
    ctx: typer.Context,
    init: Annotated[
        bool,
        typer.Option('--init', help='Create user config and credential templates.'),
    ] = False,
    migrate_project: Annotated[
        bool,
        typer.Option(
            '--migrate-project',
            help='Copy validated current-project model settings to user config.',
        ),
    ] = False,
) -> None:
    '''Check the Anthropic-compatible model configuration.'''
    home = forge_home()
    location: WorkspaceLocation = ctx.obj['workspace']
    if migrate_project:
        try:
            existing = ForgeConfig.from_env(cwd=location.root, home=home)
        except ConfigurationError as error:
            print_configuration_error(error)
            raise typer.Exit(code=1) from error
        config_path, env_path = write_user_config(existing, home=home)
        typer.echo(f'Migrated user config: {config_path}')
        typer.echo(f'Migrated credentials: {env_path}')
        typer.echo('API key value was not printed.')
        return
    if init:
        config_path, env_path = initialize_user_config(home=home)
        typer.echo(f'User config: {config_path}')
        typer.echo(f'Credentials: {env_path}')
        typer.echo('Edit both files, then run `forge config` to validate them.')
        return
    try:
        config = ForgeConfig.from_env(cwd=location.root, home=home)
    except ConfigurationError as error:
        print_configuration_error(error)
        raise typer.Exit(code=1) from error

    typer.echo('Anthropic configuration is ready.')
    typer.echo(f'Model ID: {config.model_id}')
    typer.echo(f'Base URL: {config.base_url}')
    typer.echo(f'Max output tokens: {config.max_tokens:,}')
    typer.echo(
        f'Model request timeout: {config.request_timeout_seconds:g} seconds'
    )
    typer.echo(
        'Context window: '
        + (
            f'{config.context_window:,}'
            if config.context_window is not None
            else 'not configured'
        )
    )
    typer.echo('API key: configured')
    typer.echo(f'User config: {home / "config.toml"}')
    typer.echo(f'Project config: {location.root / ".forge" / "config.toml"}')


@app.command('doctor')
def doctor(ctx: typer.Context) -> None:
    '''Diagnose global command, workspace, Git, and model configuration.'''
    location: WorkspaceLocation = ctx.obj['workspace']
    typer.echo('ForgeCode doctor')
    typer.echo(f'Executable: {shutil.which("forge") or "not on PATH"}')
    typer.echo(f'Workspace: {location.root} ({location.source})')
    typer.echo(f'Cwd: {location.cwd}')
    typer.echo(f'User config: {forge_home() / "config.toml"}')
    typer.echo(
        f'Project config: {location.root / ".forge" / "config.toml"}'
    )
    try:
        config = ForgeConfig.from_env(cwd=location.root)
    except ConfigurationError as error:
        typer.echo(f'Model config: invalid ({error})')
        raise typer.Exit(code=1) from error
    typer.echo(f'Model config: ready ({config.model_id})')


if __name__ == '__main__':
    app()
