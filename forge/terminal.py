'''Rich terminal presentation for ForgeCode.'''

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
import select
import sys
from time import monotonic
from typing import Any, Protocol

from prompt_toolkit import PromptSession
from prompt_toolkit.application.current import get_app
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.shortcuts import CompleteStyle, choice
from prompt_toolkit.styles import Style
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from forge import __version__
from forge.config import SUPPORTED_MODEL_IDS
from forge.runtime.agent_state import AgentPhase
from forge.runtime.state import ContextCompacted, TokenUsage, ToolCall, TurnResult
from forge.context.manager import ContextStats
from forge.context.manager import CompactionReport
from forge.tools.base import ToolResult
from forge.tools.search import iter_files


PermissionSelector = Callable[[str], str | None]
ModelSelector = Callable[[str], str | None]
SessionChoice = tuple[str, str, str]
SessionSelector = Callable[[tuple[SessionChoice, ...]], str | None]
ApprovalSelector = Callable[[ToolCall, object], str]


PERMISSION_CHOICES = (
    (
        'readonly',
        'Read Only',
        'Inspect files and repository state; block writes and commands.',
    ),
    (
        'strict',
        'Ask for approval',
        'Ask before every write or process action.',
    ),
    (
        'auto',
        'Approve for me',
        'Auto-approve workspace edits and low-risk local commands.',
    ),
    (
        'trusted',
        'Full Access',
        'Run available tools without approval prompts.',
    ),
)

APPROVAL_CHOICES = (
    (
        'allow_once',
        'Allow once',
        'Run only this operation.',
    ),
    (
        'allow_session',
        'Allow similar this session',
        'Reuse approval for the same command or tool target.',
    ),
    (
        'deny',
        'Deny',
        'Do not run this operation.',
    ),
)

MODEL_CHOICES = tuple(
    (model_id, model_id, 'Set as the current and global default model.')
    for model_id in SUPPORTED_MODEL_IDS
)


class InlineChoiceCompleter(Completer):
    '''Render a small choice list using the normal terminal completion menu.'''

    def __init__(
        self,
        choices: tuple[tuple[str, str, str], ...],
        *,
        current: str | None = None,
    ) -> None:
        self.choices = choices
        self.current = current

    def get_completions(
        self,
        document: Document,
        complete_event: object,
    ):
        del complete_event
        query = document.text_before_cursor.casefold()
        for value, label, description in self.choices:
            if query and query not in value.casefold() and query not in label.casefold():
                continue
            marker = '\u25cf ' if value == self.current else '  '
            yield Completion(
                value,
                start_position=-len(document.text_before_cursor),
                display=f'{marker}{label}',
                display_meta=description,
            )


INLINE_CHOICE_KEY_BINDINGS = KeyBindings()


@INLINE_CHOICE_KEY_BINDINGS.add('escape', eager=True)
def _cancel_inline_choice(event: object) -> None:
    event.app.exit(result='')


@dataclass(frozen=True, slots=True)
class SlashCommandSpec:
    '''One discoverable local command shown by the interactive prompt.'''

    completion: str
    usage: str
    description: str


SLASH_COMMANDS = (
    SlashCommandSpec('/context', '/context', '查看当前上下文统计'),
    SlashCommandSpec('/compact', '/compact', '立即压缩当前会话'),
    SlashCommandSpec('/resume', '/resume', '选择并恢复保存的会话'),
    SlashCommandSpec('/fork', '/fork', '分叉最近保存的会话'),
    SlashCommandSpec('/worktrees', '/worktrees', '列出保留的子 Agent worktree'),
    SlashCommandSpec('/mode', '/mode', '查看当前交互模式'),
    SlashCommandSpec(
        '/mode ',
        '/mode auto|plan|code',
        '切换交互模式',
    ),
    SlashCommandSpec('/plan', '/plan', '切换到只读计划模式'),
    SlashCommandSpec('/code', '/code', '切换到代码执行模式'),
    SlashCommandSpec('/model', '/model', '选择当前和全局默认模型'),
    SlashCommandSpec(
        '/permissions',
        '/permissions',
        '打开权限模式内联菜单',
    ),
    SlashCommandSpec('/mcp', '/mcp', '查看 MCP 服务器与工具状态'),
    SlashCommandSpec('/hooks', '/hooks', '查看当前 Hook 注册列表'),
    SlashCommandSpec('/todo', '/todo', '查看当前 TODO 计划'),
    SlashCommandSpec('/exit', '/exit', '退出 ForgeCode'),
    SlashCommandSpec('/task', '/task', '查看当前任务与计划'),
    SlashCommandSpec('/memory list', '/memory list', '列出仓库记忆'),
)


class SlashCommandCompleter(Completer):
    '''Offer local commands only while the input starts with a slash.'''

    def get_completions(
        self,
        document: Document,
        complete_event: object,
    ):
        del complete_event
        text = document.text_before_cursor
        if not text.startswith('/'):
            return
        normalized = text.casefold()
        for command in SLASH_COMMANDS:
            if not command.completion.casefold().startswith(normalized):
                continue
            yield Completion(
                command.completion,
                start_position=-len(text),
                display=command.usage,
                display_meta=command.description,
            )


SLASH_COMMAND_COMPLETER = SlashCommandCompleter()


def workspace_file_match_score(
    path: str,
    query: str,
) -> tuple[int, int, int, str] | None:
    '''Rank one workspace path using prefix, substring, then subsequence match.'''
    candidate = path.casefold()
    normalized = query.casefold().replace('\\', '/')
    if not normalized:
        return (0, 0, len(candidate), candidate)

    name = candidate.rsplit('/', 1)[-1]
    if candidate.startswith(normalized):
        return (0, 0, len(candidate), candidate)
    if name.startswith(normalized):
        return (1, 0, len(candidate), candidate)
    if normalized in candidate:
        return (2, candidate.index(normalized), len(candidate), candidate)

    position = -1
    gap = 0
    for character in normalized:
        next_position = candidate.find(character, position + 1)
        if next_position < 0:
            return None
        gap += next_position - position - 1
        position = next_position
    return (3, gap, len(candidate), candidate)


class WorkspaceFileCompleter(Completer):
    '''Complete an active @token with a protected workspace-relative file.'''

    MAX_RESULTS = 100
    _mention = re.compile(r'(?<!\S)@(?P<query>\S*)$')

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.paths: tuple[str, ...] = ()
        self.refresh()

    def refresh(self) -> None:
        try:
            self.paths = tuple(
                path.relative_to(self.root).as_posix()
                for path in iter_files(self.root)
            )
        except OSError:
            self.paths = ()

    def get_completions(
        self,
        document: Document,
        complete_event: object,
    ):
        del complete_event
        match = self._mention.search(document.text_before_cursor)
        if match is None:
            return
        query = match.group('query')
        ranked: list[tuple[tuple[int, int, int, str], str]] = []
        for path in self.paths:
            score = workspace_file_match_score(path, query)
            if score is not None:
                ranked.append((score, path))
        ranked.sort()
        start_position = -(len(query) + 1)
        for _, path in ranked[: self.MAX_RESULTS]:
            yield Completion(
                f'@{path} ',
                start_position=start_position,
                display=path,
                display_meta='workspace file',
            )


class ForgePromptCompleter(Completer):
    '''Combine Slash Commands and @ workspace-file mentions.'''

    def __init__(self, root: Path) -> None:
        self.workspace_files = WorkspaceFileCompleter(root)

    def refresh(self) -> None:
        self.workspace_files.refresh()

    def get_completions(
        self,
        document: Document,
        complete_event: object,
    ):
        if document.text_before_cursor.startswith('/'):
            yield from SLASH_COMMAND_COMPLETER.get_completions(
                document,
                complete_event,
            )
            return
        yield from self.workspace_files.get_completions(
            document,
            complete_event,
        )


@dataclass(slots=True)
class _TextTimelineBlock:
    text: str = ''


@dataclass(slots=True)
class _ToolActivity:
    tool_call: ToolCall
    result: ToolResult | None = None


@dataclass(slots=True)
class _ToolTimelineBlock:
    activities: list[_ToolActivity] = field(default_factory=list)


@dataclass(slots=True)
class _NoticeTimelineBlock:
    title: str
    lines: tuple[str, ...] = ()
    style: str = 'green'


type _TimelineBlock = _TextTimelineBlock | _ToolTimelineBlock | _NoticeTimelineBlock


class _InteractivePrompt(Protocol):
    def prompt(self, message: Any = '', **kwargs: Any) -> str:
        ...


class EncodingSafeTextIO:
    '''Text stream wrapper that replaces characters unsupported by the sink.'''

    def __init__(self, wrapped: Any) -> None:
        self.wrapped = wrapped

    @property
    def encoding(self) -> str | None:
        return getattr(self.wrapped, 'encoding', None)

    def isatty(self) -> bool:
        isatty = getattr(self.wrapped, 'isatty', None)
        return bool(isatty()) if isatty is not None else False

    def flush(self) -> None:
        self.wrapped.flush()

    def write(self, text: str) -> int:
        encoding = self.encoding or 'utf-8'
        safe = text.encode(encoding, errors='replace').decode(encoding)
        return self.wrapped.write(safe)


_MOJIBAKE_MARKERS = frozenset(
    '鏍嵁缁瀹屽杽椤圭洰鍔熻兘锛氫紭鍏堝疄鐜版湰'
    '鍦版渶楂樺垎璁板綍銆傝锋妸鎺叆鏄剧ず褰撳墠'
    '鏈苟鎴忕粨淇濆瓨'
    '鍘﹂棬浠婂ぉ澶╂皵鑱旂綉灏辩敤鏌ヤ竴涓嬶紝'
    '绠鐭鍥炵瓟'
)
_RECOVERED_ZH_MARKERS = (
    '根据',
    '继续',
    '完善',
    '项目',
    '功能',
    '实现',
    '保存',
    '修改',
    '优化',
    '删除',
    '替换',
    '天气',
    '联网',
    '查询',
    '回答',
)


def repair_input_text(value: str) -> str:
    '''Repair invalid surrogate and common Windows UTF-8/GBK stdin mojibake.'''
    repaired = value.encode('utf-8', errors='replace').decode('utf-8')
    if sum(character in _MOJIBAKE_MARKERS for character in repaired) < 3:
        return repaired
    try:
        candidate = repaired.encode(
            'gb18030',
            errors='ignore',
        ).decode('utf-8', errors='replace')
    except UnicodeError:
        return repaired
    if any(marker in candidate for marker in _RECOVERED_ZH_MARKERS):
        return candidate
    return repaired


class TerminalUI:
    '''Render the interactive ForgeCode conversation.'''

    def __init__(
        self,
        console: Console | None = None,
        prompt_session: _InteractivePrompt | None = None,
        workspace_root: Path | None = None,
        permission_selector: PermissionSelector | None = None,
        model_selector: ModelSelector | None = None,
        session_selector: SessionSelector | None = None,
        approval_selector: ApprovalSelector | None = None,
    ) -> None:
        self.console = (
            console
            if console is not None
            else Console(file=EncodingSafeTextIO(sys.stdout))
        )
        self.workspace_root = (workspace_root or Path.cwd()).resolve()
        self.prompt_completer = ForgePromptCompleter(self.workspace_root)
        self.prompt_session = prompt_session
        self.permission_selector = permission_selector
        self.model_selector = model_selector
        self.session_selector = session_selector
        self.approval_selector = approval_selector
        if self.prompt_session is None and self.console.is_terminal:
            self.prompt_session = PromptSession(
                completer=self.prompt_completer,
                complete_while_typing=True,
                reserve_space_for_menu=8,
            )

    def show_welcome(
        self,
        model: str,
        *,
        workspace_root: Path | None = None,
        cwd: Path | None = None,
    ) -> None:
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
        resolved_cwd = (cwd or Path.cwd()).resolve()
        resolved_root = (workspace_root or resolved_cwd).resolve()
        details.add_row('workspace', Text(str(resolved_root), style='bright_white'))
        if resolved_cwd != resolved_root:
            details.add_row('cwd', Text(str(resolved_cwd), style='bright_white'))

        self.console.print(
            Panel.fit(
                details,
                title=title,
                subtitle=Text('Esc interrupt · Ctrl+C exit', style='dim'),
                border_style='bright_cyan',
                padding=(1, 2),
            )
        )
        self.console.print(
            '[dim]Ask a question or describe a coding task. '
            'Type @ for workspace files and / for commands.[/]'
        )
        self.console.print()

    def read_prompt(self) -> str:
        '''Read one message, preserving bracketed multi-line terminal paste.'''
        if self.prompt_session is not None:
            self.prompt_completer.refresh()
            return repair_input_text(
                self.prompt_session.prompt(
                    [('ansibrightcyan bold', '\u276f ')],
                    completer=self.prompt_completer,
                    complete_while_typing=True,
                    complete_style=CompleteStyle.COLUMN,
                    reserve_space_for_menu=8,
                )
            )
        return repair_input_text(
            self.console.input('[bold bright_cyan]>[/] ')
        )

    def stream_response(self) -> StreamingResponseView:
        '''Create a live view for one streaming model response.'''
        return StreamingResponseView(
            self.console,
            approval_selector=(
                self.approval_selector or self.select_tool_approval
            ),
        )

    def select_permission_mode(self, current: str) -> str | None:
        '''Open a keyboard-first permission preset picker.'''
        if self.permission_selector is not None:
            return self.permission_selector(current)
        if self.console.is_terminal:
            return self._select_inline(
                'Permissions \u276f ',
                PERMISSION_CHOICES,
                current=current,
            )
        self.console.print('[bold]ForgeCode Permissions[/]')
        for index, (_, label, description) in enumerate(
            PERMISSION_CHOICES,
            start=1,
        ):
            self.console.print(f'{index}. {label} — {description}')
        answer = self.console.input('Select 1-4 (blank to cancel): ').strip()
        if answer.isdigit() and 1 <= int(answer) <= len(PERMISSION_CHOICES):
            return PERMISSION_CHOICES[int(answer) - 1][0]
        return None

    def select_model(self, current: str) -> str | None:
        '''Open a keyboard-first model picker.'''
        if self.model_selector is not None:
            return self.model_selector(current)
        if self.console.is_terminal:
            return self._select_inline(
                'Model \u276f ',
                MODEL_CHOICES,
                current=current,
            )
        self.console.print('[bold]ForgeCode Model[/]')
        for index, (model_id, _, description) in enumerate(
            MODEL_CHOICES,
            start=1,
        ):
            self.console.print(f'{index}. {model_id} — {description}')
        answer = self.console.input(
            f'Select 1-{len(MODEL_CHOICES)} (blank to cancel): '
        ).strip()
        if answer.isdigit() and 1 <= int(answer) <= len(MODEL_CHOICES):
            return MODEL_CHOICES[int(answer) - 1][0]
        return None

    def select_session(
        self,
        choices: tuple[SessionChoice, ...],
    ) -> str | None:
        '''Open a keyboard-first saved session picker.'''
        if self.session_selector is not None:
            return self.session_selector(choices)
        if not choices:
            return None
        if self.console.is_terminal:
            bindings = KeyBindings()

            @bindings.add('escape')
            def cancel(event: object) -> None:
                event.app.exit(result=None)

            style = Style.from_dict(
                {
                    'choice.label': '#f5f5f5 bold',
                    'choice.meta': '#777777',
                    'bottom-toolbar': 'bg:default #666666',
                }
            )
            options = tuple(
                (
                    session_id,
                    [
                        ('class:choice.label', label),
                        ('class:choice.meta', f'  {description}'),
                    ],
                )
                for session_id, label, description in choices
            )
            try:
                return choice(
                    'Resume \u276f ',
                    options=options,
                    default=choices[0][0],
                    symbol='\u25cf',
                    show_frame=False,
                    style=style,
                    key_bindings=bindings,
                    bottom_toolbar=(
                        '\u2191/\u2193 select  Enter confirm  '
                        'Esc cancel'
                    ),
                )
            except (KeyboardInterrupt, EOFError):
                return None
        self.console.print('[bold]ForgeCode Sessions[/]')
        for index, (session_id, label, description) in enumerate(
            choices,
            start=1,
        ):
            self.console.print(
                f'{index}. {label} — {description or session_id}'
            )
        answer = self.console.input(
            f'Select 1-{len(choices)} (blank to cancel): '
        ).strip()
        if answer.isdigit() and 1 <= int(answer) <= len(choices):
            return choices[int(answer) - 1][0]
        return None

    def select_tool_approval(
        self,
        tool_call: ToolCall,
        effect: object,
    ) -> str:
        '''Ask for one tool approval in an inline terminal completion menu.'''
        details = permission_request_details(tool_call, effect)
        if self.console.is_terminal:
            self.console.print('[bold yellow]Approval required[/]')
            self.console.print(Text(details))
            return self._select_inline(
                'Approval \u276f ',
                APPROVAL_CHOICES,
                current='deny',
            ) or 'deny'
        self.console.print('[bold yellow]Permission required[/]')
        self.console.print(Text(details))
        self.console.print('1. Allow once')
        self.console.print('2. Allow similar actions this session')
        self.console.print('3. Deny')
        answer = self.console.input('Select 1-3 (default 3): ')
        return {
            '1': 'allow_once',
            '2': 'allow_session',
            'y': 'allow_once',
            'yes': 'allow_once',
        }.get(answer.strip().casefold(), 'deny')

    def _select_inline(
        self,
        message: str,
        choices: tuple[tuple[str, str, str], ...],
        *,
        current: str | None = None,
    ) -> str | None:
        if self.prompt_session is None:
            return None

        def open_menu() -> None:
            get_app().current_buffer.start_completion(select_first=False)

        try:
            answer = self.prompt_session.prompt(
                [('ansibrightcyan bold', message)],
                completer=InlineChoiceCompleter(choices, current=current),
                complete_while_typing=True,
                complete_style=CompleteStyle.COLUMN,
                reserve_space_for_menu=len(choices) + 1,
                bottom_toolbar='\u2191/\u2193 select  Enter confirm  Esc cancel',
                key_bindings=INLINE_CHOICE_KEY_BINDINGS,
                pre_run=open_menu,
            ).strip()
        except (KeyboardInterrupt, EOFError):
            return None
        allowed = {value for value, _, _ in choices}
        return answer if answer in allowed else None

    async def wait_for_interrupt(self) -> None:
        '''Wait until Esc is pressed while a response is running.'''
        await wait_for_escape_key()

    def show_interrupted(self) -> None:
        '''Confirm that only the active response was stopped.'''
        self.console.print('[yellow]Interrupted current response.[/]')

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

    def show_context(self, stats: ContextStats) -> None:
        '''Render estimated input categories and remaining context capacity.'''
        table = Table.grid(padding=(0, 2))
        table.add_column(style='dim', no_wrap=True)
        table.add_column(style='bright_white', justify='right')
        table.add_row('stored messages', f'{stats.stored_messages:,}')
        table.add_row(
            'stored history',
            f'~{stats.stored_tokens:,} tokens '
            f'({stats.stored_characters:,} characters)',
        )
        table.add_row(
            'stored tool results',
            f'{stats.stored_tool_characters:,} chars',
        )
        table.add_row('request messages', f'{stats.message_count:,}')
        table.add_row('system', f'~{stats.system_tokens:,} tokens')
        table.add_row('repository', f'~{stats.repository_tokens:,} tokens')
        table.add_row('tools', f'~{stats.tool_schema_tokens:,} tokens')
        table.add_row(
            'request history',
            f'~{stats.history_tokens:,} tokens '
            f'({stats.estimated_characters:,} characters)',
        )
        table.add_row(
            'request tool results',
            f'{stats.tool_result_characters:,} chars',
        )
        table.add_row('estimated input', f'~{stats.estimated_tokens:,} tokens')
        table.add_row(
            'reserved output',
            f'{stats.reserved_output_tokens:,} tokens',
        )
        table.add_row(
            'projected total',
            f'~{stats.projected_tokens:,} tokens',
        )
        if stats.context_window_tokens is None:
            table.add_row('context window', 'not configured')
            table.add_row('remaining', 'unavailable')
        else:
            table.add_row(
                'context window',
                f'{stats.context_window_tokens:,} tokens',
            )
            table.add_row(
                'remaining',
                f'~{stats.remaining_tokens or 0:,} tokens',
            )
            table.add_row(
                'projected utilization',
                f'{(stats.utilization or 0) * 100:.1f}%',
            )
        self.console.print('[bold bright_cyan]Context[/]')
        self.console.print(table)
        self.console.print(
            '[dim]Request values include cheap compaction and match the '
            'automatic compaction threshold. Stored history remains available '
            'locally. The next user prompt is not included.[/]'
        )

    def show_compaction(self, report: CompactionReport) -> None:
        '''Render the result of an explicit /compact request.'''
        if report.success:
            self.console.print(
                '[bold green]Context compacted[/]  '
                f'{report.before_characters:,} → '
                f'{report.after_characters:,} characters'
            )
            if report.transcript_path:
                self.console.print(
                    f'[dim]Full transcript: {report.transcript_path}[/]'
                )
            elif report.reason:
                self.console.print(f'[dim]{report.reason}[/]')
            return
        self.console.print(
            f'[bold red]Compaction failed[/] [dim]{escape(report.reason)}[/]'
        )

    def show_notice(self, title: str, content: str) -> None:
        '''Render a local command result without starting a model turn.'''
        self.console.print(f'[bold bright_cyan]{escape(title)}[/]')
        self.console.print(escape(content))


def phase_label(phase: AgentPhase) -> str:
    return {
        AgentPhase.THINKING: '正在思考',
        AgentPhase.PREPARING_TOOLS: '准备调用工具',
        AgentPhase.EXECUTING_TOOLS: '正在执行工具',
        AgentPhase.CHECKING_RESULT: '检查结果',
        AgentPhase.RECOVERING: '失败恢复',
        AgentPhase.COMPLETED: '完成',
        AgentPhase.FAILED: '失败',
    }[phase]


def tool_action_label(tool_name: str) -> str:
    if tool_name in {
        'read_file',
        'list_directory',
        'find_files',
        'grep',
        'git_status',
        'git_diff',
    }:
        return '查看项目'
    if tool_name in {
        'write_file',
        'write_file_chunk',
        'replace_text',
        'apply_patch',
        'create_directory',
    }:
        return '修改文件'
    if tool_name == 'verify':
        return '运行验证'
    if tool_name == 'run_command':
        return '运行命令'
    if tool_name in {'todo_write', 'task_plan', 'task_update'}:
        return '更新计划'
    if tool_name == 'finish_task':
        return '整理结果'
    return '执行工具'


def phase_status(phase: AgentPhase, reason: str) -> str:
    if reason.startswith('executing_tool:'):
        tool_name = reason.split(':', 1)[1]
        return f'正在{tool_action_label(tool_name)}'
    if reason in {'preparing_model_request', 'preparing_recovery_request'}:
        return '正在准备下一步'
    if reason == 'model_requested_tool_calls':
        return '正在准备工具'
    if reason == 'turn_completed':
        return '完成'
    if phase is AgentPhase.RECOVERING:
        lowered = reason.casefold()
        if 'verification' in lowered or 'verify' in lowered:
            return '正在修复验证失败'
        if 'final' in lowered or 'completion' in lowered:
            return '正在整理结果'
        if 'token' in lowered or 'context' in lowered:
            return '正在整理进度'
        return '正在恢复'
    return phase_label(phase)


def tool_group_action_summary(group: _ToolTimelineBlock) -> str:
    actions = [
        tool_action_label(activity.tool_call.name)
        for activity in group.activities
    ]
    unique_actions = list(dict.fromkeys(actions))
    if len(unique_actions) == 1:
        return unique_actions[0]
    if len(unique_actions) <= 3:
        return '、'.join(unique_actions)
    return '混合操作'


def tool_group_title(
    group: _ToolTimelineBlock,
    *,
    pending: bool,
    failed: bool,
) -> str:
    count = len(group.activities)
    summary = tool_group_action_summary(group)
    if pending:
        if count == 1:
            return f'正在{summary}'
        return f'正在运行 {count} 个工具 · {summary}'
    if failed:
        return f'工具执行完成，存在失败 · {summary}'
    if count == 1:
        return f'已运行 {group.activities[0].tool_call.name} · {summary}'
    return f'已运行 {count} 个工具 · {summary}'


def tool_result_annotation(result: ToolResult) -> str:
    decision = result.metadata.get('transaction_decision')
    cache_hit = bool(result.metadata.get('cache_hit')) or decision == 'cache_hit'
    if cache_hit:
        return f' — 复用缓存: {result.summary}'
    if decision == 'blocked':
        return f' — 阶段拦截: {result.summary}'
    phase = result.metadata.get('transaction_phase')
    if isinstance(phase, str) and phase:
        return f' — {phase}: {result.summary}'
    return f' — {result.summary}'


SENSITIVE_ARGUMENT_KEYS = frozenset(
    {
        'api_key',
        'apikey',
        'authorization',
        'auth',
        'content',
        'new_text',
        'old_text',
        'password',
        'secret',
        'token',
    }
)


def summarize_tool_arguments(
    arguments: dict[str, Any],
    *,
    max_characters: int = 120,
) -> str:
    '''Render compact, terminal-safe tool arguments.'''
    summarized = summarize_argument_value(arguments)
    rendered = json.dumps(summarized, ensure_ascii=False, default=str)
    return truncate_middle(rendered, max_characters)


def summarize_argument_value(value: object) -> object:
    if isinstance(value, dict):
        compact: dict[str, object] = {}
        for key, item in value.items():
            normalized = str(key).casefold()
            if normalized in SENSITIVE_ARGUMENT_KEYS:
                compact[str(key)] = summarize_sensitive_value(item)
            else:
                compact[str(key)] = summarize_argument_value(item)
        return compact
    if isinstance(value, list):
        if len(value) > 6:
            return [
                *(summarize_argument_value(item) for item in value[:5]),
                f'... {len(value) - 5} more',
            ]
        return [summarize_argument_value(item) for item in value]
    if isinstance(value, str):
        if '\n' in value:
            lines = value.count('\n') + 1
            return f'<{len(value)} chars, {lines} lines>'
        if len(value) > 80:
            return truncate_middle(value, 80)
        return value
    return value


def summarize_sensitive_value(value: object) -> str:
    if isinstance(value, str):
        lines = value.count('\n') + 1
        return f'<redacted {len(value)} chars, {lines} lines>'
    return '<redacted>'


def truncate_middle(text: str, max_characters: int) -> str:
    if len(text) <= max_characters:
        return text
    if max_characters <= 12:
        return text[:max_characters]
    head = max_characters // 2 - 2
    tail = max_characters - head - 5
    return f'{text[:head]} ... {text[-tail:]}'


def summarize_diagnostic(
    diagnostic: str,
    *,
    max_lines: int = 10,
    max_characters: int = 800,
) -> str:
    '''Keep actionable failure output visible without taking over the frame.'''
    text = diagnostic.strip()
    if not text:
        return ''
    lines = text.splitlines()
    truncated = False
    if len(lines) > max_lines:
        kept = max(1, max_lines // 2)
        lines = [
            *lines[:kept],
            f'... {len(text.splitlines()) - kept * 2} lines omitted ...',
            *lines[-kept:],
        ]
        truncated = True
    text = '\n'.join(lines)
    if len(text) > max_characters:
        text = truncate_middle(text, max_characters)
        truncated = True
    if truncated:
        text += '\n...[diagnostic shortened]...'
    return text


class StreamingResponseView:
    '''Update streamed Markdown and exact usage in place.'''

    def __init__(
        self,
        console: Console,
        *,
        approval_selector: ApprovalSelector | None = None,
    ) -> None:
        self.console = console
        self.approval_selector = approval_selector
        self.timeline: list[_TimelineBlock] = []
        self.usage: TokenUsage | None = None
        self.request_usage: TokenUsage | None = None
        self.model_calls = 0
        self.completed = False
        self.interrupted = False
        self.result: TurnResult | None = None
        self.phase: AgentPhase | None = None
        self.phase_reason = ''
        self._last_refresh_at = 0.0
        self._refresh_interval_seconds = 0.08
        self.live = Live(
            self._render(),
            console=console,
            refresh_per_second=16,
            vertical_overflow='ellipsis',
            transient=False,
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
        '''Append one provider text delta and refresh on a bounded cadence.'''
        if (
            self.timeline
            and isinstance(self.timeline[-1], _TextTimelineBlock)
        ):
            self.timeline[-1].text += text
        else:
            self.timeline.append(_TextTimelineBlock(text=text))
        self._schedule_update()

    def update_usage(
        self,
        usage: TokenUsage,
        *,
        request_usage: TokenUsage | None = None,
        model_calls: int = 1,
    ) -> None:
        '''Refresh the exact usage snapshot reported by the provider.'''
        self.usage = usage
        self.request_usage = request_usage
        self.model_calls = model_calls
        self._schedule_update()

    def start_tool(self, tool_call: ToolCall) -> None:
        '''Show a model-requested tool while it is executing.'''
        if (
            self.timeline
            and isinstance(self.timeline[-1], _ToolTimelineBlock)
        ):
            group = self.timeline[-1]
        else:
            group = _ToolTimelineBlock()
            self.timeline.append(group)
        group.activities.append(_ToolActivity(tool_call=tool_call))
        self._schedule_update(force=True)

    def complete_tool(
        self,
        tool_call: ToolCall,
        result: ToolResult,
    ) -> None:
        '''Replace one pending tool activity with its result summary.'''
        for block in reversed(self.timeline):
            if not isinstance(block, _ToolTimelineBlock):
                continue
            for activity in reversed(block.activities):
                if activity.tool_call.id == tool_call.id:
                    activity.result = result
                    self._schedule_update(force=True)
                    return
        self._schedule_update(force=True)

    def complete(self, result: TurnResult) -> None:
        '''Finalize the view with validated text and exact final usage.'''
        visible_text = ''.join(
            block.text
            for block in self.timeline
            if isinstance(block, _TextTimelineBlock)
        ).strip()
        final_text_is_present = (
            visible_text == result.text
            or (
                bool(self.timeline)
                and isinstance(self.timeline[-1], _TextTimelineBlock)
                and self.timeline[-1].text.strip() == result.text
            )
        )
        if result.text and not final_text_is_present:
            self.timeline.append(_TextTimelineBlock(text=result.text))
        self.usage = result.usage
        self.request_usage = result.last_request_usage
        self.model_calls = result.model_calls
        self.result = result
        self.completed = True
        self._schedule_update(force=True)

    def block_completion(self, reasons: tuple[str, ...]) -> None:
        '''Show why a tentative final answer was rejected by the runtime.'''
        details = '\n'.join(f'- {reason}' for reason in reasons)
        self.timeline.append(
            _TextTimelineBlock(
                text=f'Completion check: continuing work.\n\n{details}'
            )
        )
        self._schedule_update(force=True)

    def interrupt(self) -> None:
        '''Finalize the live frame after a user-requested interruption.'''
        self.interrupted = True
        self.completed = True
        self.timeline.append(
            _NoticeTimelineBlock(
                title='回答已中断',
                lines=('已完成的工具操作不会回滚',),
                style='yellow',
            )
        )
        self._schedule_update(force=True)

    def update_phase(self, phase: AgentPhase, reason: str) -> None:
        '''Show the current orchestration phase without polluting the timeline.'''
        self.phase = phase
        self.phase_reason = reason
        self._schedule_update()

    def request_permission(self, tool_call: ToolCall, effect: object) -> str:
        '''Pause live rendering and ask whether one sensitive tool may run.'''
        self.live.stop()
        try:
            if self.approval_selector is not None:
                return self.approval_selector(tool_call, effect)
            details = permission_request_details(tool_call, effect)
            self.console.print('[bold yellow]Permission required[/]')
            self.console.print(Text(details))
            self.console.print('1. Allow once')
            self.console.print('2. Allow similar actions this session')
            self.console.print('3. Deny')
            answer = self.console.input(
                'Select 1-3 (default 3): '
            )
            return {
                '1': 'allow_once',
                '2': 'allow_session',
                'y': 'allow_once',
                'yes': 'allow_once',
            }.get(answer.strip().casefold(), 'deny')
        finally:
            self.live.start(refresh=True)

    def compact_context(self, event: ContextCompacted) -> None:
        '''Show automatic context compaction while the turn continues.'''
        lines = (
            (
                f'{event.before_characters:,} -> '
                f'{event.after_characters:,} characters'
            ),
            *(
                (f'Full transcript: {event.transcript_path}',)
                if event.transcript_path
                else ()
            ),
        )
        self.timeline.append(
            _NoticeTimelineBlock(
                title='Context compacted',
                lines=lines,
                style='green',
            )
        )
        self._schedule_update(force=True)

    def _schedule_update(self, *, force: bool = False) -> None:
        now = monotonic()
        should_refresh = (
            force
            or now - self._last_refresh_at >= self._refresh_interval_seconds
        )
        if should_refresh:
            self._last_refresh_at = now
        self.live.update(self._render(), refresh=should_refresh)

    def _render(self) -> Group:
        content = self._render_timeline()
        renderables: list[object] = [content]
        if not self.completed and self.phase is not None:
            renderables.append(
                Text(
                    phase_status(self.phase, self.phase_reason),
                    style='dim bright_cyan',
                )
            )
        if not self.completed:
            renderables.append(Text('Esc 中断当前回答', style='dim'))
        if self.result is not None and (
            self.result.changed_paths
            or self.result.verification is not None
            or self.result.status != 'completed'
        ):
            renderables.append(completion_evidence_summary(self.result))
        renderables.append(
            token_usage_summary(
                self.usage,
                streaming=not self.completed,
                request_usage=self.request_usage,
                model_calls=self.model_calls,
            )
        )
        return Group(*renderables)

    def _render_timeline(self) -> Group | Spinner:
        if not self.timeline:
            return Spinner(
                'dots',
                Text('Waiting for model...', style='bright_cyan'),
            )
        if self.completed:
            renderables = [
                self._render_timeline_block(block, line_budget=None)
                for block in self.timeline
            ]
        else:
            renderables = self._render_streaming_timeline()
        separated: list[object] = []
        for position, renderable in enumerate(renderables):
            if position:
                separated.append(Text(''))
            separated.append(renderable)
        return Group(*separated)

    def _render_streaming_timeline(self) -> list[object]:
        remaining_lines = max(4, self.console.height - 4)
        selected: list[tuple[_TimelineBlock, int]] = []
        for block in reversed(self.timeline):
            if isinstance(block, _ToolTimelineBlock):
                desired_lines = min(len(block.activities), 6) + 1
            elif isinstance(block, _NoticeTimelineBlock):
                desired_lines = len(block.lines) + 1
            else:
                desired_lines = max(1, len(block.text.splitlines()))
            line_budget = min(remaining_lines, desired_lines)
            selected.append((block, max(1, line_budget)))
            remaining_lines -= line_budget
            if remaining_lines <= 0:
                break
        selected.reverse()

        renderables: list[object] = []
        if len(selected) < len(self.timeline):
            renderables.append(Text('… earlier activity', style='dim'))
        renderables.extend(
            self._render_timeline_block(block, line_budget=line_budget)
            for block, line_budget in selected
        )
        return renderables

    def _render_timeline_block(
        self,
        block: _TimelineBlock,
        *,
        line_budget: int | None,
    ) -> object:
        if isinstance(block, _TextTimelineBlock):
            if self.completed:
                return Markdown(block.text)
            max_lines = line_budget or 4
            return Text(
                streaming_preview(
                    block.text,
                    max_lines=max_lines,
                    max_characters=max(
                        200,
                        self.console.width * max_lines // 2,
                    ),
                )
            )
        if isinstance(block, _NoticeTimelineBlock):
            rendered = Text()
            rendered.append('✓ ', style=f'bold {block.style}')
            rendered.append(block.title, style=f'bold {block.style}')
            for line in block.lines:
                rendered.append(f'\n  {line}', style='dim')
            return rendered
        activity_limit = (
            None if line_budget is None else max(1, line_budget - 1)
        )
        return self._render_tool_group(block, limit=activity_limit)

    def _render_tool_group(
        self,
        group: _ToolTimelineBlock,
        *,
        limit: int | None,
    ) -> Text:
        rendered = Text()
        activities = group.activities
        hidden_count = 0
        if limit is not None and len(activities) > limit:
            hidden_count = len(activities) - limit
            activities = activities[-limit:]

        pending = any(activity.result is None for activity in group.activities)
        failed = any(
            activity.result is not None and not activity.result.success
            for activity in group.activities
        )
        if pending:
            rendered.append('● ', style='bold bright_cyan')
            rendered.append(
                tool_group_title(group, pending=pending, failed=failed),
                style='bold',
            )
        elif failed:
            rendered.append('× ', style='bold red')
            rendered.append(
                tool_group_title(group, pending=pending, failed=failed),
                style='bold',
            )
        else:
            rendered.append('✓ ', style='bold green')
            rendered.append(
                tool_group_title(group, pending=pending, failed=failed),
                style='dim',
            )
        rendered.append('\n')

        if hidden_count:
            rendered.append(
                f'  … {hidden_count} 个更早的工具调用\n',
                style='dim',
            )
        for position, activity in enumerate(activities):
            is_last = position == len(activities) - 1
            rendered.append('  └─ ' if is_last else '  ├─ ', style='dim')
            result = activity.result
            if result is None:
                rendered.append('● ', style='bright_cyan')
            elif result.success:
                rendered.append('✓ ', style='green')
            else:
                rendered.append('× ', style='red')
            rendered.append(activity.tool_call.name, style='bold')
            arguments = summarize_tool_arguments(
                activity.tool_call.arguments
            )
            rendered.append(f' {arguments}', style='dim')
            if result is not None:
                rendered.append(tool_result_annotation(result), style='dim')
                diagnostic = summarize_diagnostic(result.content)
                if not result.success and diagnostic:
                    diagnostic = diagnostic.replace('\n', '\n       ')
                    rendered.append(
                        f'\n       {diagnostic}',
                        style='dim red',
                    )
            if not is_last:
                rendered.append('\n')
        return rendered


def streaming_preview(
    text: str,
    *,
    max_lines: int,
    max_characters: int,
) -> str:
    '''Return a bounded tail so a live frame stays inside the terminal.'''
    lines = text.splitlines()
    preview = '\n'.join(lines[-max_lines:])
    truncated = len(lines) > max_lines
    if len(preview) > max_characters:
        preview = preview[-max_characters:]
        truncated = True
    if truncated:
        preview = f'…\n{preview.lstrip()}'
    return preview


async def wait_for_escape_key() -> None:
    '''Poll the active terminal for Esc without blocking the event loop.'''
    if not sys.stdin.isatty():
        await asyncio.Future()
        return
    if os.name == 'nt':
        import msvcrt

        while True:
            while msvcrt.kbhit():
                character = msvcrt.getwch()
                if character == '\x1b':
                    return
                try:
                    msvcrt.ungetwch(character)
                except OSError:
                    pass
                await asyncio.Future()
                return
            await asyncio.sleep(0.05)

    file_descriptor = sys.stdin.fileno()
    with terminal_cbreak(file_descriptor):
        while True:
            readable, _, _ = select.select([file_descriptor], [], [], 0)
            if readable and os.read(file_descriptor, 1) == b'\x1b':
                return
            await asyncio.sleep(0.05)


@contextmanager
def terminal_cbreak(file_descriptor: int) -> Iterator[None]:
    '''Temporarily enable single-key reads and restore terminal settings.'''
    import termios
    import tty

    previous = termios.tcgetattr(file_descriptor)
    tty.setcbreak(file_descriptor)
    try:
        yield
    finally:
        termios.tcsetattr(file_descriptor, termios.TCSADRAIN, previous)


def permission_answer_allows(answer: str) -> bool:
    '''Return whether an interactive permission answer approves a tool call.'''
    return answer.strip().casefold() in {
        'y',
        'yes',
        '是',
        '同意',
        '允许',
        '可以',
        '确认',
        'approve',
        'allow',
    }


def permission_request_details(tool_call: ToolCall, effect: object) -> str:
    arguments = json.dumps(
        tool_call.arguments,
        ensure_ascii=False,
        indent=2,
        default=str,
    )
    if len(arguments) > 3_000:
        arguments = arguments[:2_997] + '...'
    return (
        f'Tool: {tool_call.name}\n'
        f'Effect: {effect}\n\n'
        f'Arguments:\n{arguments}\n\n'
        'Allow this operation?'
    )


def token_usage_summary(
    usage: TokenUsage | None,
    *,
    streaming: bool,
    request_usage: TokenUsage | None = None,
    model_calls: int = 1,
) -> Text:
    '''Build the live or final token usage line.'''
    if request_usage is not None:
        prefix = (
            '\u21b3 turn cumulative (streaming)'
            if streaming
            else '\u21b3 turn cumulative'
        )
    else:
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
    if request_usage is not None:
        if streaming and request_usage.total_tokens == 0:
            summary.append(
                '\n  last request  waiting for provider usage ...',
                style='dim yellow',
            )
            summary.append(f'  {model_calls} model calls', style='dim')
            return summary
        summary.append('\n  last request  input ', style='dim')
        summary.append(
            f'{request_usage.total_input_tokens:,}',
            style='bright_cyan',
        )
        summary.append('  output ', style='dim')
        summary.append(
            f'{request_usage.output_tokens:,}',
            style='bright_cyan',
        )
        summary.append('  turn total above', style='dim')
        summary.append(
            f'  {model_calls} model calls',
            style='dim',
        )
    return summary


def completion_evidence_summary(result: TurnResult) -> Text:
    '''Render the objective completion state below the final answer.'''
    rendered = Text()
    status_style = 'green' if result.status == 'completed' else 'red'
    rendered.append('↳ task ', style='dim')
    rendered.append(result.status, style=f'bold {status_style}')
    if result.changed_paths:
        rendered.append('  changed ', style='dim')
        rendered.append(', '.join(result.changed_paths), style='bright_white')
    if result.verification is not None:
        evidence = result.verification
        rendered.append('\n  verify ', style='dim')
        rendered.append(evidence.command, style='bright_white')
        rendered.append(
            f'  exit {evidence.exit_code}  {evidence.duration_seconds:.3f}s',
            style='dim',
        )
    for reason in result.completion_reasons:
        rendered.append(f'\n  × {reason}', style='red')
    return rendered
