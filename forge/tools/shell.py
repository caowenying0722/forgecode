'''Local command execution tool and shared subprocess helpers.'''

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
import re
from time import perf_counter

from pydantic import Field

from forge.tools.base import (
    Tool,
    ToolExecutionError,
    ToolInput,
    ToolResult,
    display_path,
    resolve_repository_path,
)


@dataclass(frozen=True, slots=True)
class ProcessResult:
    exit_code: int
    stdout: str
    stderr: str
    duration_seconds: float
    timed_out: bool = False


async def run_process(
    command: list[str] | str,
    *,
    cwd: Path,
    timeout_seconds: float,
    input_text: str | None = None,
    shell: bool = False,
) -> ProcessResult:
    '''Run one subprocess and capture deterministic result fields.'''
    started = perf_counter()
    stdin = asyncio.subprocess.PIPE if input_text is not None else None
    if shell:
        if not isinstance(command, str):
            raise TypeError('Shell commands must be strings.')
        process = await asyncio.create_subprocess_shell(
            command,
            cwd=cwd,
            stdin=stdin,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    else:
        if isinstance(command, str):
            raise TypeError('Executable commands must be argument lists.')
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=cwd,
            stdin=stdin,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            process.communicate(
                input_text.encode('utf-8') if input_text is not None else None
            ),
            timeout=timeout_seconds,
        )
        timed_out = False
    except TimeoutError:
        process.kill()
        stdout_bytes, stderr_bytes = await process.communicate()
        timed_out = True

    return ProcessResult(
        exit_code=process.returncode if process.returncode is not None else -1,
        stdout=stdout_bytes.decode('utf-8', errors='replace'),
        stderr=stderr_bytes.decode('utf-8', errors='replace'),
        duration_seconds=perf_counter() - started,
        timed_out=timed_out,
    )


def process_metadata(result: ProcessResult) -> dict[str, object]:
    return {
        'exit_code': result.exit_code,
        'stdout': result.stdout,
        'stderr': result.stderr,
        'duration_seconds': result.duration_seconds,
        'timed_out': result.timed_out,
    }


def render_process_output(result: ProcessResult) -> str:
    sections: list[str] = []
    if result.stdout:
        sections.append(f'stdout:\n{result.stdout.rstrip()}')
    if result.stderr:
        sections.append(f'stderr:\n{result.stderr.rstrip()}')
    return '\n\n'.join(sections)


class RunCommandInput(ToolInput):
    command: str = Field(min_length=1)
    cwd: str = '.'
    timeout_seconds: float = Field(default=120.0, gt=0, le=600)


class RunCommandTool(Tool[RunCommandInput]):
    name = 'run_command'
    description = (
        'Run a local shell command inside the repository for inspection, '
        'tests, builds, or validation. Do not write repository files through '
        'inline Node, Python, PowerShell, or shell redirection. Use write_file, '
        'replace_text, or apply_patch for source changes.'
    )
    input_model = RunCommandInput
    effect = 'process'

    async def execute(self, arguments: RunCommandInput) -> ToolResult:
        denied_reason = shell_file_write_reason(arguments.command)
        if denied_reason is not None:
            raise ToolExecutionError(
                'shell_file_write_denied',
                'run_command cannot be used to write repository files. '
                'Use write_file, replace_text, or apply_patch instead.',
                details={'detected': denied_reason},
            )
        cwd = resolve_repository_path(self.root, arguments.cwd)
        if not cwd.is_dir():
            raise ToolExecutionError(
                'not_a_directory',
                f'Command cwd is not a directory: {arguments.cwd}',
            )
        result = await run_process(
            arguments.command,
            cwd=cwd,
            timeout_seconds=arguments.timeout_seconds,
            shell=True,
        )
        metadata = {
            **process_metadata(result),
            'command': arguments.command,
            'cwd': display_path(self.root, cwd),
        }
        content = render_process_output(result)
        if result.timed_out:
            return ToolResult.fail(
                'command_timeout',
                f'Command timed out after {arguments.timeout_seconds:g}s.',
                content=content,
                metadata=metadata,
            )
        if result.exit_code != 0:
            return ToolResult.fail(
                'command_failed',
                f'Command exited with code {result.exit_code}.',
                content=content,
                metadata=metadata,
            )
        return ToolResult.ok(
            f'Command completed with exit code 0 in '
            f'{result.duration_seconds:.3f}s.',
            content=content,
            metadata=metadata,
        )


SCRIPT_WRITE_PATTERNS = (
    (
        re.compile(
            r'\b(?:writeFile|writeFileSync|appendFile|appendFileSync)\s*\(',
            re.IGNORECASE,
        ),
        'Node filesystem write API',
    ),
    (
        re.compile(r'\.(?:write_text|write_bytes)\s*\(', re.IGNORECASE),
        'Python pathlib write API',
    ),
    (
        re.compile(
            r'\bopen\s*\([^\n]*,\s*[\x27\x22](?:w|a|x|\+)',
            re.IGNORECASE,
        ),
        'Python writable open mode',
    ),
    (
        re.compile(
            r'\b(?:Set-Content|Add-Content|Out-File)\b',
            re.IGNORECASE,
        ),
        'PowerShell file-writing command',
    ),
)


def shell_file_write_reason(command: str) -> str | None:
    '''Detect common direct file-writing shortcuts before shell execution.'''
    for pattern, reason in SCRIPT_WRITE_PATTERNS:
        if pattern.search(command):
            return reason
    if has_unquoted_output_redirection(command):
        return 'shell output redirection'
    return None


def has_unquoted_output_redirection(command: str) -> bool:
    single_quoted = False
    double_quoted = False
    escaped = False
    for index, character in enumerate(command):
        if escaped:
            escaped = False
            continue
        if character == '\\':
            escaped = True
            continue
        if character == chr(39) and not double_quoted:
            single_quoted = not single_quoted
            continue
        if character == chr(34) and not single_quoted:
            double_quoted = not double_quoted
            continue
        if character != '>' or single_quoted or double_quoted:
            continue
        following = command[index + 1:index + 2]
        preceding = command[index - 1:index] if index else ''
        if following == '&' or preceding == '=':
            continue
        return True
    return False
