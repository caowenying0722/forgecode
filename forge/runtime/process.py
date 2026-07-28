'''Shared subprocess execution primitives for runtime and tool layers.'''

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import os
from pathlib import Path
import signal
from time import perf_counter


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
    started = perf_counter()
    stdin = asyncio.subprocess.PIPE if input_text is not None else None
    process_group = (
        {'creationflags': 0x00000200}
        if os.name == 'nt'
        else {'start_new_session': True}
    )
    if shell:
        if not isinstance(command, str):
            raise TypeError('Shell commands must be strings.')
        process = await asyncio.create_subprocess_shell(
            command,
            cwd=cwd,
            stdin=stdin,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            **process_group,
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
            **process_group,
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
        stdout_bytes, stderr_bytes = await terminate_process_tree(process)
        timed_out = True
    except asyncio.CancelledError:
        await terminate_process_tree(process)
        raise
    return ProcessResult(
        exit_code=process.returncode if process.returncode is not None else -1,
        stdout=stdout_bytes.decode('utf-8', errors='replace'),
        stderr=stderr_bytes.decode('utf-8', errors='replace'),
        duration_seconds=perf_counter() - started,
        timed_out=timed_out,
    )


async def terminate_process_tree(
    process: asyncio.subprocess.Process,
) -> tuple[bytes, bytes]:
    '''Terminate a foreground command and descendants, then drain its pipes.'''
    if process.returncode is None:
        if os.name == 'nt':
            killer = await asyncio.create_subprocess_exec(
                'taskkill',
                '/PID',
                str(process.pid),
                '/T',
                '/F',
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await killer.communicate()
        else:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
    return await process.communicate()


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
