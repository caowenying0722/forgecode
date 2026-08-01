'''Shared subprocess execution primitives for runtime and tool layers.'''

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from hashlib import sha256
import locale
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
    stdout_encoding: str = 'utf-8'
    stderr_encoding: str = 'utf-8'
    decode_warnings: tuple[str, ...] = ()
    stdout_raw_hash: str = ''
    stderr_raw_hash: str = ''


@dataclass(frozen=True, slots=True)
class DecodedProcessOutput:
    text: str
    encoding: str
    warning: str = ''


def decode_process_output(
    data: bytes,
    *,
    stream: str,
    preferred_encoding: str | None = None,
    windows: bool | None = None,
) -> DecodedProcessOutput:
    '''Decode one subprocess stream without losing execution diagnostics.'''
    if not data:
        return DecodedProcessOutput('', 'utf-8')
    preferred = preferred_encoding or locale.getpreferredencoding(False)
    candidates = ['utf-8', preferred]
    is_windows = windows if windows is not None else os.name == 'nt'
    if is_windows:
        candidates.extend(('cp936', 'gbk'))
    unique_candidates = tuple(
        dict.fromkeys(
            encoding.strip().casefold()
            for encoding in candidates
            if encoding and encoding.strip()
        )
    )
    failures: list[str] = []
    for encoding in unique_candidates:
        try:
            text = data.decode(encoding, errors='strict')
        except (LookupError, UnicodeDecodeError):
            failures.append(encoding)
            continue
        warning = ''
        if failures:
            warning = (
                f'{stream} was decoded using {encoding} after strict decode '
                f'failed for {", ".join(failures)}.'
            )
        return DecodedProcessOutput(text, encoding, warning)
    fallback = unique_candidates[0] if unique_candidates else 'utf-8'
    return DecodedProcessOutput(
        data.decode(fallback, errors='replace'),
        fallback,
        f'{stream} contained invalid bytes; undecodable sequences were replaced.',
    )


async def run_process(
    command: list[str] | str,
    *,
    cwd: Path,
    timeout_seconds: float,
    input_text: str | None = None,
    shell: bool = False,
    env: dict[str, str] | None = None,
) -> ProcessResult:
    started = perf_counter()
    stdin = asyncio.subprocess.PIPE if input_text is not None else None
    process_group = (
        {'creationflags': 0x00000200}
        if os.name == 'nt'
        else {'start_new_session': True}
    )
    process_env = None if env is None else {**os.environ, **env}
    if shell:
        if not isinstance(command, str):
            raise TypeError('Shell commands must be strings.')
        process = await asyncio.create_subprocess_shell(
            command,
            cwd=cwd,
            stdin=stdin,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=process_env,
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
            env=process_env,
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
    stdout = decode_process_output(stdout_bytes, stream='stdout')
    stderr = decode_process_output(stderr_bytes, stream='stderr')
    decode_warnings = tuple(
        item for item in (stdout.warning, stderr.warning) if item
    )
    return ProcessResult(
        exit_code=process.returncode if process.returncode is not None else -1,
        stdout=stdout.text,
        stderr=stderr.text,
        duration_seconds=perf_counter() - started,
        timed_out=timed_out,
        stdout_encoding=stdout.encoding,
        stderr_encoding=stderr.encoding,
        decode_warnings=decode_warnings,
        stdout_raw_hash=sha256(stdout_bytes).hexdigest(),
        stderr_raw_hash=sha256(stderr_bytes).hexdigest(),
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
        'stdout_encoding': result.stdout_encoding,
        'stderr_encoding': result.stderr_encoding,
        'decode_warnings': list(result.decode_warnings),
        'stdout_raw_hash': result.stdout_raw_hash,
        'stderr_raw_hash': result.stderr_raw_hash,
    }


def render_process_output(result: ProcessResult) -> str:
    sections: list[str] = []
    if result.stdout:
        sections.append(f'stdout:\n{result.stdout.rstrip()}')
    if result.stderr:
        sections.append(f'stderr:\n{result.stderr.rstrip()}')
    return '\n\n'.join(sections)
