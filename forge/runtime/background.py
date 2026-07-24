'''Background process lifecycle for long-running run_command calls.'''

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter

from forge.tools.shell import ProcessResult, process_metadata, render_process_output


@dataclass(frozen=True, slots=True)
class BackgroundCommand:
    id: str
    command: str
    cwd: str
    status: str
    started_at: float
    completed_at: float | None = None
    exit_code: int | None = None
    log_path: str | None = None
    error: str | None = None


class BackgroundTaskManager:
    '''Run commands in asyncio tasks and collect completion notifications.'''

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.directory = self.root / '.forge' / 'background'
        self._counter = 0
        self._tasks: dict[str, BackgroundCommand] = {}
        self._results: dict[str, ProcessResult] = {}
        self._asyncio_tasks: set[asyncio.Task[None]] = set()

    def start_command(
        self,
        *,
        command: str,
        cwd: Path,
        display_cwd: str,
        timeout_seconds: float,
        input_text: str | None,
    ) -> BackgroundCommand:
        from forge.tools.shell import run_process

        self.directory.mkdir(parents=True, exist_ok=True)
        self._counter += 1
        background_id = f'bg-{self._counter:04d}'
        record = BackgroundCommand(
            id=background_id,
            command=command,
            cwd=display_cwd,
            status='running',
            started_at=perf_counter(),
        )
        self._tasks[background_id] = record

        async def worker() -> None:
            try:
                result = await run_process(
                    command,
                    cwd=cwd,
                    timeout_seconds=timeout_seconds,
                    input_text=input_text,
                    shell=True,
                )
                self._results[background_id] = result
                log_path = self._write_log(background_id, record, result)
                self._tasks[background_id] = replace(
                    record,
                    status='completed',
                    completed_at=perf_counter(),
                    exit_code=result.exit_code,
                    log_path=log_path,
                )
            except Exception as error:  # pragma: no cover - defensive boundary
                self._tasks[background_id] = replace(
                    record,
                    status='failed',
                    completed_at=perf_counter(),
                    error=f'{type(error).__name__}: {error}',
                )

        task = asyncio.create_task(worker())
        self._asyncio_tasks.add(task)
        task.add_done_callback(self._asyncio_tasks.discard)
        return record

    def collect_notifications(self) -> tuple[str, ...]:
        ready = [
            record
            for record in self._tasks.values()
            if record.status in {'completed', 'failed'}
        ]
        notifications: list[str] = []
        for record in ready:
            self._tasks.pop(record.id, None)
            result = self._results.pop(record.id, None)
            notifications.append(render_notification(record, result))
        return tuple(notifications)

    def _write_log(
        self,
        background_id: str,
        record: BackgroundCommand,
        result: ProcessResult,
    ) -> str:
        path = self.directory / f'{background_id}.log'
        content = [
            f'id: {background_id}',
            f'cwd: {record.cwd}',
            f'command: {record.command}',
            f'exit_code: {result.exit_code}',
            f'timed_out: {str(result.timed_out).lower()}',
            f'duration_seconds: {result.duration_seconds:.3f}',
            '',
            render_process_output(result),
        ]
        path.write_text('\n'.join(content).rstrip() + '\n', encoding='utf-8')
        return path.relative_to(self.root).as_posix()


def render_notification(
    record: BackgroundCommand,
    result: ProcessResult | None,
) -> str:
    if result is None:
        return (
            '<task_notification>\n'
            f'  <task_id>{record.id}</task_id>\n'
            f'  <status>{record.status}</status>\n'
            f'  <command>{record.command}</command>\n'
            f'  <summary>{record.error or "Background command failed."}</summary>\n'
            '</task_notification>'
        )
    status = 'completed' if result.exit_code == 0 and not result.timed_out else 'failed'
    summary = render_process_output(result).strip()
    if not summary:
        summary = '(no output)'
    if len(summary) > 500:
        summary = summary[:500] + '\n[truncated; full output is in the log file]'
    metadata = process_metadata(result)
    return (
        '<task_notification>\n'
        f'  <task_id>{record.id}</task_id>\n'
        f'  <status>{status}</status>\n'
        f'  <command>{record.command}</command>\n'
        f'  <exit_code>{metadata["exit_code"]}</exit_code>\n'
        f'  <timed_out>{str(metadata["timed_out"]).lower()}</timed_out>\n'
        f'  <log_path>{record.log_path or ""}</log_path>\n'
        f'  <summary>{escape_notification_text(summary)}</summary>\n'
        '</task_notification>'
    )


def escape_notification_text(value: str) -> str:
    return (
        value.replace('&', '&amp;')
        .replace('<', '&lt;')
        .replace('>', '&gt;')
    )
