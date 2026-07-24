'''Tests for background run_command execution.'''

from __future__ import annotations

import asyncio
from pathlib import Path

from forge.runtime.background import BackgroundTaskManager
from forge.tools.shell import RunCommandTool


def test_run_command_can_start_background_process_and_collect_notification(
    tmp_path: Path,
) -> None:
    manager = BackgroundTaskManager(tmp_path)
    tool = RunCommandTool(tmp_path, background_manager=manager)

    async def scenario():
        result = await tool.run(
            {
                'command': 'python -c "print(123)"',
                'run_in_background': True,
            }
        )
        for _ in range(50):
            notifications = manager.collect_notifications()
            if notifications:
                return result, notifications
            await asyncio.sleep(0.05)
        return result, ()

    result, notifications = asyncio.run(scenario())

    assert result.success is True
    assert result.metadata['background_started'] is True
    assert result.metadata['background_id'].startswith('bg-')
    assert len(notifications) == 1
    assert '<task_notification>' in notifications[0]
    assert '<status>completed</status>' in notifications[0]
    assert '123' in notifications[0]
    assert (
        tmp_path
        / '.forge'
        / 'background'
        / f'{result.metadata["background_id"]}.log'
    ).exists()
