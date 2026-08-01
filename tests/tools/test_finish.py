'''Tests for the model-declared completion tool.'''

import asyncio
from pathlib import Path

from forge.tools.finish import FinishTaskTool


def run(tool: FinishTaskTool, arguments: dict[str, object]):
    return asyncio.run(tool.run(arguments))


def test_finish_task_returns_structured_declaration(tmp_path: Path) -> None:
    result = run(
        FinishTaskTool(tmp_path),
        {
            'task_kind': 'change',
            'status': 'completed',
            'summary': 'Implemented and verified.',
        },
    )

    assert result.success
    assert result.metadata['finish_task'] is True
    assert result.metadata['task_kind'] == 'change'


def test_blocked_finish_requires_reasons(tmp_path: Path) -> None:
    result = run(
        FinishTaskTool(tmp_path),
        {
            'task_kind': 'change',
            'status': 'blocked',
            'summary': 'Could not continue.',
        },
    )

    assert result.success is False
    assert result.error is not None
    assert result.error.code == 'invalid_arguments'


def test_task_completed_rejects_structured_remaining_work(
    tmp_path: Path,
) -> None:
    result = run(
        FinishTaskTool(tmp_path),
        {
            'task_kind': 'change',
            'status': 'task_completed',
            'summary': 'Only part is implemented.',
            'remaining_work': ['Implement the second deliverable'],
        },
    )

    assert result.success is False
    assert result.error is not None
    assert result.error.code == 'invalid_arguments'


def test_step_completed_is_a_structured_nonterminal_outcome(
    tmp_path: Path,
) -> None:
    result = run(
        FinishTaskTool(tmp_path),
        {
            'task_kind': 'change',
            'status': 'step_completed',
            'summary': 'Completed the current plan step.',
            'remaining_work': ['Run the next plan step'],
        },
    )

    assert result.success is True
    assert result.metadata['task_outcome'] == 'step_completed'
    assert result.metadata['remaining_work'] == ['Run the next plan step']
