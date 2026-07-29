'''Deterministic control-plane state for the Agent Loop.'''

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from forge.tools.base import ToolResult


class AgentControlState(str, Enum):
    INIT = 'init'
    EXPLORING = 'exploring'
    PLANNING = 'planning'
    IMPLEMENTING = 'implementing'
    VERIFYING = 'verifying'
    RECOVERING = 'recovering'
    DONE = 'done'
    BLOCKED = 'blocked'


@dataclass(slots=True)
class AgentController:
    '''Map structured tool failures onto bounded runtime state transitions.'''

    state: AgentControlState = AgentControlState.INIT
    planning_recovery: bool = False
    planning_recovery_calls: int = 0

    def begin_turn(self) -> None:
        self.state = AgentControlState.EXPLORING
        self.planning_recovery = False
        self.planning_recovery_calls = 0

    def enter_planning_recovery(self) -> None:
        self.state = AgentControlState.PLANNING
        self.planning_recovery = True
        self.planning_recovery_calls += 1

    def observe_tool_result(self, tool_name: str, result: ToolResult) -> None:
        if tool_name == 'todo_write' and result.success:
            self.planning_recovery = False
            self.state = AgentControlState.IMPLEMENTING
        elif tool_name == 'verify' and result.success:
            self.state = AgentControlState.VERIFYING


def is_todo_required_result(result: ToolResult) -> bool:
    return (
        result.error is not None
        and result.error.code == 'todo_required'
        and bool(result.metadata.get('todo_required'))
    )


def batch_requires_todo(results: list[tuple[object, ToolResult]]) -> bool:
    return bool(results) and any(
        is_todo_required_result(result) for _, result in results
    )


def build_planning_recovery_feedback(
    task_context: str,
    *,
    attempt: int,
) -> dict[str, str]:
    return {
        'role': 'user',
        'content': (
            f'{task_context}\n\n'
            '[ForgeCode Planning Recovery]\n'
            'A write or process tool was rejected because this complex task '
            'requires an explicit TODO plan first. The next request exposes '
            'only todo_write. Call todo_write with a short working plan and '
            'exactly one in_progress item, then continue implementation in '
            'the following request. Do not retry write/process tools before '
            f'todo_write succeeds. Planning recovery attempt: {attempt}.'
        ),
    }
