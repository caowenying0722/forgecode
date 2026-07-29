'''Deterministic control-plane state for the Agent Loop.'''

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from forge.runtime.intent import InitialToolSurface, TaskContract
from forge.tools.base import ToolResult


class AgentControlState(str, Enum):
    INIT = 'init'
    TASK_PLANNING = 'task_planning'
    EXPLORING = 'exploring'
    TARGETED_ANALYSIS = 'targeted_analysis'
    PLANNING = 'planning'
    IMPLEMENTING = 'implementing'
    FIX_REQUIRED = 'fix_required'
    READY_TO_VERIFY = 'ready_to_verify'
    VERIFYING = 'verifying'
    RECOVERING = 'recovering'
    DONE = 'done'
    BLOCKED = 'blocked'


@dataclass(slots=True)
class AgentController:
    '''Map structured tool failures onto bounded runtime state transitions.'''

    state: AgentControlState = AgentControlState.INIT
    contract: TaskContract | None = None
    planning_recovery_calls: int = 0

    def begin_turn(self, contract: TaskContract) -> None:
        self.contract = contract
        self.state = initial_state(contract)
        self.planning_recovery_calls = 0

    @property
    def planning_recovery(self) -> bool:
        '''Compatibility view; TASK_PLANNING is the source of truth.'''
        return self.state is AgentControlState.TASK_PLANNING

    def initial_tool_surface(self) -> InitialToolSurface:
        if self.contract is None:
            return 'all'
        return self.contract.initial_tool_surface

    def enter_planning_recovery(self) -> None:
        self.state = AgentControlState.TASK_PLANNING
        self.planning_recovery_calls += 1

    def enter_implementing(self) -> None:
        self.state = AgentControlState.IMPLEMENTING

    def enter_targeted_analysis(self) -> None:
        self.state = AgentControlState.TARGETED_ANALYSIS

    def enter_fix_required(self) -> None:
        self.state = AgentControlState.FIX_REQUIRED

    def enter_ready_to_verify(self) -> None:
        self.state = AgentControlState.READY_TO_VERIFY

    def observe_tool_result(self, tool_name: str, result: ToolResult) -> None:
        if tool_name == 'todo_write' and result.success:
            self.state = AgentControlState.IMPLEMENTING
        elif tool_name == 'verify' and result.success:
            self.state = AgentControlState.VERIFYING
        elif tool_name == 'verify' and not result.success:
            self.enter_fix_required()


def initial_state(contract: TaskContract) -> AgentControlState:
    if contract.initial_phase == 'planning':
        return AgentControlState.PLANNING
    if contract.initial_phase == 'implementing':
        return AgentControlState.IMPLEMENTING
    return AgentControlState.EXPLORING


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
