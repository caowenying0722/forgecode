'''Tool execution role used by the Agent Loop orchestration layer.'''

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from forge.runtime.state import ToolCall
from forge.runtime.tool_executor import ToolExecutionRecord, ToolExecutor
from forge.tools.base import ToolResult


@dataclass(frozen=True, slots=True)
class ToolRunPolicy:
    tool_count: int
    available_tools: frozenset[str]
    action_recovery: bool = False
    mutation_recovery: bool = False
    planning_recovery: bool = False
    verification_recovery: bool = False
    action_read_exhausted: bool = False
    verification_read_exhausted: bool = False
    semantic_repeat: ToolResult | None = None
    previous_count: int = 0
    previous_success: bool = True
    repeated_limit: int = 2


@dataclass(frozen=True, slots=True)
class ToolRunResult:
    result: ToolResult
    executed: bool
    transaction: ToolTransaction | None = None


TransactionDecision = Literal[
    'executed',
    'cache_hit',
    'blocked',
]


@dataclass(frozen=True, slots=True)
class ToolTransaction:
    '''One controlled tool transaction decision and its runtime evidence.'''

    tool: str
    signature: str
    revision: int
    phase: str
    decision: TransactionDecision
    executed: bool
    result_code: str = ''
    permission_mode: str = ''
    cache_hit: bool = False
    available_tools: tuple[str, ...] = ()


@dataclass(slots=True)
class ToolBatchState:
    '''All mutable observations produced by one model tool-call batch.'''

    results: list[tuple[ToolCall, ToolResult]] = field(default_factory=list)
    workspace_writes: list[
        tuple[int, ToolCall, ToolResult, bool]
    ] = field(default_factory=list)
    last_workspace_change_position: int = -1
    task_progressed: bool = False
    evidence_progressed: bool = False
    verification_progressed: bool = False
    required_change_rejected: bool = False
    accepted_finish: ToolResult | None = None
    terminal_finish_reasons: tuple[str, ...] = ()

    @property
    def workspace_progressed(self) -> bool:
        return self.last_workspace_change_position >= 0

    def pending_write_results(
        self,
        *,
        reverted_to_baseline: bool,
    ) -> list[tuple[ToolCall, ToolResult]]:
        pending = [
            (call, result)
            for position, call, result, changed in self.workspace_writes
            if (
                position > self.last_workspace_change_position
                and not changed
                and not is_tool_protocol_failure(result)
                and not is_permission_denial(result)
            )
        ]
        if reverted_to_baseline and self.workspace_writes:
            _, call, result, _ = self.workspace_writes[-1]
            return [(call, result)]
        return pending


def is_permission_denial(result: ToolResult) -> bool:
    '''Do not treat a deliberate approval denial as a failed edit attempt.'''
    return bool(
        result.error is not None
        and result.error.code == 'permission_denied'
    )


class ToolRunner:
    '''Run validated calls through the shared policy and logging boundary.'''

    def __init__(self, executor: ToolExecutor) -> None:
        self.executor = executor

    def effect(self, name: str):
        return self.executor.effect(name)

    async def execute(self, tool_call: ToolCall) -> ToolExecutionRecord:
        return await self.executor.execute(tool_call)

    async def execute_checked(
        self,
        tool_call: ToolCall,
        policy: ToolRunPolicy,
    ) -> ToolRunResult:
        '''Apply phase/repetition guards, then cross the execution boundary.'''
        if tool_call.name == 'finish_task' and policy.tool_count != 1:
            return synthetic_failure(
                'finish_must_be_alone',
                'finish_task must be the only tool call in its model response. '
                'Complete other actions first, then declare the outcome in a '
                'separate response.',
            )
        if (
            policy.planning_recovery
            and tool_call.name not in policy.available_tools
        ):
            return unavailable_in_phase(tool_call, policy, 'Planning Recovery')
        if (
            policy.action_recovery
            and tool_call.name not in policy.available_tools
        ):
            return unavailable_in_phase(tool_call, policy, 'Action Recovery')
        if (
            policy.mutation_recovery
            and tool_call.name not in policy.available_tools
        ):
            return unavailable_in_phase(tool_call, policy, 'Edit Recovery')
        if (
            policy.verification_recovery
            and tool_call.name not in policy.available_tools
        ):
            return unavailable_in_phase(
                tool_call,
                policy,
                'Verification Recovery',
            )
        if policy.action_read_exhausted:
            return synthetic_failure(
                'action_read_limit_reached',
                'Action Recovery permits only one targeted repository read '
                'or search. Use the existing evidence and make the workspace '
                'edit now.',
            )
        if policy.verification_read_exhausted:
            return synthetic_failure(
                'verification_read_limit_reached',
                'Verification Recovery permits only one targeted repository '
                'read or search after a failed verify result. Use the latest '
                'verification output and existing evidence to repair the '
                'workspace, run the concrete repair command, or call verify.',
            )
        if policy.semantic_repeat is not None:
            return ToolRunResult(policy.semantic_repeat, executed=False)
        should_block_repeat = (
            tool_call.name != 'finish_task'
            and (
                policy.previous_count >= policy.repeated_limit
                or (
                    policy.previous_count >= 1
                    and not policy.previous_success
                )
            )
        )
        if should_block_repeat:
            cause = (
                'the previous identical call failed'
                if not policy.previous_success
                else f'it already ran {policy.previous_count} times'
            )
            return synthetic_failure(
                'repeated_tool_call',
                f'Skipped repeated {tool_call.name} call because {cause}. '
                'Use the existing result, change the arguments, or choose a '
                'different next action.',
                details={
                    'tool': tool_call.name,
                    'arguments': tool_call.arguments,
                    'previous_count': policy.previous_count,
                    'previous_success': policy.previous_success,
                },
            )
        execution = await self.execute(tool_call)
        return ToolRunResult(execution.result, executed=True)

    async def transact(
        self,
        tool_call: ToolCall,
        policy: ToolRunPolicy,
        *,
        revision: int,
        signature: str,
    ) -> ToolRunResult:
        '''Run one model tool request as a stateful controlled transaction.'''
        guarded = await self.execute_checked(tool_call, policy)
        transaction = ToolTransaction(
            tool=tool_call.name,
            signature=signature,
            revision=revision,
            phase=transaction_phase(policy),
            decision=transaction_decision(guarded),
            executed=guarded.executed,
            result_code=result_code(guarded.result),
            permission_mode=(
                self.executor.permission.mode if guarded.executed else ''
            ),
            cache_hit=bool(guarded.result.metadata.get('cache_hit')),
            available_tools=tuple(sorted(policy.available_tools)),
        )
        metadata = {
            **guarded.result.metadata,
            'tool_transaction': True,
            'transaction_phase': transaction.phase,
            'transaction_decision': transaction.decision,
            'transaction_signature': transaction.signature,
            'transaction_revision': transaction.revision,
            'transaction_executed': transaction.executed,
        }
        result = ToolResult(
            success=guarded.result.success,
            summary=guarded.result.summary,
            content=guarded.result.content,
            error=guarded.result.error,
            metadata=metadata,
        )
        return ToolRunResult(
            result=result,
            executed=guarded.executed,
            transaction=transaction,
        )


def synthetic_failure(
    code: str,
    message: str,
    *,
    details: dict[str, object] | None = None,
) -> ToolRunResult:
    return ToolRunResult(
        ToolResult.fail(code, message, details=details or {}),
        executed=False,
    )


def unavailable_in_phase(
    tool_call: ToolCall,
    policy: ToolRunPolicy,
    phase: str,
) -> ToolRunResult:
    return synthetic_failure(
        'tool_not_available_in_phase',
        f'{tool_call.name} is not available during {phase}. Use one of the '
        'tools included with this request.',
        details={'available_tools': sorted(policy.available_tools)},
    )


def transaction_phase(policy: ToolRunPolicy) -> str:
    if policy.planning_recovery:
        return 'planning_recovery'
    if policy.mutation_recovery:
        return 'edit_recovery'
    if policy.verification_recovery:
        return 'verification_recovery'
    if policy.action_recovery:
        return 'action_recovery'
    return 'normal'


def transaction_decision(run: ToolRunResult) -> TransactionDecision:
    if run.executed:
        return 'executed'
    if run.result.metadata.get('cache_hit'):
        return 'cache_hit'
    return 'blocked'


def result_code(result: ToolResult) -> str:
    if result.error is not None:
        return result.error.code
    if result.success:
        return 'ok'
    return 'unknown'


def is_tool_protocol_failure(result: ToolResult) -> bool:
    return (
        not result.success
        and result.error is not None
        and result.error.code in {
            'invalid_arguments',
            'unknown_tool',
            'finish_must_be_alone',
            'unsupported_shell_syntax',
            'invalid_pattern',
            'patch_contains_read_line_numbers',
            'git_diff_path_is_directory',
            'tool_not_available_in_phase',
            'action_read_limit_reached',
            'verification_read_limit_reached',
            'todo_required',
        }
    )


def is_satisfied_non_diff_workspace_write(
    tool_call: ToolCall,
    result: ToolResult,
) -> bool:
    '''Treat successful idempotent directory setup as a satisfied write.'''
    return tool_call.name == 'create_directory' and result.success
