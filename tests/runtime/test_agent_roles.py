'''Focused tests for Agent Loop role boundaries.'''

import asyncio
from collections.abc import AsyncIterator

from forge.runtime.agent_state import AgentPhase, AgentRunState
from forge.runtime.model_runner import ModelRunner
from forge.runtime.model_failure import (
    ModelFailureAction,
    ModelFailureHandler,
)
from forge.runtime.model_client import (
    ModelCallError,
    ModelOutputTruncatedError,
    ModelProtocolError,
)
from forge.runtime.recovery_manager import RecoveryManager
from forge.runtime.request_builder import RequestBuilder, RequestState
from forge.runtime.state import (
    ModelStreamEvent,
    ModelTextDelta,
    ModelUsageUpdate,
    TokenUsage,
    ToolCall,
)
from forge.runtime.tool_runner import ToolRunPolicy, ToolRunner


class FakeClient:
    async def stream(self, **_: object) -> AsyncIterator[ModelStreamEvent]:
        yield ModelTextDelta(text='done')
        yield ModelUsageUpdate(usage=TokenUsage(3, 2))


class NeverExecute:
    def effect(self, _: str) -> None:
        return None

    async def execute(self, _: ToolCall) -> None:
        raise AssertionError('guarded calls must not reach the executor')


class EffectByName:
    def __init__(self, workspace_writes: set[str]) -> None:
        self.workspace_writes = workspace_writes

    def effect(self, name: str) -> str:
        return 'workspace_write' if name in self.workspace_writes else 'read'


def test_agent_run_state_records_only_phase_changes() -> None:
    state = AgentRunState()
    first = state.transition(
        AgentPhase.THINKING,
        reason='request',
        iteration=1,
    )
    duplicate = state.transition(
        AgentPhase.THINKING,
        reason='same phase',
        iteration=2,
    )

    assert first is not None
    assert duplicate is None
    assert state.iteration == 2
    assert len(state.transitions) == 1


def test_model_run_collects_response_and_cumulative_usage() -> None:
    run = ModelRunner(FakeClient()).run(
        messages=[],
        tools=None,
        system='test',
        completed_usage=TokenUsage(5, 1),
        iteration=2,
    )

    events = asyncio.run(collect(run))

    assert run.text == 'done'
    assert run.request_usage == TokenUsage(3, 2)
    usage = next(event for event in events if isinstance(event, ModelUsageUpdate))
    assert usage.usage == TokenUsage(8, 3)
    assert usage.model_calls == 2


def test_tool_runner_blocks_repeat_before_executor() -> None:
    runner = ToolRunner(NeverExecute())  # type: ignore[arg-type]
    call = ToolCall(index=0, id='tool-1', name='read_file', arguments={})

    result = asyncio.run(
        runner.execute_checked(
            call,
            ToolRunPolicy(
                tool_count=1,
                available_tools=frozenset({'read_file'}),
                previous_count=1,
                previous_success=False,
            ),
        )
    )

    assert not result.executed
    assert result.result.error is not None
    assert result.result.error.code == 'repeated_tool_call'


def test_request_builder_only_allows_finish_during_final_recovery() -> None:
    tools = [{'name': 'read_file'}, {'name': 'write_file'}, {'name': 'finish_task'}]
    recovery = RecoveryManager(
        tools,
        None,
        read_tools=frozenset({'read_file'}),
        excluded_write_tools=frozenset(),
    )
    builder = RequestBuilder(recovery, action_recovery_limit=3)

    spec = builder.build(
        state=RequestState(finalization_recovery=True),
        interaction_mode='code',
        all_tools=tools,
        plan_tools=[tools[0]],
        base_system_prompt='base',
        repository_context='repo',
        changed_paths=('forge/runtime/example.py',),
    )

    assert spec.tools == [{'name': 'finish_task'}]
    assert '[ForgeCode Finalization Recovery]' in spec.system_prompt
    assert spec.tool_names == frozenset({'finish_task'})


def test_request_builder_uses_plan_tool_surface() -> None:
    tools = [{'name': 'read_file'}, {'name': 'write_file'}]
    plan_tools = [tools[0]]
    recovery = RecoveryManager(
        tools,
        None,
        read_tools=frozenset({'read_file'}),
        excluded_write_tools=frozenset(),
    )

    spec = RequestBuilder(recovery, action_recovery_limit=3).build(
        state=RequestState(),
        interaction_mode='plan',
        all_tools=tools,
        plan_tools=plan_tools,
        base_system_prompt='base',
        repository_context='',
        changed_paths=(),
    )

    assert spec.tools == plan_tools
    assert spec.tool_names == frozenset({'read_file'})


def test_action_recovery_excludes_directory_only_writes() -> None:
    tools = [
        {'name': 'read_file'},
        {'name': 'create_directory'},
        {'name': 'write_file'},
    ]
    recovery = RecoveryManager(
        tools,
        EffectByName({'create_directory', 'write_file'}),
        read_tools=frozenset({'read_file'}),
        excluded_write_tools=frozenset({'create_directory'}),
    )

    selected = recovery.action_tools(read_available=False)

    assert selected == [{'name': 'write_file'}]


def test_model_failure_handler_preserves_partial_output() -> None:
    decision = ModelFailureHandler().classify(
        ModelOutputTruncatedError(),
        iteration=2,
        partial_text='partial answer',
        has_tool_calls=False,
        request_usage=TokenUsage(4, 3),
        output_continuations=0,
        max_output_continuations=2,
        reactive_compaction_attempted=False,
        protocol_recoveries=0,
        max_protocol_recoveries=2,
        available_tools=('read_file',),
    )

    assert decision.action is ModelFailureAction.CONTINUE_OUTPUT
    assert decision.consume_usage
    assert decision.preserve_partial_text
    assert decision.event.retryable
    assert len(decision.feedback) == 1


def test_model_failure_handler_requests_one_context_compaction() -> None:
    error = ModelCallError(
        'context_overflow',
        'too much context',
        retryable=False,
    )
    decision = ModelFailureHandler().classify(
        error,
        iteration=1,
        partial_text='',
        has_tool_calls=False,
        request_usage=None,
        output_continuations=0,
        max_output_continuations=2,
        reactive_compaction_attempted=False,
        protocol_recoveries=0,
        max_protocol_recoveries=2,
        available_tools=(),
    )

    assert decision.action is ModelFailureAction.COMPACT_CONTEXT


def test_model_failure_handler_stops_after_protocol_budget() -> None:
    decision = ModelFailureHandler().classify(
        ModelProtocolError('invalid'),
        iteration=3,
        partial_text='',
        has_tool_calls=False,
        request_usage=None,
        output_continuations=0,
        max_output_continuations=2,
        reactive_compaction_attempted=False,
        protocol_recoveries=2,
        max_protocol_recoveries=2,
        available_tools=(),
    )

    assert decision.action is ModelFailureAction.RAISE
    assert not decision.event.retryable


async def collect(run: object) -> list[ModelStreamEvent]:
    return [event async for event in run]  # type: ignore[attr-defined]
