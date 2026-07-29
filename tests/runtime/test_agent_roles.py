'''Focused tests for Agent Loop role boundaries.'''

import asyncio
from collections.abc import AsyncIterator

from forge.runtime.agent_loop import tool_call_signature
from forge.runtime.intent import infer_task_contract
from forge.runtime.agent_controller import AgentControlState, AgentController
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
from forge.runtime.progress import evaluate_progress
from forge.runtime.state import (
    ModelStreamEvent,
    ModelTextDelta,
    ModelUsageUpdate,
    TokenUsage,
    ToolCall,
)
from forge.runtime.tool_runner import ToolRunPolicy, ToolRunner
from forge.tools.base import ToolResult
from forge.hooks.builtin import should_require_todo_plan


class FakePermission:
    mode = 'trusted'


class FakeClient:
    async def stream(self, **_: object) -> AsyncIterator[ModelStreamEvent]:
        yield ModelTextDelta(text='done')
        yield ModelUsageUpdate(usage=TokenUsage(3, 2))


class NeverExecute:
    def effect(self, _: str) -> None:
        return None

    async def execute(self, _: ToolCall) -> None:
        raise AssertionError('guarded calls must not reach the executor')


class TransactionExecutor:
    permission = FakePermission()

    def effect(self, _: str) -> None:
        return None

    async def execute(self, _: ToolCall):
        from forge.runtime.tool_executor import ToolExecutionRecord

        return ToolExecutionRecord(
            result=ToolResult.ok('Read sample.', content='sample'),
            effect='read_only',
            duration_seconds=0.01,
            permission_mode='trusted',
        )


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


def test_tool_runner_wraps_execution_as_transaction() -> None:
    runner = ToolRunner(TransactionExecutor())  # type: ignore[arg-type]
    call = ToolCall(index=0, id='tool-1', name='read_file', arguments={})

    result = asyncio.run(
        runner.transact(
            call,
            ToolRunPolicy(
                tool_count=1,
                available_tools=frozenset({'read_file'}),
            ),
            revision=3,
            signature='3:read_file:abc123',
        )
    )

    assert result.executed is True
    assert result.transaction is not None
    assert result.transaction.decision == 'executed'
    assert result.transaction.phase == 'normal'
    assert result.transaction.permission_mode == 'trusted'
    assert result.result.metadata['tool_transaction'] is True
    assert result.result.metadata['transaction_revision'] == 3


def test_tool_runner_wraps_cache_hit_as_transaction() -> None:
    runner = ToolRunner(NeverExecute())  # type: ignore[arg-type]
    call = ToolCall(index=0, id='tool-1', name='read_file', arguments={})

    result = asyncio.run(
        runner.transact(
            call,
            ToolRunPolicy(
                tool_count=1,
                available_tools=frozenset({'read_file'}),
                semantic_repeat=ToolResult.ok(
                    'Cache hit.',
                    metadata={'cache_hit': True},
                ),
            ),
            revision=3,
            signature='3:read_file:abc123',
        )
    )

    assert result.executed is False
    assert result.transaction is not None
    assert result.transaction.decision == 'cache_hit'
    assert result.result.metadata['transaction_decision'] == 'cache_hit'


def test_tool_signature_normalizes_defaults_and_paths() -> None:
    explicit = ToolCall(
        index=0,
        id='explicit',
        name='list_directory',
        arguments={'path': '.\\src\\', 'max_results': 1000},
    )
    implicit = ToolCall(
        index=0,
        id='implicit',
        name='list_directory',
        arguments={'path': './src'},
    )

    assert tool_call_signature(explicit, 4) == tool_call_signature(implicit, 4)


def test_grep_signature_normalizes_file_types() -> None:
    first = ToolCall(
        index=0,
        id='first',
        name='grep',
        arguments={'pattern': 'Player', 'file_types': ['TS', '.tsx']},
    )
    second = ToolCall(
        index=0,
        id='second',
        name='grep',
        arguments={
            'pattern': 'Player',
            'path': '.',
            'regex': True,
            'case_sensitive': True,
            'max_results': 200,
            'file_types': ['.tsx', '.ts'],
        },
    )

    assert tool_call_signature(first, 4) == tool_call_signature(second, 4)


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


def test_request_builder_uses_contract_read_only_surface() -> None:
    tools = [{'name': 'read_file'}, {'name': 'write_file'}]
    plan_tools = [tools[0]]
    recovery = RecoveryManager(
        tools,
        None,
        read_tools=frozenset({'read_file'}),
        excluded_write_tools=frozenset(),
    )
    contract = infer_task_contract('给出一个修复方案')

    spec = RequestBuilder(recovery, action_recovery_limit=3).build(
        state=RequestState(task_contract=contract),
        interaction_mode='auto',
        all_tools=tools,
        plan_tools=plan_tools,
        base_system_prompt='base',
        repository_context='',
        changed_paths=(),
    )

    assert spec.tools == plan_tools
    assert '[ForgeCode Turn Task Contract]' in spec.system_prompt


def test_request_builder_uses_no_tool_surface_for_knowledge_answer() -> None:
    tools = [{'name': 'read_file'}, {'name': 'write_file'}]
    recovery = RecoveryManager(
        tools,
        None,
        read_tools=frozenset({'read_file'}),
        excluded_write_tools=frozenset(),
    )
    contract = infer_task_contract('解释一下 Python 里的生成器是什么')

    spec = RequestBuilder(recovery, action_recovery_limit=3).build(
        state=RequestState(
            control_state=AgentControlState.EXPLORING,
            task_contract=contract,
        ),
        interaction_mode='auto',
        all_tools=tools,
        plan_tools=[tools[0]],
        base_system_prompt='base',
        repository_context='',
        changed_paths=(),
    )

    assert spec.tools is None
    assert spec.tool_names == frozenset()


def test_request_builder_allows_only_read_tools_for_file_analysis() -> None:
    tools = [{'name': 'read_file'}, {'name': 'write_file'}]
    plan_tools = [tools[0]]
    recovery = RecoveryManager(
        tools,
        None,
        read_tools=frozenset({'read_file'}),
        excluded_write_tools=frozenset(),
    )
    contract = infer_task_contract('分析 forge/runtime/intent.py 的职责')

    spec = RequestBuilder(recovery, action_recovery_limit=3).build(
        state=RequestState(
            control_state=AgentControlState.EXPLORING,
            task_contract=contract,
            change_required=contract.requires_change,
        ),
        interaction_mode='auto',
        all_tools=tools,
        plan_tools=plan_tools,
        base_system_prompt='base',
        repository_context='',
        changed_paths=(),
    )

    assert spec.tools == plan_tools
    assert contract.requires_change is False
    assert '[ForgeCode Turn Change Contract]' not in spec.system_prompt


def test_request_builder_control_state_overrides_conflicting_booleans() -> None:
    tools = [
        {'name': 'read_file'},
        {'name': 'write_file'},
        {'name': 'finish_task'},
    ]
    recovery = RecoveryManager(
        tools,
        EffectByName({'write_file'}),
        read_tools=frozenset({'read_file'}),
        excluded_write_tools=frozenset(),
    )
    contract = infer_task_contract('分析 forge/runtime/intent.py')

    spec = RequestBuilder(recovery, action_recovery_limit=3).build(
        state=RequestState(
            control_state=AgentControlState.TARGETED_ANALYSIS,
            task_contract=contract,
            planning_recovery=True,
            action_recovery=False,
            action_read_used=True,
        ),
        interaction_mode='auto',
        all_tools=tools,
        plan_tools=[tools[0]],
        base_system_prompt='base',
        repository_context='',
        changed_paths=(),
    )

    assert spec.tool_names == frozenset({'write_file', 'finish_task'})
    assert '[ForgeCode Action Recovery]' in spec.system_prompt
    assert '[ForgeCode Planning Recovery]' not in spec.system_prompt


def test_request_builder_injects_runtime_task_model_for_change() -> None:
    tools = [{'name': 'read_file'}, {'name': 'write_file'}]
    recovery = RecoveryManager(
        tools,
        None,
        read_tools=frozenset({'read_file'}),
        excluded_write_tools=frozenset(),
    )
    contract = infer_task_contract('帮我修复 src/player.ts')

    spec = RequestBuilder(recovery, action_recovery_limit=3).build(
        state=RequestState(
            task_contract=contract,
            task_goal='帮我修复 src/player.ts',
            change_required=True,
            task_scope_patterns=('src/player.ts',),
        ),
        interaction_mode='auto',
        all_tools=tools,
        plan_tools=[tools[0]],
        base_system_prompt='base',
        repository_context='',
        changed_paths=(),
    )

    assert '[ForgeCode Runtime Task Model]' in spec.system_prompt
    assert 'Completion conditions:' in spec.system_prompt
    assert 'Expected impact scope:' in spec.system_prompt
    assert 'src/player.ts' in spec.system_prompt


def test_single_file_fix_does_not_force_todo_plan() -> None:
    assert should_require_todo_plan('请修复 forge/runtime/intent.py') is False


def test_multi_module_architecture_forces_todo_plan() -> None:
    assert should_require_todo_plan('请重构多个模块的架构并更新相关代码') is True


def test_planning_completion_enters_implementing() -> None:
    controller = AgentController()
    controller.begin_turn(
        infer_task_contract('请重构多个模块的架构并更新相关代码')
    )
    controller.enter_planning_recovery()

    controller.observe_tool_result('todo_write', ToolResult.ok('planned'))

    assert controller.state is AgentControlState.IMPLEMENTING
    assert controller.planning_recovery is False


def test_failed_verification_stays_out_of_exploring() -> None:
    controller = AgentController()
    controller.begin_turn(infer_task_contract('请修复 forge/runtime/intent.py'))

    controller.observe_tool_result(
        'verify',
        ToolResult.fail('verification_failed', 'tests failed'),
    )

    assert controller.state is AgentControlState.FIX_REQUIRED
    assert controller.state is not AgentControlState.EXPLORING


def test_progress_evaluator_counts_verification_as_progress() -> None:
    progress = evaluate_progress(
        workspace_progressed=False,
        task_progressed=False,
        evidence_progressed=False,
        verification_progressed=True,
        review_progressed=False,
        protocol_failure=False,
        mutation_recovery_active=False,
    )

    assert progress.progressed is True
    assert progress.signal == 'verification_evidence'


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
