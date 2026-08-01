'''Focused tests for Agent Loop role boundaries.'''

import asyncio
from collections.abc import AsyncIterator

import pytest

from forge.runtime.agent_loop import (
    early_mutation_relevance_failure,
    tool_call_signature,
)
from forge.runtime.intent import infer_task_contract
from forge.runtime.agent_controller import (
    AgentControlState,
    AgentController,
    SynthesisMode,
    TurnRuntimeState,
)
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
from forge.runtime.recovery_manager import RecoveryScope
from forge.runtime.request_builder import (
    RequestBuilder,
    RequestState,
    _latest_verification,
)
from forge.runtime.progress import evaluate_progress
from forge.runtime.state import (
    ModelStreamEvent,
    ModelTextDelta,
    ModelUsageUpdate,
    TokenUsage,
    ToolCall,
    VerificationEvidence,
)
from forge.runtime.tool_runner import (
    ToolRunPolicy,
    ToolRunner,
    transaction_phase,
)
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


def test_early_mutation_relevance_guard_blocks_off_scope_target() -> None:
    result = early_mutation_relevance_failure(
        ToolCall(
            0,
            'write',
            'write_file',
            {'path': 'notes/unrelated.txt', 'content': 'x'},
        ),
        tool_effect='workspace_write',
        change_required=True,
        task_scope_patterns=('src/app.py',),
    )

    assert result is not None
    assert result.error is not None
    assert result.error.code == 'irrelevant_mutation_target'


def test_early_mutation_relevance_guard_allows_in_scope_target() -> None:
    result = early_mutation_relevance_failure(
        ToolCall(
            0,
            'write',
            'write_file',
            {'path': 'src/app.py', 'content': 'x'},
        ),
        tool_effect='workspace_write',
        change_required=True,
        task_scope_patterns=('src/app.py',),
    )

    assert result is None


def test_early_mutation_relevance_guard_allows_scoped_directory_target() -> None:
    result = early_mutation_relevance_failure(
        ToolCall(
            0,
            'mkdir',
            'create_directory',
            {'path': 'game'},
        ),
        tool_effect='workspace_write',
        change_required=True,
        task_scope_patterns=('game/**',),
    )

    assert result is None


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
    contract = infer_task_contract('请修复 sample.txt')
    runtime = TurnRuntimeState(
        control_state=AgentControlState.IMPLEMENTING,
        contract=contract,
    )
    runtime.synthesis.mode = SynthesisMode.FINALIZATION

    spec = builder.build(
        state=RequestState(runtime=runtime),
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


def test_request_builder_uses_runtime_action_recovery_scope() -> None:
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
    runtime = TurnRuntimeState(
        control_state=AgentControlState.TARGETED_ANALYSIS,
        contract=contract,
    )
    runtime.action_recovery_state.read_used = True

    spec = RequestBuilder(recovery, action_recovery_limit=3).build(
        state=RequestState(
            runtime=runtime,
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


def test_request_builder_runtime_snapshot_overrides_legacy_booleans() -> None:
    tools = [
        {'name': 'read_file'},
        {'name': 'todo_write'},
        {'name': 'write_file'},
        {'name': 'finish_task'},
    ]
    recovery = RecoveryManager(
        tools,
        EffectByName({'write_file'}),
        read_tools=frozenset({'read_file'}),
        excluded_write_tools=frozenset(),
    )
    contract = infer_task_contract('请重构多个模块的架构并更新相关代码')
    runtime = TurnRuntimeState(
        control_state=AgentControlState.PLANNING,
        contract=contract,
    )

    spec = RequestBuilder(recovery, action_recovery_limit=3).build(
        state=RequestState(
            runtime=runtime,
        ),
        interaction_mode='auto',
        all_tools=tools,
        plan_tools=[tools[0], tools[1]],
        base_system_prompt='base',
        repository_context='',
        changed_paths=(),
    )

    assert spec.tool_names == frozenset({'read_file', 'todo_write'})
    assert '[ForgeCode Finalization Recovery]' not in spec.system_prompt
    assert '[ForgeCode Verification Recovery]' not in spec.system_prompt


def test_tool_run_policy_uses_runtime_snapshot_for_transaction_phase() -> None:
    contract = infer_task_contract('请修复 forge/runtime/intent.py')
    runtime = TurnRuntimeState(
        control_state=AgentControlState.IMPLEMENTING,
        contract=contract,
    )
    runtime.edit_recovery.failures.append({'code': 'edit_failed'})

    policy = ToolRunPolicy(
        tool_count=1,
        available_tools=frozenset({'read_file'}),
        runtime=runtime,
        control_state=AgentControlState.TASK_PLANNING,
    )

    assert transaction_phase(policy) == 'edit_recovery'


def test_request_builder_reads_latest_verification_from_ledger() -> None:
    contract = infer_task_contract('请修复 forge/runtime/intent.py')
    runtime = TurnRuntimeState(
        control_state=AgentControlState.FIX_REQUIRED,
        contract=contract,
    )
    runtime.verification.latest = VerificationEvidence(
        command='uv run pytest -q',
        cwd='.',
        exit_code=2,
        duration_seconds=0.1,
        timed_out=False,
        workspace_revision=1,
        source_revision=1,
        filesystem_revision=1,
        status='failed',
        failure_signature='old-failure',
        verification_type='test',
    )
    runtime.verification_ledger.record_from_metadata(
        {
            'verification': True,
            'verification_status': 'passed',
            'verification_type': 'test',
            'command': 'uv run pytest -q',
            'cwd': '.',
            'workspace_revision': 1,
            'source_revision': 1,
            'filesystem_revision': 2,
            'exit_code': 0,
            'duration_seconds': 0.2,
            'timed_out': False,
        },
        content='passed',
        evidence_source='run_command',
    )

    latest = _latest_verification(RequestState(runtime=runtime))

    assert latest is not None
    assert latest.success is True
    assert latest.filesystem_revision == 2


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


def test_invalid_verify_does_not_enter_fix_required() -> None:
    controller = AgentController()
    controller.begin_turn(infer_task_contract('请修复 forge/runtime/intent.py'))

    controller.observe_tool_result(
        'verify',
        ToolResult.fail(
            'verification_command_invalid',
            'Verification command is invalid.',
            metadata={
                'verification': True,
                'verification_status': 'invalid',
                'exit_code': -1,
                'timed_out': False,
            },
        ),
    )

    assert controller.state is AgentControlState.READY_TO_VERIFY
    runtime = controller.snapshot()
    runtime.verification.latest = VerificationEvidence(
        command='npm install',
        cwd='.',
        exit_code=-1,
        duration_seconds=0,
        timed_out=False,
        workspace_revision=1,
        status='invalid',
    )
    assert runtime.verification_fix_required is False


def test_action_recovery_is_derived_from_control_state() -> None:
    controller = AgentController()
    controller.begin_turn(infer_task_contract('请修复 src/app.py'))

    assert controller.action_recovery is False
    controller.enter_targeted_analysis()

    assert controller.state is AgentControlState.TARGETED_ANALYSIS
    assert controller.action_recovery is True


def test_controller_uses_turn_kind_for_read_only_initial_states() -> None:
    answer = AgentController()
    answer.begin_turn(infer_task_contract('解释一下 Python 里的生成器是什么'))
    advisory = AgentController()
    advisory.begin_turn(
        infer_task_contract('还有加入其他功能让整个项目更完善吗？')
    )
    change = AgentController()
    change.begin_turn(infer_task_contract('直接实现经验等级和三选一升级。'))

    assert answer.state is AgentControlState.ANSWERING
    assert advisory.state is AgentControlState.ADVISING
    assert change.state is AgentControlState.IMPLEMENTING


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


@pytest.mark.parametrize(
    (
        'workspace_progressed',
        'evidence_progressed',
        'changed_paths',
        'evidence_paths',
        'repair_target_paths',
        'expected_signal',
    ),
    [
        (
            True,
            False,
            ('notes/unrelated.txt',),
            (),
            (),
            'unrelated_workspace_revision',
        ),
        (
            False,
            True,
            (),
            ('notes/unrelated.txt',),
            (),
            'unrelated_repository_evidence',
        ),
        (
            False,
            True,
            (),
            ('src/app.py',),
            ('src/app.py',),
            'repository_evidence',
        ),
    ],
)
def test_progress_evaluator_requires_task_or_repair_relevance(
    workspace_progressed: bool,
    evidence_progressed: bool,
    changed_paths: tuple[str, ...],
    evidence_paths: tuple[str, ...],
    repair_target_paths: tuple[str, ...],
    expected_signal: str,
) -> None:
    progress = evaluate_progress(
        workspace_progressed=workspace_progressed,
        task_progressed=False,
        evidence_progressed=evidence_progressed,
        verification_progressed=False,
        review_progressed=False,
        protocol_failure=False,
        mutation_recovery_active=False,
        requires_change=True,
        task_scope_patterns=('src/app.py',),
        changed_paths=changed_paths,
        evidence_paths=evidence_paths,
        repair_target_paths=repair_target_paths,
    )

    assert progress.signal == expected_signal
    assert progress.progressed is (not expected_signal.startswith('unrelated'))


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

    selected = recovery.action_tools(
        scope=RecoveryScope.action(read_used=True)
    )

    assert selected == [{'name': 'write_file'}]


def test_recovery_scope_controls_action_reads() -> None:
    tools = [
        {'name': 'read_file'},
        {'name': 'grep'},
        {'name': 'write_file'},
        {'name': 'finish_task'},
    ]
    recovery = RecoveryManager(
        tools,
        EffectByName({'write_file'}),
        read_tools=frozenset({'read_file', 'grep'}),
        excluded_write_tools=frozenset(),
    )

    first = recovery.action_tools(
        scope=RecoveryScope.action(read_used=False)
    )
    exhausted = recovery.action_tools(
        scope=RecoveryScope.action(read_used=True)
    )

    assert {tool['name'] for tool in first or ()} == {
        'read_file',
        'grep',
        'write_file',
        'finish_task',
    }
    assert {tool['name'] for tool in exhausted or ()} == {
        'write_file',
        'finish_task',
    }


def test_recovery_scope_controls_mutation_reads() -> None:
    tools = [
        {'name': 'read_file'},
        {'name': 'grep'},
        {'name': 'replace_text'},
    ]
    recovery = RecoveryManager(
        tools,
        EffectByName({'replace_text'}),
        read_tools=frozenset({'read_file', 'grep'}),
        excluded_write_tools=frozenset(),
    )
    failures = [{'tool': 'replace_text', 'code': 'patch_context_not_found'}]

    first = recovery.mutation_tools(
        failures,
        scope=RecoveryScope.mutation(read_used=False),
    )
    exhausted = recovery.mutation_tools(
        failures,
        scope=RecoveryScope.mutation(read_used=True),
    )

    assert {tool['name'] for tool in first or ()} == {
        'read_file',
        'grep',
        'replace_text',
    }
    assert {tool['name'] for tool in exhausted or ()} == {'replace_text'}


def test_recovery_scope_controls_verification_reads() -> None:
    tools = [
        {'name': 'find_files'},
        {'name': 'read_file'},
        {'name': 'grep'},
        {'name': 'write_file'},
        {'name': 'run_command'},
        {'name': 'task'},
        {'name': 'verify'},
        {'name': 'finish_task'},
    ]
    recovery = RecoveryManager(
        tools,
        EffectByName({'write_file'}),
        read_tools=frozenset({'read_file', 'grep'}),
        excluded_write_tools=frozenset(),
    )

    first = recovery.verification_tools(
        fix_available=True,
        scope=RecoveryScope.verification(read_count=1, read_budget=2),
        verify_available=False,
    )
    exhausted = recovery.verification_tools(
        fix_available=True,
        scope=RecoveryScope.verification(read_count=2, read_budget=2),
        verify_available=False,
    )

    assert {tool['name'] for tool in first or ()} == {
        'find_files',
        'read_file',
        'grep',
        'write_file',
        'run_command',
        'finish_task',
    }
    assert {tool['name'] for tool in exhausted or ()} == {
        'write_file',
        'run_command',
        'finish_task',
    }
    assert 'task' not in {tool['name'] for tool in first or ()}


def test_recovery_manager_extracts_verification_repair_target() -> None:
    recovery = RecoveryManager(
        [{'name': 'read_file'}],
        None,
        read_tools=frozenset({'read_file'}),
        excluded_write_tools=frozenset(),
    )
    result = ToolResult.fail(
        'verification_failed',
        'Verification exited with code 1.',
        content="src/app.py:12: NameError: name 'missing_value' is not defined",
        metadata={
            'verification_status': 'failed',
            'failure_signature': 'abc123',
        },
    )

    target = recovery.verification_repair_target_from_result(
        result,
        changed_paths=('src/app.py',),
    )

    assert target.source == 'verify:failed'
    assert target.paths == ('src/app.py',)
    assert target.line_numbers == (12,)
    assert target.symbols == ('missing_value',)
    assert target.failure_signature == 'abc123'
    assert target.expected_action == (
        'repair the failing changed code or project configuration'
    )


def test_invalid_verify_does_not_create_source_repair_target() -> None:
    recovery = RecoveryManager(
        [{'name': 'read_file'}],
        None,
        read_tools=frozenset({'read_file'}),
        excluded_write_tools=frozenset(),
    )
    result = ToolResult.fail(
        'verification_command_invalid',
        'Verification command is invalid.',
        metadata={
            'verification_status': 'invalid',
            'failure_signature': 'invalid-command',
        },
    )

    target = recovery.verification_repair_target_from_result(
        result,
        changed_paths=('src/app.py',),
    )

    assert target.source == 'verify:invalid'
    assert target.paths == ()
    assert target.direct_dependencies == ()
    assert target.expected_action == (
        'choose a valid non-interactive validation command'
    )


def test_artifact_recovery_does_not_require_source_edit() -> None:
    recovery = RecoveryManager(
        [{'name': 'verify'}, {'name': 'write_file'}],
        EffectByName({'write_file'}),
        read_tools=frozenset(),
        excluded_write_tools=frozenset(),
    )
    result = ToolResult.fail(
        'verification_side_effect',
        'Build output integrity changed.',
        metadata={
            'verification_status': 'failed',
            'generated_artifact_paths': ['dist/index.html'],
            'artifact_deltas': [{'path': 'dist/index.html'}],
        },
    )

    target = recovery.verification_repair_target_from_result(
        result,
        changed_paths=('src/app.ts',),
    )
    tools = recovery.verification_tools(
        fix_available=True,
        scope=RecoveryScope.verification(read_count=0, read_budget=1),
        recovery_kind=target.recovery_kind,
    )

    assert target.recovery_kind == 'artifact_recovery'
    assert target.requires_source_edit is False
    assert 'write_file' not in {tool['name'] for tool in tools or ()}


def test_verification_command_recovery_does_not_require_source_edit() -> None:
    recovery = RecoveryManager(
        [{'name': 'verify'}, {'name': 'write_file'}],
        EffectByName({'write_file'}),
        read_tools=frozenset(),
        excluded_write_tools=frozenset(),
    )
    result = ToolResult.fail(
        'verification_command_invalid',
        'Invalid verification command.',
        metadata={'verification_status': 'invalid'},
    )

    target = recovery.verification_repair_target_from_result(
        result,
        changed_paths=('src/app.ts',),
    )

    assert target.recovery_kind == 'verification_command_repair'
    assert target.requires_source_edit is False


def test_source_compile_failure_still_requires_source_repair() -> None:
    recovery = RecoveryManager(
        [{'name': 'read_file'}],
        None,
        read_tools=frozenset({'read_file'}),
        excluded_write_tools=frozenset(),
    )
    result = ToolResult.fail(
        'verification_failed',
        'TypeScript compilation failed.',
        content="src/app.ts:12: error TS2304: Cannot find name 'missingValue'.",
        metadata={'verification_status': 'failed', 'source_revision': 1},
    )

    target = recovery.verification_repair_target_from_result(
        result,
        changed_paths=('src/app.ts',),
    )

    assert target.recovery_kind == 'source_repair'
    assert target.requires_source_edit is True


def test_model_cannot_escape_artifact_recovery_with_unrelated_source_edit() -> None:
    recovery = RecoveryManager(
        [],
        None,
        read_tools=frozenset(),
        excluded_write_tools=frozenset(),
    )
    result = ToolResult.fail(
        'verification_side_effect',
        'Build artifact recovery is required.',
        metadata={
            'verification_status': 'failed',
            'source_revision': 1,
            'generated_artifact_paths': ['dist/index.html'],
        },
    )
    target = recovery.verification_repair_target_from_result(
        result,
        changed_paths=('src/app.ts',),
    )

    progressed = recovery.repair_progressed(
        target,
        source_revision=2,
        changed_paths=('src/unrelated.ts',),
    )

    assert progressed is False


def test_summary_recovery_does_not_open_source_write_tools() -> None:
    recovery = RecoveryManager(
        [{'name': 'finish_task'}, {'name': 'write_file'}, {'name': 'verify'}],
        EffectByName({'write_file'}),
        read_tools=frozenset(),
        excluded_write_tools=frozenset(),
    )

    tools = recovery.finalization_tools()

    assert {tool['name'] for tool in tools or ()} == {'finish_task'}


def test_recovery_manager_extracts_ts2305_repair_target() -> None:
    recovery = RecoveryManager(
        [{'name': 'read_file'}],
        None,
        read_tools=frozenset({'read_file'}),
        excluded_write_tools=frozenset(),
    )
    result = ToolResult.fail(
        'verification_failed',
        'Verification exited with code 2.',
        content=(
            "src/app.ts:1:10 - error TS2305: Module './lib' has no "
            "exported member 'Foo'."
        ),
        metadata={
            'verification_status': 'failed',
            'failure_signature': 'ts2305',
        },
    )

    target = recovery.verification_repair_target_from_result(
        result,
        changed_paths=('src/app.ts',),
    )

    assert target.paths == ('src/app.ts',)
    assert target.line_numbers == (1,)
    assert target.symbols == ('Foo',)
    assert target.missing_exports == ('Foo',)
    assert target.modules == ('./lib',)
    assert 'src/lib.ts' in target.direct_dependencies
    assert recovery.verification_read_budget(target) > 1


def test_recovery_manager_extracts_mutation_repair_target() -> None:
    recovery = RecoveryManager(
        [{'name': 'apply_patch'}],
        None,
        read_tools=frozenset({'read_file'}),
        excluded_write_tools=frozenset(),
    )
    call = ToolCall(
        0,
        'patch',
        'apply_patch',
        {'patch': '*** Update File: src/app.py\n'},
    )
    result = ToolResult.fail(
        'patch_context_not_found',
        'Could not find context in src/app.py:7.',
        content='Closest current text in src/app.py:7: return old_value',
    )

    target = recovery.mutation_repair_target(call, result)

    assert target.source == 'apply_patch:patch_context_not_found'
    assert 'src/app.py' in target.paths
    assert target.line_numbers == (7,)
    assert target.expected_action == (
        'retry a smaller corrected patch against the same target'
    )


def test_request_builder_injects_verification_repair_target() -> None:
    tools = [{'name': 'read_file'}, {'name': 'write_file'}, {'name': 'verify'}]
    recovery = RecoveryManager(
        tools,
        EffectByName({'write_file'}),
        read_tools=frozenset({'read_file'}),
        excluded_write_tools=frozenset(),
    )
    failed = ToolResult.fail(
        'verification_failed',
        'Verification exited with code 1.',
        content="src/app.py:12: NameError: name 'missing_value' is not defined",
        metadata={
            'verification_status': 'failed',
            'failure_signature': 'abc123',
        },
    )
    target = recovery.verification_repair_target_from_result(
        failed,
        changed_paths=('src/app.py',),
    )
    evidence = VerificationEvidence(
        command='python -m pytest -q',
        cwd='.',
        exit_code=1,
        duration_seconds=0.1,
        timed_out=False,
        workspace_revision=3,
        status='failed',
        failure_signature='abc123',
    )
    contract = infer_task_contract('请修复 src/app.py')
    runtime = TurnRuntimeState(
        control_state=AgentControlState.FIX_REQUIRED,
        contract=contract,
    )
    runtime.verification.latest = evidence
    runtime.verification.repair_target = target

    spec = RequestBuilder(recovery, action_recovery_limit=3).build(
        state=RequestState(
            runtime=runtime,
        ),
        interaction_mode='auto',
        all_tools=tools,
        plan_tools=[tools[0]],
        base_system_prompt='base',
        repository_context='',
        changed_paths=('src/app.py',),
    )

    assert '[ForgeCode Verification Recovery]' in spec.system_prompt
    assert '[ForgeCode Repair Target]' in spec.system_prompt
    assert 'src/app.py' in spec.system_prompt
    assert 'missing_value' in spec.system_prompt
    assert 'verify' not in spec.tool_names


def test_invalid_verify_keeps_verify_available_without_repair_target() -> None:
    tools = [
        {'name': 'read_file'},
        {'name': 'write_file'},
        {'name': 'verify'},
        {'name': 'finish_task'},
        {'name': 'git_status'},
        {'name': 'git_diff'},
    ]
    recovery = RecoveryManager(
        tools,
        EffectByName({'write_file'}),
        read_tools=frozenset({'read_file'}),
        excluded_write_tools=frozenset(),
    )
    contract = infer_task_contract('请修复 src/app.py')
    runtime = TurnRuntimeState(
        control_state=AgentControlState.READY_TO_VERIFY,
        contract=contract,
    )
    runtime.verification.latest = VerificationEvidence(
        command='npm install',
        cwd='.',
        exit_code=-1,
        duration_seconds=0.0,
        timed_out=False,
        workspace_revision=3,
        status='invalid',
    )

    spec = RequestBuilder(recovery, action_recovery_limit=3).build(
        state=RequestState(runtime=runtime),
        interaction_mode='auto',
        all_tools=tools,
        plan_tools=[tools[0]],
        base_system_prompt='base',
        repository_context='',
        changed_paths=('src/app.py',),
    )

    assert {'verify', 'finish_task', 'git_status', 'git_diff'} <= spec.tool_names
    assert 'write_file' not in spec.tool_names
    assert '[ForgeCode Repair Target]' not in spec.system_prompt
    assert 'previous verify command was invalid' in spec.system_prompt
    assert 'Do not edit files' in spec.system_prompt


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
