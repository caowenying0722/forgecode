'''Multi-step model and tool execution for the M1 Agent Loop.'''

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from functools import cache
import hashlib
from itertools import count
import json
from pathlib import Path
from typing import Any, Literal

from forge.context.compactor import CompactionConfig
from forge.context.manager import (
    CompactionReport,
    ContextManager,
    ContextStats,
)
from forge.context.working import WorkingState
from forge.config import (
    ForgeConfig,
    SUPPORTED_MODEL_IDS,
    forge_home,
    normalize_supported_model_id,
    update_user_model_id,
)
from forge.hooks import TodoPlanningHook
from forge.hooks.registry import HookRegistry
from forge.hooks.state import HookContext
from forge.mcp.client import (
    MCPConfigurationError,
    forge_app_root,
    mcp_config_sources,
    parse_mcp_config,
)
from forge.runtime.intent import (
    ModelSemanticTaskClassifier,
    SemanticTaskClassifier,
    TaskContract,
    infer_task_contract,
    refine_task_contract_async,
)
from forge.runtime.agent_state import AgentPhase, AgentRunState
from forge.runtime.agent_messages import (
    append_notification_message,
    build_assistant_message,
    build_tool_result_message,
)
from forge.runtime.agent_controller import (
    AgentControlState,
    AgentController,
    SynthesisMode,
    batch_requires_todo,
    build_planning_recovery_feedback,
    is_todo_required_result,
)
from forge.runtime.completion_checker import (
    CompletionChecker,
    build_completion_feedback,
    build_finalization_recovery_feedback,
    completion_review_paths,
    only_verification_blocked,
    render_completion_ready_context,
    verification_from_result,
)
from forge.runtime.model_runner import ModelRunner, add_token_usage
from forge.runtime.model_failure import (
    ModelFailureAction,
    ModelFailureHandler,
)
from forge.runtime.protocol_recovery import (
    build_synthesis_retry_feedback,
    build_tool_protocol_feedback,
)
from forge.runtime.progress import evaluate_progress
from forge.runtime.request_builder import RequestBuilder, RequestState
from forge.runtime.recovery_manager import RecoveryManager, RepairTarget
from forge.runtime.recovery_feedback import (
    action_recovery_stuck_reason,
    build_action_recovery_feedback,
    build_mutation_recovery_feedback,
    build_stagnation_feedback,
    build_stagnation_final_recovery_feedback,
    build_token_limit_recovery_feedback,
    mutation_failure_record,
    mutation_recovery_stuck_reason,
    render_mutation_recovery_context,
)
from forge.runtime.session_manager import SessionManager
from forge.runtime.tool_runner import (
    ToolBatchState,
    ToolRunner,
    ToolRunPolicy,
    is_satisfied_non_diff_workspace_write,
    is_tool_protocol_failure,
)
from forge.runtime.model_client import (
    AnthropicModelClient,
    ModelClient,
)
from forge.runtime.completion import CompletionGate, TaskPolicy
from forge.runtime.background import BackgroundTaskManager
from forge.runtime.team import MessageBus, render_team_notification
from forge.runtime.state import (
    AgentPhaseChanged,
    CompletionBlocked,
    ConversationEvent,
    ContextCompacted,
    ModelCallCompleted,
    ModelCallStarted,
    ModelTextDelta,
    ModelToolCallArgumentsDelta,
    ModelUsageUpdate,
    TokenUsage,
    ToolExecutionCompleted,
    ToolExecutionStarted,
    TurnCompleted,
    TurnResult,
    ToolCall,
    VerificationCompleted,
    VerificationEvidence,
    WorkspaceChanged,
)
from forge.runtime.task_scope import (
    TaskScope,
    evaluate_change_relevance,
    infer_task_scope,
)
from forge.runtime.tool_targets import mutation_target_paths
from forge.runtime.tool_executor import (
    PermissionMiddleware,
    ToolExecutionLogger,
    ToolExecutor,
    normalize_permission_mode,
    render_permission_notice,
)
from forge.runtime.workspace import WorkspaceTracker
from forge.sessions.store import SessionStore
from forge.tasks.manager import TaskManager
from forge.tools.base import ToolRegistry, ToolResult
from forge.tools.shell import RunCommandTool
from forge.tools.subagent import TaskSubagentTool
from forge.tools.task import create_task_tools
from forge.tools.team import create_team_tools
from forge.tools.todo import TodoList, TodoWriteTool


ACTION_RECOVERY_READ_TOOLS = frozenset(
    {'read_file', 'grep'}
)
VERIFICATION_RECOVERY_READ_TOOLS = frozenset(
    {'find_files', 'grep', 'read_file'}
)
ACTION_RECOVERY_EXCLUDED_WRITE_TOOLS = frozenset(
    {
        'create_directory',
        'task_create',
        'task_claim',
        'task_complete',
        'claim_next_task',
    }
)
PARENT_NOT_FOUND_WRITE_TOOLS = frozenset(
    {
        'apply_patch',
        'create_directory',
        'write_file',
        'write_file_chunk',
    }
)
InteractionMode = Literal['auto', 'plan', 'code']
PLAN_MODE_TOOLS = frozenset(
    {
        'list_directory',
        'find_files',
        'read_file',
        'grep',
        'git_status',
        'memory_list',
        'memory_read',
        'task',
        'task_list',
        'task_graph_get',
        'task_graph_plan',
        'todo_write',
        'finish_task',
    }
)


class ModelResponseError(RuntimeError):
    '''Raised when a model response cannot continue the Agent Loop.'''


class AgentLoopLimitError(RuntimeError):
    '''Raised when one user turn exceeds its model-call safety limit.'''


@dataclass(frozen=True, slots=True)
class TurnInterrupted:
    '''Persist an unexpected process-level turn interruption.'''

    error_type: str
    message: str


@cache
def load_system_prompt() -> str:
    '''Load the packaged ForgeCode identity and behavior prompt.'''
    prompt_path = Path(__file__).resolve().parents[1] / 'prompts' / 'system.md'
    prompt = prompt_path.read_text(encoding='utf-8').strip()
    if not prompt:
        raise RuntimeError('ForgeCode system prompt is empty.')
    return prompt


class Conversation:
    '''Keep model-visible message history for an interactive session.'''

    def __init__(
        self,
        client: ModelClient | None = None,
        system_prompt: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        registry: ToolRegistry | None = None,
        max_iterations: int | None = None,
        task_policy: TaskPolicy | None = None,
        max_completion_blocks: int = 3,
        context_config: CompactionConfig | None = None,
        context_root: Path | None = None,
        max_protocol_recoveries: int = 2,
        max_tool_protocol_recoveries: int = 3,
        max_output_continuations: int = 2,
        repeated_tool_limit: int = 2,
        stagnation_warning: int = 4,
        stagnation_limit: int = 16,
        completion_decision_limit: int = 8,
        mutation_recovery_limit: int = 5,
        pre_mutation_limit: int = 4,
        action_recovery_limit: int = 3,
        max_turn_input_tokens: int | None = None,
        max_turn_tool_calls: int | None = None,
        intent_classifier: SemanticTaskClassifier | None = None,
    ) -> None:
        if tools is not None and registry is not None:
            raise ValueError('Pass tools or registry, not both.')
        if max_iterations is not None and max_iterations < 1:
            raise ValueError('max_iterations must be positive')
        if max_completion_blocks < 1:
            raise ValueError('max_completion_blocks must be positive')
        if max_protocol_recoveries < 0:
            raise ValueError('max_protocol_recoveries must not be negative')
        if max_tool_protocol_recoveries < 1:
            raise ValueError(
                'max_tool_protocol_recoveries must be positive'
            )
        if max_output_continuations < 0:
            raise ValueError('max_output_continuations must not be negative')
        if repeated_tool_limit < 1:
            raise ValueError('repeated_tool_limit must be positive')
        if stagnation_warning < 1:
            raise ValueError('stagnation_warning must be positive')
        if stagnation_limit <= stagnation_warning:
            raise ValueError(
                'stagnation_limit must be greater than stagnation_warning'
            )
        if completion_decision_limit < 1:
            raise ValueError('completion_decision_limit must be positive')
        if mutation_recovery_limit < 1:
            raise ValueError('mutation_recovery_limit must be positive')
        if pre_mutation_limit < 1:
            raise ValueError('pre_mutation_limit must be positive')
        if action_recovery_limit < 1:
            raise ValueError('action_recovery_limit must be positive')
        if max_turn_input_tokens is not None and max_turn_input_tokens < 1:
            raise ValueError('max_turn_input_tokens must be positive')
        if max_turn_tool_calls is not None and max_turn_tool_calls < 1:
            raise ValueError('max_turn_tool_calls must be positive')
        config_root = (
            context_root
            or getattr(
                getattr(registry, 'workspace_tracker', None),
                'root',
                None,
            )
            or Path.cwd()
        )
        self.config_root = config_root
        self.client = (
            client
            if client is not None
            else AnthropicModelClient.from_config(config_cwd=config_root)
        )
        self.system_prompt = (
            system_prompt
            if system_prompt is not None
            else load_system_prompt()
        )
        self.messages: list[dict[str, Any]] = []
        self.registry = registry
        self.max_iterations = max_iterations
        tracker = (
            getattr(registry, 'workspace_tracker', None)
            if registry is not None
            else None
        )
        if task_policy is not None and tracker is None:
            raise ValueError(
                'task_policy requires a ToolRegistry with a '
                'WorkspaceTracker'
            )
        self.workspace_tracker: WorkspaceTracker | None = tracker
        resolved_context_root = (
            context_root
            if context_root is not None
            else tracker.root
            if tracker is not None
            else Path.cwd()
        )
        self.permission = PermissionMiddleware('trusted')
        self.todo_list = TodoList()
        self.todo_planning = TodoPlanningHook()
        self.tool_logger = ToolExecutionLogger(resolved_context_root)
        self.hook_registry = HookRegistry(
            [self.todo_planning, self.permission, self.tool_logger]
        )
        self.task_manager = TaskManager(resolved_context_root)
        self.background_manager = BackgroundTaskManager(resolved_context_root)
        self.team_bus = MessageBus(resolved_context_root)
        self.session_manager = SessionManager(
            SessionStore(resolved_context_root)
        )
        self.session_id: str | None = None
        self._rollout_enabled = False
        self._inflight_messages: list[dict[str, Any]] | None = None
        self.interaction_mode: InteractionMode = 'auto'
        self.working_state = WorkingState()
        self.run_state = AgentRunState()
        self.agent_controller = AgentController()
        if registry is not None:
            if 'task' in registry.names:
                registry.replace(
                    TaskSubagentTool(
                        resolved_context_root,
                        permission=self.permission,
                        workspace_tracker=tracker,
                        team_bus=self.team_bus,
                    )
                )
            if 'run_command' in registry.names:
                registry.replace(
                    RunCommandTool(
                        resolved_context_root,
                        background_manager=self.background_manager,
                    )
                )
            for task_tool in create_task_tools(
                resolved_context_root,
                self.task_manager,
            ):
                registry.register(task_tool)
            for team_tool in create_team_tools(
                resolved_context_root,
                bus=self.team_bus,
                agent_id='lead',
            ):
                if team_tool.name in registry.names:
                    registry.replace(team_tool)
                else:
                    registry.register(team_tool)
            if 'todo_write' not in registry.names:
                registry.register(
                    TodoWriteTool(resolved_context_root, self.todo_list)
                )
        tool_executor = (
            ToolExecutor(
                registry,
                root=resolved_context_root,
                workspace_tracker=tracker,
                permission=self.permission,
                logger=self.tool_logger,
                hooks=self.hook_registry,
            )
            if registry is not None
            else None
        )
        self.model_runner = ModelRunner(self.client)
        self.model_failure_handler = ModelFailureHandler()
        self.tool_runner = (
            ToolRunner(tool_executor)
            if tool_executor is not None
            else None
        )
        self.tools = registry.definitions if registry is not None else tools
        self.finish_protocol = (
            registry is not None and 'finish_task' in registry.names
        )
        self.context = ContextManager(
            self.messages,
            resolved_context_root,
            context_config,
        )
        completion_gate = (
            CompletionGate(tracker.root, task_policy)
            if tracker is not None
            else None
        )
        self.completion_checker = CompletionChecker(
            tracker,
            completion_gate,
            self.task_manager,
        )
        self.recovery_manager = RecoveryManager(
            self.tools,
            self.tool_runner,
            read_tools=ACTION_RECOVERY_READ_TOOLS,
            excluded_write_tools=ACTION_RECOVERY_EXCLUDED_WRITE_TOOLS,
        )
        self.request_builder = RequestBuilder(
            self.recovery_manager,
            action_recovery_limit=action_recovery_limit,
        )
        self.max_completion_blocks = max_completion_blocks
        self.max_protocol_recoveries = max_protocol_recoveries
        self.max_tool_protocol_recoveries = max_tool_protocol_recoveries
        self.max_output_continuations = max_output_continuations
        self.repeated_tool_limit = repeated_tool_limit
        self.stagnation_warning = stagnation_warning
        self.stagnation_limit = stagnation_limit
        self.completion_decision_limit = completion_decision_limit
        self.mutation_recovery_limit = mutation_recovery_limit
        self.pre_mutation_limit = pre_mutation_limit
        self.action_recovery_limit = action_recovery_limit
        self.max_turn_input_tokens = max_turn_input_tokens
        self.max_turn_tool_calls = max_turn_tool_calls
        self._intent_classifier_overridden = intent_classifier is not None
        self.intent_classifier = (
            intent_classifier
            if intent_classifier is not None
            else (
                None
                if getattr(self.client, 'provider', '') == 'fake'
                else ModelSemanticTaskClassifier(self.client)
            )
        )
        self._last_repository_context = self.context.repository.system_suffix('')
        self._last_task_context = ''

    @property
    def context_stats(self) -> ContextStats:
        '''Return current committed conversation context statistics.'''
        return self.context.stats_for_request(
            system_prompt=self._system_prompt_with_task(),
            repository_context=self._last_repository_context,
            tools=self.tools,
            context_window_tokens=getattr(
                self.client,
                'context_window',
                None,
            ),
            reserved_output_tokens=getattr(self.client, 'max_tokens', 0),
        )

    def _transition(
        self,
        phase: AgentPhase,
        *,
        reason: str,
        iteration: int,
    ) -> AgentPhaseChanged | None:
        transition = self.run_state.transition(
            phase,
            reason=reason,
            iteration=iteration,
        )
        if transition is None:
            return None
        return AgentPhaseChanged(
            phase=transition.current,
            previous_phase=transition.previous,
            reason=transition.reason,
            iteration=transition.iteration,
        )

    async def stream(self, prompt: str) -> AsyncIterator[ConversationEvent]:
        '''Expose lifecycle transitions around the internal turn engine.'''
        if self._rollout_enabled and prompt.strip():
            self._inflight_messages = [
                *self.messages,
                {'role': 'user', 'content': prompt},
            ]
        try:
            async for event in self._stream_turn(prompt):
                if isinstance(event, TurnCompleted):
                    phase_event = self._transition(
                        (
                            AgentPhase.COMPLETED
                            if event.result.status == 'completed'
                            else AgentPhase.FAILED
                        ),
                        reason=f'turn_{event.result.status}',
                        iteration=self.run_state.iteration,
                    )
                    if phase_event is not None:
                        self._persist_rollout_event(phase_event)
                        yield phase_event
                self._persist_rollout_event(event)
                yield event
        except asyncio.CancelledError:
            phase_event = self._transition(
                AgentPhase.FAILED,
                reason='user_interrupted',
                iteration=self.run_state.iteration,
            )
            if phase_event is not None:
                self._persist_rollout_event(phase_event)
            self._persist_rollout_interruption(
                RuntimeError('User interrupted the active turn with Esc.')
            )
            raise
        except Exception as error:
            phase_event = self._transition(
                AgentPhase.FAILED,
                reason='unhandled_turn_error',
                iteration=self.run_state.iteration,
            )
            if phase_event is not None:
                self._persist_rollout_event(phase_event)
                yield phase_event
            self._persist_rollout_interruption(error)
            raise
        finally:
            self._inflight_messages = None

    async def _stream_turn(
        self,
        prompt: str,
    ) -> AsyncIterator[ConversationEvent]:
        '''Run model-tool cycles until the model returns a final text answer.'''
        if not prompt.strip():
            raise ValueError('prompt must not be empty')

        self.task_manager.begin_turn(prompt)
        self.working_state = WorkingState()
        self.run_state = AgentRunState()
        task_contract = await self._resolved_task_contract(prompt)
        self.agent_controller.begin_turn(task_contract)
        runtime = self.agent_controller.snapshot()
        runtime.budget.max_model_calls = self.max_iterations
        runtime.budget.max_tool_calls = self.max_turn_tool_calls
        verification_state = runtime.verification
        edit_recovery = runtime.edit_recovery
        action_recovery_state = runtime.action_recovery_state
        synthesis_state = runtime.synthesis
        self.todo_planning.configure(required=task_contract.requires_plan)
        await self.hook_registry.run(
            HookContext(
                event='user_prompt_submit',
                root=self.task_manager.root,
                prompt=prompt,
                permission_mode=self.permission.mode,
                metadata={'todo_required': task_contract.requires_plan},
            )
        )
        self._last_task_context = self.task_manager.system_suffix()
        user_message = {'role': 'user', 'content': prompt}
        request_messages = (
            self._inflight_messages
            if self._inflight_messages is not None
            else [*self.messages, user_message]
        )
        completed_usage = TokenUsage(input_tokens=0, output_tokens=0)
        all_tool_calls: list[ToolCall] = []
        verification_state.latest = None
        verification_state.repair_target = None
        mutation_attempted = False
        change_required = task_contract.requires_change
        tool_attempts: dict[str, tuple[int, bool]] = {}
        calls_without_progress = 0
        pre_mutation_calls = 0
        action_recovery_state.calls = 0
        action_recovery_state.read_used = False
        action_recovery_state.block_events = 0
        edit_recovery.failure_count = 0
        edit_recovery.failures = []
        edit_recovery.read_used = False
        edit_recovery.context = ''
        synthesis_state.clear()
        tool_protocol_failures = 0
        synthesis_retries = 0
        completion_blocks = 0
        last_completion_reasons: tuple[str, ...] = ()
        verification_state.failed_revision = None
        verification_state.read_count = 0
        verification_state.recovery_calls = 0
        verification_state.last_failure_signature = ''
        token_limit_reason = ''
        completion_ready_revision: int | None = None
        completion_decision_calls = 0
        completion_ready_context = ''
        completion_reviewed_paths: set[str] = set()
        if self.workspace_tracker is not None:
            await self.workspace_tracker.begin_turn()

        self._last_repository_context = (
            self.context.repository.system_suffix(prompt)
        )
        reactive_compaction_attempted = False
        protocol_recoveries = 0
        output_continuations = 0
        continued_text_parts: list[str] = []

        iterations = (
            count(1)
            if self.max_iterations is None
            else range(1, self.max_iterations + 1)
        )
        for iteration in iterations:
            control_state = self.agent_controller.state
            is_recovering = (
                bool(edit_recovery.failures)
                or self.agent_controller.planning_recovery
                or verification_state.recovery_active(runtime.control_state)
                or synthesis_state.mode
                in {
                    SynthesisMode.FINALIZATION,
                    SynthesisMode.STAGNATION_FINAL,
                    SynthesisMode.TOKEN_LIMIT,
                }
                or (
                    control_state
                    in {
                        AgentControlState.TASK_PLANNING,
                        AgentControlState.TARGETED_ANALYSIS,
                        AgentControlState.FIX_REQUIRED,
                    }
                )
            )
            phase_event = self._transition(
                (
                    AgentPhase.RECOVERING
                    if is_recovering
                    else AgentPhase.THINKING
                ),
                reason=(
                    'preparing_recovery_request'
                    if is_recovering
                    else 'preparing_model_request'
                ),
                iteration=iteration,
            )
            if phase_event is not None:
                yield phase_event
            text_parts: list[str] = []
            tool_calls: list[ToolCall] = []
            request_usage: TokenUsage | None = None
            background_notifications = (
                self.background_manager.collect_notifications()
            )
            if background_notifications:
                append_notification_message(
                    request_messages,
                    background_notifications,
                )
            team_notifications = render_team_notification(
                self.team_bus.collect('lead')
            )
            if team_notifications:
                append_notification_message(
                    request_messages,
                    team_notifications,
                )

            if (
                self.max_turn_input_tokens is not None
                and synthesis_state.mode is not SynthesisMode.TOKEN_LIMIT
                and completed_usage.total_input_tokens
                >= self.max_turn_input_tokens
            ):
                token_limit_reason = (
                    'Stopped after the turn consumed '
                    f'{completed_usage.total_input_tokens} input tokens, '
                    'reaching the configured cumulative input-token limit '
                    f'of {self.max_turn_input_tokens}.'
                )
                synthesis_state.mode = SynthesisMode.TOKEN_LIMIT
                request_messages.append(
                    build_token_limit_recovery_feedback(
                        token_limit_reason,
                        self.task_manager.system_suffix(),
                        self.working_state.system_suffix(),
                    )
                )

            runtime = self.agent_controller.snapshot()
            runtime.completion_ready_context = completion_ready_context
            request_state = RequestState(
                runtime=runtime,
                control_state=self.agent_controller.state,
                task_contract=task_contract,
                change_required=change_required,
                mutation_attempted=mutation_attempted,
                task_scope_patterns=self.completion_checker.task_scope_patterns(
                    evidence_paths=self.working_state.evidence_paths,
                ),
                task_goal=(
                    self.task_manager.active.goal
                    if self.task_manager.active is not None
                    else prompt
                ),
            )
            request_spec = self.request_builder.build(
                state=request_state,
                interaction_mode=self.interaction_mode,
                all_tools=self.tools,
                plan_tools=self._plan_mode_tools(),
                base_system_prompt=self._system_prompt_with_task(
                    include_tool_availability=(
                        not request_state.tool_free_recovery
                        and runtime.synthesis.mode
                        is not SynthesisMode.FINALIZATION
                    ),
                ),
                repository_context=self._last_repository_context,
                changed_paths=(
                    self.workspace_tracker.changed_paths
                    if self.workspace_tracker is not None
                    else ()
                ),
            )
            request_tools = request_spec.tools
            request_tool_names = request_spec.tool_names
            request_system_prompt = request_spec.system_prompt
            compaction_report = await self.context.compact_history(
                request_messages,
                self.client,
                system_prompt=request_system_prompt,
                repository_context=self._last_repository_context,
                tools=request_tools,
                context_window_tokens=getattr(
                    self.client,
                    'context_window',
                    None,
                ),
                reserved_output_tokens=getattr(
                    self.client,
                    'max_tokens',
                    0,
                ),
            )
            if (
                compaction_report is not None
                and compaction_report.success
                and compaction_report.automatic
            ):
                yield ContextCompacted(
                    before_characters=compaction_report.before_characters,
                    after_characters=compaction_report.after_characters,
                    transcript_path=compaction_report.transcript_path,
                    automatic=compaction_report.automatic,
                )
            runtime.budget.observe_model_call()
            budget_reasons = runtime.budget.exceeded()
            if budget_reasons:
                self.task_manager.stuck(budget_reasons)
                self.messages[:] = request_messages
                self.context.capture_explicit_memory(prompt)
                yield TurnCompleted(
                    result=TurnResult(
                        text='Stopped after exceeding turn budget.',
                        usage=completed_usage,
                        last_request_usage=request_usage,
                        model_calls=iteration,
                        tool_calls=tuple(all_tool_calls),
                        status='stuck',
                        changed_paths=(
                            self.workspace_tracker.changed_paths
                            if self.workspace_tracker is not None
                            else ()
                        ),
                        verification=verification_state.latest,
                        completion_reasons=budget_reasons,
                    )
                )
                return
            yield ModelCallStarted(iteration=iteration)
            model_run = self.model_runner.run(
                messages=self.context.prepare(request_messages),
                tools=request_tools,
                system=request_system_prompt,
                completed_usage=completed_usage,
                iteration=iteration,
            )
            text_parts = model_run.text_parts
            tool_calls = model_run.tool_calls
            try:
                async for event in model_run:
                    yield event
            except Exception as error:
                partial_text = ''.join(text_parts)
                failure = self.model_failure_handler.classify(
                    error,
                    iteration=iteration,
                    partial_text=partial_text,
                    has_tool_calls=bool(tool_calls),
                    request_usage=model_run.request_usage,
                    output_continuations=output_continuations,
                    max_output_continuations=(
                        self.max_output_continuations
                    ),
                    reactive_compaction_attempted=(
                        reactive_compaction_attempted
                    ),
                    protocol_recoveries=protocol_recoveries,
                    max_protocol_recoveries=self.max_protocol_recoveries,
                    available_tools=tuple(sorted(request_tool_names)),
                )
                if failure.action is ModelFailureAction.COMPACT_CONTEXT:
                    reactive_compaction_attempted = True
                    report = await self.context.compact_history(
                        request_messages,
                        self.client,
                        force=True,
                    )
                    if report is not None and report.success:
                        continue
                    yield failure.event
                    raise
                if failure.action is ModelFailureAction.CONTINUE_OUTPUT:
                    output_continuations += 1
                    if (
                        failure.consume_usage
                        and model_run.request_usage is None
                    ):
                        raise AssertionError(
                            'Output continuation requires request usage.'
                        )
                    if model_run.request_usage is not None:
                        completed_usage = add_token_usage(
                            completed_usage,
                            model_run.request_usage,
                        )
                    if failure.preserve_partial_text:
                        continued_text_parts.append(partial_text)
                    request_messages.extend(
                        [
                            {
                                'role': 'assistant',
                                'content': partial_text,
                            },
                            *failure.feedback,
                        ]
                    )
                    yield failure.event
                    continue
                if failure.action is ModelFailureAction.RECOVER_PROTOCOL:
                    protocol_recoveries += 1
                    if (
                        failure.consume_usage
                        and model_run.request_usage is not None
                    ):
                        completed_usage = add_token_usage(
                            completed_usage,
                            model_run.request_usage,
                        )
                    yield failure.event
                    request_messages.extend(failure.feedback)
                    continue
                yield failure.event
                raise
            request_usage = model_run.request_usage
            yield ModelCallCompleted(iteration=iteration)

            text = ''.join(text_parts).strip()
            complete_text = ''.join(
                [*continued_text_parts, text]
            ).strip()
            if not text and not tool_calls:
                if (
                    request_usage is not None
                    and self._pending_required_change(
                        change_required,
                        mutation_attempted=mutation_attempted,
                    )
                ):
                    completed_usage = add_token_usage(
                        completed_usage,
                        request_usage,
                    )
                    if self.agent_controller.action_recovery:
                        action_recovery_state.calls += 1
                    else:
                        self.agent_controller.enter_targeted_analysis()
                        action_recovery_state.calls = 0
                        action_recovery_state.read_used = False
                    action_recovery_state.block_events += 1
                    change_reason = required_change_block_reason()
                    yield CompletionBlocked(
                        attempt=action_recovery_state.block_events,
                        reasons=(change_reason,),
                    )
                    if (
                        action_recovery_state.calls
                        >= self.action_recovery_limit
                    ):
                        reason = action_recovery_stuck_reason(
                            action_recovery_state.calls
                        )
                        self.task_manager.stuck((reason, change_reason))
                        self.messages[:] = request_messages
                        yield TurnCompleted(
                            result=TurnResult(
                                text=reason,
                                usage=completed_usage,
                                last_request_usage=request_usage,
                                model_calls=iteration,
                                tool_calls=tuple(all_tool_calls),
                                status='stuck',
                                changed_paths=(),
                                verification=verification_state.latest,
                                completion_reasons=(
                                    reason,
                                    change_reason,
                                ),
                            )
                        )
                        return
                    request_messages.append(
                        build_action_recovery_feedback(
                            self.task_manager.system_suffix(),
                            action_recovery_state.calls,
                            self.action_recovery_limit,
                            read_used=action_recovery_state.read_used,
                        )
                    )
                    continue
                if (
                    (
                        synthesis_state.mode is not SynthesisMode.NORMAL
                        or completion_blocks > 0
                    )
                    and request_usage is not None
                ):
                    completed_usage = add_token_usage(
                        completed_usage,
                        request_usage,
                    )
                    reason = (
                        'The model returned no usable answer after ForgeCode '
                        'requested a final synthesis or completion recovery.'
                    )
                    reasons = (reason, *last_completion_reasons)
                    self.task_manager.stuck(reasons)
                    self.messages[:] = request_messages
                    yield TurnCompleted(
                        result=TurnResult(
                            text=reason,
                            usage=completed_usage,
                            last_request_usage=request_usage,
                            model_calls=iteration,
                            tool_calls=tuple(all_tool_calls),
                            status='stuck',
                            changed_paths=(
                                self.workspace_tracker.changed_paths
                                if self.workspace_tracker is not None
                                else ()
                            ),
                            verification=verification_state.latest,
                            completion_reasons=reasons,
                        )
                    )
                    return
                if self.finish_protocol and request_usage is not None:
                    completed_usage = add_token_usage(
                        completed_usage,
                        request_usage,
                    )
                    reason = (
                        'The model returned no text or tool action, so the '
                        'trajectory cannot continue.'
                    )
                    self.task_manager.stuck((reason,))
                    self.messages[:] = request_messages
                    yield TurnCompleted(
                        result=TurnResult(
                            text=reason,
                            usage=completed_usage,
                            last_request_usage=request_usage,
                            model_calls=iteration,
                            tool_calls=tuple(all_tool_calls),
                            status='stuck',
                            changed_paths=(
                                self.workspace_tracker.changed_paths
                                if self.workspace_tracker is not None
                                else ()
                            ),
                            verification=verification_state.latest,
                            completion_reasons=(reason,),
                        )
                    )
                    return
                raise ModelResponseError(
                    'Model response did not contain any text or tool calls.'
                )
            if request_usage is None:
                raise ModelResponseError(
                    'Model response did not contain token usage.'
                )

            completed_usage = add_token_usage(
                completed_usage,
                request_usage,
            )
            tool_calls.sort(key=lambda call: call.index)
            request_messages.append(
                build_assistant_message(text, tool_calls)
            )

            if (
                synthesis_state.mode
                in {
                    SynthesisMode.FINALIZATION,
                    SynthesisMode.STAGNATION_FINAL,
                    SynthesisMode.TOKEN_LIMIT,
                }
            ) and tool_calls:
                finalization_finish = (
                    synthesis_state.mode is SynthesisMode.FINALIZATION
                    and len(tool_calls) == 1
                    and tool_calls[0].name == 'finish_task'
                )
                if finalization_finish:
                    pass
                else:
                    all_tool_calls.extend(tool_calls)
                    if synthesis_state.mode is SynthesisMode.FINALIZATION:
                        recovery_name = 'finalization recovery'
                    elif (
                        synthesis_state.mode
                        is SynthesisMode.STAGNATION_FINAL
                    ):
                        recovery_name = 'stagnation final recovery'
                    else:
                        recovery_name = 'token-limit recovery'
                    reason = (
                        f'The model requested another tool during the dedicated '
                        f'{recovery_name} instead of returning its final '
                        'evidence-based answer.'
                    )
                    self.task_manager.stuck((reason,))
                    self.messages[:] = request_messages
                    yield TurnCompleted(
                        result=TurnResult(
                            text=reason,
                            usage=completed_usage,
                            last_request_usage=request_usage,
                            model_calls=iteration,
                            tool_calls=tuple(all_tool_calls),
                            status='stuck',
                            changed_paths=(
                                self.workspace_tracker.changed_paths
                                if self.workspace_tracker is not None
                                else ()
                            ),
                            verification=verification_state.latest,
                            completion_reasons=(reason,),
                        )
                    )
                    return
            if not tool_calls:
                phase_event = self._transition(
                    AgentPhase.CHECKING_RESULT,
                    reason='model_returned_final_text',
                    iteration=iteration,
                )
                if phase_event is not None:
                    yield phase_event
                if synthesis_state.mode is SynthesisMode.TOKEN_LIMIT:
                    reason = (
                        token_limit_reason
                        or 'The turn reached the cumulative input-token limit.'
                    )
                    self.task_manager.stuck((reason,))
                    self.messages[:] = request_messages
                    self.context.capture_explicit_memory(prompt)
                    yield TurnCompleted(
                        result=TurnResult(
                            text=complete_text,
                            usage=completed_usage,
                            last_request_usage=request_usage,
                            model_calls=iteration,
                            tool_calls=tuple(all_tool_calls),
                            status='stuck',
                            changed_paths=(
                                self.workspace_tracker.changed_paths
                                if self.workspace_tracker is not None
                                else ()
                            ),
                            verification=verification_state.latest,
                            completion_reasons=(reason,),
                        )
                    )
                    return
                if edit_recovery.failures:
                    reason = (
                        f'Stopped after {edit_recovery.failure_count} failed '
                        'workspace-write attempt(s) because the model '
                        'returned text without correcting the latest edit '
                        'failure.'
                    )
                    self.task_manager.stuck((reason,))
                    self.messages[:] = request_messages
                    yield TurnCompleted(
                        result=TurnResult(
                            text=reason,
                            usage=completed_usage,
                            last_request_usage=request_usage,
                            model_calls=iteration,
                            tool_calls=tuple(all_tool_calls),
                            status='stuck',
                            changed_paths=(
                                self.workspace_tracker.changed_paths
                                if self.workspace_tracker is not None
                                else ()
                            ),
                            verification=verification_state.latest,
                            completion_reasons=(reason,),
                        )
                    )
                    return
                if self._pending_required_change(
                    change_required,
                    mutation_attempted=mutation_attempted,
                ):
                    if self.agent_controller.action_recovery:
                        action_recovery_state.calls += 1
                    else:
                        self.agent_controller.enter_targeted_analysis()
                        action_recovery_state.calls = 0
                        action_recovery_state.read_used = False
                    action_recovery_state.block_events += 1
                    change_reason = required_change_block_reason()
                    yield CompletionBlocked(
                        attempt=action_recovery_state.block_events,
                        reasons=(change_reason,),
                    )
                    if (
                        action_recovery_state.calls
                        >= self.action_recovery_limit
                    ):
                        reason = action_recovery_stuck_reason(
                            action_recovery_state.calls
                        )
                        self.task_manager.stuck((reason, change_reason))
                        self.messages[:] = request_messages
                        yield TurnCompleted(
                            result=TurnResult(
                                text=reason,
                                usage=completed_usage,
                                last_request_usage=request_usage,
                                model_calls=iteration,
                                tool_calls=tuple(all_tool_calls),
                                status='stuck',
                                changed_paths=(),
                                verification=verification_state.latest,
                                completion_reasons=(
                                    reason,
                                    change_reason,
                                ),
                            )
                        )
                        return
                    request_messages.append(
                        build_action_recovery_feedback(
                            self.task_manager.system_suffix(),
                            action_recovery_state.calls,
                            self.action_recovery_limit,
                            read_used=action_recovery_state.read_used,
                        )
                    )
                    continue
                if (
                    synthesis_state.mode is not SynthesisMode.NORMAL
                    and self.working_state.evidence_paths
                    and not self.working_state.answer_mentions_evidence(
                        complete_text
                    )
                ):
                    synthesis_retries += 1
                    reason = (
                        'The synthesis did not reference any collected '
                        'repository evidence.'
                    )
                    if synthesis_retries <= 1:
                        request_messages.append(
                            build_synthesis_retry_feedback(
                                self.task_manager.system_suffix(),
                                self.working_state.system_suffix(),
                            )
                        )
                        continue
                    self.task_manager.stuck((reason,))
                    self.messages[:] = request_messages
                    yield TurnCompleted(
                        result=TurnResult(
                            text=complete_text,
                            usage=completed_usage,
                            last_request_usage=request_usage,
                            model_calls=iteration,
                            tool_calls=tuple(all_tool_calls),
                            status='stuck',
                            changed_paths=(
                                self.workspace_tracker.changed_paths
                                if self.workspace_tracker is not None
                                else ()
                            ),
                            verification=verification_state.latest,
                            completion_reasons=(reason,),
                        )
                    )
                    return
                if (
                    self.workspace_tracker is not None
                    and self.completion_checker.available
                ):
                    change = await self.workspace_tracker.refresh()
                    if change is not None:
                        self.working_state.advance_revision(
                            change.revision,
                            change.paths,
                        )
                        yield WorkspaceChanged(
                            revision=change.revision,
                            paths=change.paths,
                        )
                    decision = await self.completion_checker.evaluate(
                        verification_state.latest,
                        mutation_attempted=(
                            mutation_attempted or change_required
                        ),
                        reviewed_paths=completion_reviewed_paths,
                        evidence_paths=self.working_state.evidence_paths,
                    )
                    if not decision.allowed:
                        last_completion_reasons = decision.reasons
                        completion_blocks += 1
                        yield CompletionBlocked(
                            attempt=completion_blocks,
                            reasons=decision.reasons,
                        )
                        if request_state.tool_free_recovery:
                            reasons = (
                                'The agent stopped making progress before '
                                'the task satisfied its completion checks.',
                                *decision.reasons,
                            )
                            self.task_manager.stuck(reasons)
                            self.messages[:] = request_messages
                            self.context.capture_explicit_memory(prompt)
                            yield TurnCompleted(
                                result=TurnResult(
                                    text=complete_text,
                                    usage=completed_usage,
                                    last_request_usage=request_usage,
                                    model_calls=iteration,
                                    tool_calls=tuple(all_tool_calls),
                                    status='stuck',
                                    changed_paths=(
                                        self.workspace_tracker.changed_paths
                                    ),
                                    verification=verification_state.latest,
                                    completion_reasons=reasons,
                                )
                            )
                            return
                        if only_verification_blocked(decision.reasons):
                            verification_state.recovery_calls += 1
                            if (
                                verification_state.recovery_calls
                                > self.action_recovery_limit
                            ):
                                self.task_manager.stuck(decision.reasons)
                                self.messages[:] = request_messages
                                self.context.capture_explicit_memory(prompt)
                                yield TurnCompleted(
                                    result=TurnResult(
                                        text=complete_text,
                                        usage=completed_usage,
                                        last_request_usage=request_usage,
                                        model_calls=iteration,
                                        tool_calls=tuple(all_tool_calls),
                                        status='stuck',
                                        changed_paths=(
                                            self.workspace_tracker.changed_paths
                                        ),
                                        verification=verification_state.latest,
                                        completion_reasons=decision.reasons,
                                    )
                                )
                                return
                            verification_failure_blocked = any(
                                marker in reason
                                for marker in (
                                    'latest verification failed',
                                    'latest verification timed out',
                                    'latest verification command was invalid',
                                )
                                for reason in decision.reasons
                            )
                            verification_needs_repair = (
                                verification_failure_blocked
                                and verification_state.latest is not None
                                and not verification_state.latest.success
                            )
                            verification_state.failed_revision = (
                                verification_state.latest.workspace_revision
                                if verification_needs_repair
                                and verification_state.latest is not None
                                else None
                            )
                            if verification_needs_repair:
                                self.agent_controller.enter_fix_required()
                            else:
                                self.agent_controller.enter_ready_to_verify()
                            if (
                                synthesis_state.mode
                                is SynthesisMode.CHECKPOINT
                            ):
                                synthesis_state.clear()
                            synthesis_retries = 0
                            if (
                                synthesis_state.mode
                                is SynthesisMode.STAGNATION_FINAL
                            ):
                                synthesis_state.clear()
                            request_messages.append(
                                build_completion_feedback(
                                    decision.reasons,
                                    task_context=(
                                        self.task_manager.system_suffix()
                                    ),
                                )
                            )
                            continue
                        if completion_blocks < self.max_completion_blocks:
                            request_messages.append(
                                build_completion_feedback(
                                    decision.reasons,
                                    task_context=(
                                        self.task_manager.system_suffix()
                                    ),
                                )
                            )
                            continue
                        self.task_manager.stuck(decision.reasons)
                        self.messages[:] = request_messages
                        self.context.capture_explicit_memory(prompt)
                        yield TurnCompleted(
                            result=TurnResult(
                                text=complete_text,
                                usage=completed_usage,
                                last_request_usage=request_usage,
                                model_calls=iteration,
                                tool_calls=tuple(all_tool_calls),
                                status='stuck',
                                changed_paths=(
                                    self.workspace_tracker.changed_paths
                                ),
                                verification=verification_state.latest,
                                completion_reasons=decision.reasons,
                            )
                        )
                        return
                self.task_manager.complete()
                self.messages[:] = request_messages
                self.context.capture_explicit_memory(prompt)
                yield TurnCompleted(
                    result=TurnResult(
                        text=complete_text,
                        usage=completed_usage,
                        last_request_usage=request_usage,
                        model_calls=iteration,
                        tool_calls=tuple(all_tool_calls),
                        changed_paths=(
                            self.workspace_tracker.changed_paths
                            if self.workspace_tracker is not None
                            else ()
                        ),
                        verification=verification_state.latest,
                    )
                )
                return

            if self.registry is None:
                raise ModelResponseError(
                    'Model requested tools, but no ToolRegistry is configured.'
                )
            if self.tool_runner is None:
                raise ModelResponseError(
                    'Model requested tools, but no ToolExecutor is configured.'
                )

            phase_event = self._transition(
                AgentPhase.PREPARING_TOOLS,
                reason='model_requested_tool_calls',
                iteration=iteration,
            )
            if phase_event is not None:
                yield phase_event

            all_tool_calls.extend(tool_calls)
            batch = ToolBatchState()
            evidence_paths_before_batch = set(self.working_state.evidence_paths)
            for tool_position, tool_call in enumerate(tool_calls):
                finish_rejection: tuple[str, ...] = ()
                tool_effect = self.tool_runner.effect(tool_call.name)
                runtime.budget.observe_tool_call(tool_effect, tool_call.name)
                budget_reasons = runtime.budget.exceeded()
                if budget_reasons:
                    self.task_manager.stuck(budget_reasons)
                    self.messages[:] = request_messages
                    self.context.capture_explicit_memory(prompt)
                    yield TurnCompleted(
                        result=TurnResult(
                            text='Stopped after exceeding turn budget.',
                            usage=completed_usage,
                            last_request_usage=request_usage,
                            model_calls=iteration,
                            tool_calls=tuple(all_tool_calls),
                            status='stuck',
                            changed_paths=(
                                self.workspace_tracker.changed_paths
                                if self.workspace_tracker is not None
                                else ()
                            ),
                            verification=verification_state.latest,
                            completion_reasons=budget_reasons,
                        )
                    )
                    return
                if tool_effect == 'workspace_write':
                    mutation_attempted = True
                    change_required = True
                if (
                    tool_call.name == 'finish_task'
                    and tool_call.arguments.get('task_kind') == 'change'
                ):
                    change_required = True
                action_read_call = (
                    self.agent_controller.action_recovery
                    and tool_call.name in ACTION_RECOVERY_READ_TOOLS
                )
                action_read_exhausted = (
                    action_read_call and action_recovery_state.read_used
                )
                if action_read_call and not action_recovery_state.read_used:
                    action_recovery_state.read_used = True
                mutation_read_call = (
                    bool(edit_recovery.failures)
                    and tool_call.name in ACTION_RECOVERY_READ_TOOLS
                )
                if (
                    mutation_read_call
                    and not edit_recovery.read_used
                ):
                    edit_recovery.read_used = True
                verification_read_call = (
                    verification_state.recovery_active(runtime.control_state)
                    and verification_state.requires_repair(
                        runtime.control_state
                    )
                    and tool_call.name in VERIFICATION_RECOVERY_READ_TOOLS
                )
                verification_read_budget = (
                    self.recovery_manager.verification_read_budget(
                        verification_state.repair_target
                    )
                )
                verification_read_exhausted = (
                    verification_read_call
                    and verification_state.read_count >= verification_read_budget
                )
                if (
                    verification_read_call
                    and verification_state.read_count < verification_read_budget
                ):
                    verification_state.read_count += 1
                yield ToolExecutionStarted(tool_call=tool_call)
                phase_event = self._transition(
                    AgentPhase.EXECUTING_TOOLS,
                    reason=f'executing_tool:{tool_call.name}',
                    iteration=iteration,
                )
                if phase_event is not None:
                    yield phase_event
                revision = (
                    self.workspace_tracker.revision
                    if self.workspace_tracker is not None
                    else 0
                )
                signature = tool_call_signature(tool_call, revision)
                previous_count, previous_success = tool_attempts.get(
                    signature,
                    (0, True),
                )
                early_relevance_failure = early_mutation_relevance_failure(
                    tool_call,
                    tool_effect=tool_effect,
                    change_required=change_required,
                    task_scope_patterns=(
                        self._static_task_scope_patterns(task_contract)
                    ),
                )
                semantic_repeat = self.working_state.preflight(
                    tool_call,
                    revision,
                    signature,
                )
                run = await self.tool_runner.transact(
                    tool_call,
                    ToolRunPolicy(
                        tool_count=len(tool_calls),
                        available_tools=frozenset(request_tool_names),
                        runtime=runtime,
                        control_state=runtime.control_state,
                        action_read_exhausted=action_read_exhausted,
                        verification_read_exhausted=(
                            verification_read_exhausted
                        ),
                        semantic_repeat=(
                            early_relevance_failure or semantic_repeat
                        ),
                        previous_count=previous_count,
                        previous_success=previous_success,
                        repeated_limit=self.repeated_tool_limit,
                    ),
                    revision=revision,
                    signature=signature,
                )
                result = run.result
                self.agent_controller.observe_tool_result(
                    tool_call.name,
                    result,
                )
                if (
                    run.executed
                    and tool_call.name != 'finish_task'
                    and not is_todo_required_result(result)
                ):
                    tool_attempts[signature] = (
                        previous_count + 1,
                        result.success,
                    )
                if (
                    run.executed
                    and tool_call.name == 'create_directory'
                    and result.success
                ):
                    tool_attempts = {
                        attempted_signature: attempt
                        for attempted_signature, attempt in tool_attempts.items()
                        if attempt[1]
                    }
                if tool_call.name == 'finish_task' and result.success:
                    finish_reasons = await self.completion_checker.finish_rejection_reasons(
                        result,
                        working_state=self.working_state,
                        mutation_attempted=mutation_attempted,
                        change_required=change_required,
                        verification=verification_state.latest,
                        reviewed_paths=completion_reviewed_paths,
                        evidence_paths=self.working_state.evidence_paths,
                    )
                    if (
                        result.metadata.get('status') != 'blocked'
                        and edit_recovery.failures
                    ):
                        finish_reasons = (
                            'A workspace-write failure is still unresolved. '
                            'Produce a real workspace revision that clears '
                            'Edit Recovery before declaring completion.',
                            *finish_reasons,
                        )
                        finish_reasons = tuple(
                            dict.fromkeys(finish_reasons)
                        )
                    if finish_reasons:
                        finish_rejection = finish_reasons
                        last_completion_reasons = finish_reasons
                        pending_required_change = (
                            self._pending_required_change(
                                change_required,
                                mutation_attempted=mutation_attempted,
                            )
                        )
                        if pending_required_change:
                            batch.required_change_rejected = True
                            action_recovery_state.block_events += 1
                        else:
                            completion_blocks += 1
                        if synthesis_state.mode is SynthesisMode.CHECKPOINT:
                            synthesis_state.clear()
                        calls_without_progress = 0
                        result = ToolResult.fail(
                            'finish_rejected',
                            'The finish_task declaration did not match the '
                            'available execution evidence.',
                            details={'reasons': list(finish_reasons)},
                        )
                        if (
                            not pending_required_change
                            and completion_blocks
                            >= self.max_completion_blocks
                        ):
                            batch.terminal_finish_reasons = finish_reasons
                    else:
                        batch.accepted_finish = result
                batch.evidence_progressed = (
                    self.working_state.observe(
                        tool_call,
                        result,
                        revision,
                        signature,
                    )
                    or batch.evidence_progressed
                )
                batch.results.append((tool_call, result))
                yield ToolExecutionCompleted(
                    tool_call=tool_call,
                    result=result,
                )
                if result.metadata.get('permission_terminal'):
                    reason = result.summary
                    self.task_manager.block((reason,))
                    request_messages.append(
                        build_tool_result_message(batch.results)
                    )
                    self.messages[:] = request_messages
                    yield TurnCompleted(
                        result=TurnResult(
                            text=reason,
                            usage=completed_usage,
                            last_request_usage=request_usage,
                            model_calls=iteration,
                            tool_calls=tuple(all_tool_calls),
                            status='blocked',
                            changed_paths=(
                                self.workspace_tracker.changed_paths
                                if self.workspace_tracker is not None
                                else ()
                            ),
                            verification=verification_state.latest,
                            completion_reasons=(reason,),
                        )
                    )
                    return
                if finish_rejection:
                    yield CompletionBlocked(
                        attempt=(
                            action_recovery_state.block_events
                            if batch.required_change_rejected
                            else completion_blocks
                        ),
                        reasons=finish_rejection,
                    )
                if (
                    tool_call.name == 'task'
                    and not result.success
                    and self._pending_required_change(
                        change_required,
                        mutation_attempted=mutation_attempted,
                    )
                ):
                    batch.required_change_rejected = True
                tool_changed_workspace = False
                if self.workspace_tracker is not None:
                    change = await self.workspace_tracker.refresh()
                    if change is not None:
                        tool_changed_workspace = True
                        batch.last_workspace_change_position = tool_position
                        self.working_state.advance_revision(
                            change.revision,
                            change.paths,
                        )
                        if tool_effect == 'process':
                            mutation_attempted = True
                        yield WorkspaceChanged(
                            revision=change.revision,
                            paths=change.paths,
                        )
                    elif is_satisfied_non_diff_workspace_write(
                        tool_call,
                        result,
                    ):
                        tool_changed_workspace = True
                elif tool_effect == 'workspace_write' and result.success:
                    tool_changed_workspace = True
                    batch.last_workspace_change_position = tool_position
                if tool_effect == 'workspace_write':
                    batch.workspace_writes.append(
                        (
                            tool_position,
                            tool_call,
                            result,
                            tool_changed_workspace,
                        )
                    )
                if tool_call.name == 'verify':
                    if (
                        result.error is not None
                        and result.error.code == 'repeated_tool_call'
                        and verification_state.latest is not None
                        and not verification_state.latest.success
                    ):
                        reason = (
                            'Verification Recovery stopped after the same '
                            'verification failure repeated.'
                        )
                        self.task_manager.stuck((reason,))
                        self.messages[:] = request_messages
                        yield TurnCompleted(
                            result=TurnResult(
                                text=reason,
                                usage=completed_usage,
                                last_request_usage=request_usage,
                                model_calls=iteration,
                                tool_calls=tuple(all_tool_calls),
                                status='stuck',
                                changed_paths=(
                                    self.workspace_tracker.changed_paths
                                    if self.workspace_tracker is not None
                                    else ()
                                ),
                                verification=verification_state.latest,
                                completion_reasons=(reason,),
                            )
                        )
                        return
                    verification_evidence = verification_from_result(result)
                    if verification_evidence is None:
                        continue
                    verification_state.latest = verification_evidence
                    if verification_state.latest.success:
                        verification_state.clear_failure()
                    else:
                        verification_state.repair_target = (
                            self.recovery_manager
                            .verification_repair_target_from_result(
                                result,
                                changed_paths=(
                                    self.workspace_tracker.changed_paths
                                    if self.workspace_tracker is not None
                                    else ()
                                ),
                            )
                        )
                        if (
                            verification_state.latest is not None
                            and verification_state.latest.failure_signature
                            and verification_state.latest.failure_signature
                            == verification_state.last_failure_signature
                        ):
                            reason = (
                                'Verification Recovery stopped after the '
                                'same verification failure repeated.'
                            )
                            self.task_manager.stuck((reason,))
                            self.messages[:] = request_messages
                            yield TurnCompleted(
                                result=TurnResult(
                                    text=reason,
                                    usage=completed_usage,
                                    last_request_usage=request_usage,
                                    model_calls=iteration,
                                    tool_calls=tuple(all_tool_calls),
                                    status='stuck',
                                    changed_paths=(
                                        self.workspace_tracker.changed_paths
                                        if self.workspace_tracker is not None
                                        else ()
                                    ),
                                    verification=verification_state.latest,
                                    completion_reasons=(reason,),
                                )
                            )
                            return
                        verification_state.last_failure_signature = (
                            verification_state.latest.failure_signature
                            if verification_state.latest is not None
                            else ''
                        )
                        verification_state.failed_revision = (
                            verification_state.latest.workspace_revision
                            if verification_state.latest is not None
                            else None
                        )
                        verification_state.read_count = 0
                    if verification_state.latest is not None:
                        batch.verification_progressed = True
                        yield VerificationCompleted(
                            evidence=verification_state.latest
                        )
                if tool_call.name == 'task_update' and result.success:
                    batch.task_progressed = True
            request_messages.append(build_tool_result_message(batch.results))

            if batch.terminal_finish_reasons:
                self.task_manager.stuck(batch.terminal_finish_reasons)
                self.messages[:] = request_messages
                yield TurnCompleted(
                    result=TurnResult(
                        text=(
                            'ForgeCode rejected the model completion '
                            'declaration after repeated evidence failures.'
                        ),
                        usage=completed_usage,
                        last_request_usage=request_usage,
                        model_calls=iteration,
                        tool_calls=tuple(all_tool_calls),
                        status='stuck',
                        changed_paths=(
                            self.workspace_tracker.changed_paths
                            if self.workspace_tracker is not None
                            else ()
                        ),
                        verification=verification_state.latest,
                        completion_reasons=batch.terminal_finish_reasons,
                    )
                )
                return

            if batch.accepted_finish is not None:
                declaration_status = str(
                    batch.accepted_finish.metadata['status']
                )
                summary = str(batch.accepted_finish.metadata['summary'])
                blocked_reasons = tuple(
                    str(reason)
                    for reason in batch.accepted_finish.metadata.get(
                        'blocked_reasons',
                        [],
                    )
                )
                if declaration_status == 'blocked':
                    self.task_manager.block(blocked_reasons)
                else:
                    self.task_manager.complete()
                self.messages[:] = request_messages
                self.context.capture_explicit_memory(prompt)
                yield TurnCompleted(
                    result=TurnResult(
                        text=summary,
                        usage=completed_usage,
                        last_request_usage=request_usage,
                        model_calls=iteration,
                        tool_calls=tuple(all_tool_calls),
                        status=(
                            'blocked'
                            if declaration_status == 'blocked'
                            else 'completed'
                        ),
                        changed_paths=(
                            self.workspace_tracker.changed_paths
                            if self.workspace_tracker is not None
                            else ()
                        ),
                        verification=verification_state.latest,
                        completion_reasons=blocked_reasons,
                    )
                )
                return

            workspace_progressed = batch.workspace_progressed
            batch_reverted_to_baseline = (
                workspace_progressed
                and self.workspace_tracker is not None
                and not self.workspace_tracker.changed_paths
            )
            if batch_reverted_to_baseline:
                workspace_progressed = False
            if workspace_progressed:
                actual_workspace_write = any(
                    result.success and changed
                    for _, _, result, changed in batch.workspace_writes
                )
                had_required_verification_repair = (
                    verification_state.requires_repair(runtime.control_state)
                )
                verification_repair_relevant = True
                if (
                    verification_state.requires_repair(runtime.control_state)
                    and self.workspace_tracker is not None
                ):
                    repair_patterns = (
                        (
                            *verification_state.repair_target.paths,
                            *verification_state.repair_target.direct_dependencies,
                        )
                        if verification_state.repair_target is not None
                        else self._static_task_scope_patterns(task_contract)
                    )
                    verification_repair_relevant = evaluate_change_relevance(
                        self.workspace_tracker.changed_paths,
                        TaskScope(patterns=repair_patterns),
                    ).relevant
                verification_repair_progressed = (
                    verification_state.requires_repair(runtime.control_state)
                    and self.workspace_tracker is not None
                    and verification_state.failed_revision is not None
                    and self.workspace_tracker.revision
                    > verification_state.failed_revision
                    and verification_repair_relevant
                )
                edit_recovery.failure_count = 0
                edit_recovery.failures.clear()
                edit_recovery.read_used = False
                edit_recovery.context = ''
                pre_mutation_calls = 0
                action_recovery_state.calls = 0
                action_recovery_state.read_used = False
                if synthesis_state.mode is SynthesisMode.CHECKPOINT:
                    synthesis_state.clear()
                synthesis_retries = 0
                if synthesis_state.mode is SynthesisMode.STAGNATION_FINAL:
                    synthesis_state.clear()
                if (
                    had_required_verification_repair
                    and not verification_repair_progressed
                ):
                    self.agent_controller.enter_fix_required()
                elif (
                    self.agent_controller.state
                    is AgentControlState.TARGETED_ANALYSIS
                    and not actual_workspace_write
                ):
                    pass
                else:
                    verification_state.clear_failure()
                    self.agent_controller.enter_ready_to_verify()
                if not verification_state.recovery_active(
                    runtime.control_state
                ):
                    verification_state.recovery_calls = 0
                completion_ready_revision = None
                completion_decision_calls = 0
                completion_ready_context = ''
                completion_reviewed_paths.clear()
            reviewed_now = completion_review_paths(
                batch.results,
                (
                    self.workspace_tracker.changed_paths
                    if self.workspace_tracker is not None
                    else ()
                ),
            )
            new_reviews = reviewed_now - completion_reviewed_paths
            completion_reviewed_paths.update(reviewed_now)
            pending_write_results = batch.pending_write_results(
                reverted_to_baseline=batch_reverted_to_baseline,
            )
            if pending_write_results:
                edit_recovery.read_used = False
                edit_recovery.failure_count += len(pending_write_results)
                for failed_call, failed_result in pending_write_results:
                    edit_recovery.failures.append(
                        mutation_failure_record(
                            failed_call,
                            failed_result,
                        )
                    )
                edit_recovery.failures = edit_recovery.failures[-3:]
            if edit_recovery.failures:
                self.agent_controller.enter_fix_required()
                action_recovery_state.calls = 0
                action_recovery_state.read_used = False
                edit_recovery.context = (
                    render_mutation_recovery_context(
                        edit_recovery.failures,
                        edit_recovery.failure_count,
                    )
                )
                if batch.workspace_writes:
                    request_messages.append(
                        build_mutation_recovery_feedback(
                            edit_recovery.failures,
                            edit_recovery.failure_count,
                            self.task_manager.system_suffix(),
                        )
                    )
                if (
                    edit_recovery.failure_count
                    >= self.mutation_recovery_limit
                ):
                    reason = mutation_recovery_stuck_reason(
                        edit_recovery.failures,
                        edit_recovery.failure_count,
                    )
                    self.task_manager.stuck((reason,))
                    self.messages[:] = request_messages
                    yield TurnCompleted(
                        result=TurnResult(
                            text=reason,
                            usage=completed_usage,
                            last_request_usage=request_usage,
                            model_calls=iteration,
                            tool_calls=tuple(all_tool_calls),
                            status='stuck',
                            changed_paths=(
                                self.workspace_tracker.changed_paths
                                if self.workspace_tracker is not None
                                else ()
                            ),
                            verification=verification_state.latest,
                            completion_reasons=(reason,),
                        )
                    )
                    return
            protocol_failure = bool(batch.results) and all(
                is_tool_protocol_failure(result)
                for _, result in batch.results
            )
            if protocol_failure:
                tool_protocol_failures += 1
            elif any(result.success for _, result in batch.results):
                tool_protocol_failures = 0
            if batch_requires_todo(batch.results):
                self.agent_controller.enter_planning_recovery()
                if synthesis_state.mode is SynthesisMode.CHECKPOINT:
                    synthesis_state.clear()
                synthesis_retries = 0
                calls_without_progress = 0
                action_recovery_state.calls = 0
                action_recovery_state.read_used = False
                request_messages.append(
                    build_planning_recovery_feedback(
                        self.task_manager.system_suffix(),
                        attempt=(
                            self.agent_controller.planning_recovery_calls
                        ),
                    )
                )
                continue
            pending_required_change = self._pending_required_change(
                change_required,
                mutation_attempted=mutation_attempted,
            )
            if (
                pending_required_change
                and not edit_recovery.failures
                and not protocol_failure
            ):
                entered_action_recovery = False
                if self.agent_controller.action_recovery:
                    action_recovery_state.calls += 1
                elif batch.required_change_rejected or batch_reverted_to_baseline:
                    self.agent_controller.enter_targeted_analysis()
                    action_recovery_state.calls = 0
                    action_recovery_state.read_used = False
                    entered_action_recovery = True
                elif batch.task_progressed:
                    pre_mutation_calls = 0
                else:
                    pre_mutation_calls += 1
                    if (
                        pre_mutation_calls > self.pre_mutation_limit
                        and _can_enter_pre_mutation_action_recovery(
                            task_contract
                        )
                    ):
                        self.agent_controller.enter_targeted_analysis()
                        action_recovery_state.calls = 0
                        action_recovery_state.read_used = False
                        entered_action_recovery = True
                if self.agent_controller.action_recovery:
                    if synthesis_state.mode is SynthesisMode.CHECKPOINT:
                        synthesis_state.clear()
                    synthesis_retries = 0
                    if (
                        synthesis_state.mode
                        is SynthesisMode.STAGNATION_FINAL
                    ):
                        synthesis_state.clear()
                    calls_without_progress = 0
                    if entered_action_recovery:
                        action_recovery_state.block_events += 1
                        yield CompletionBlocked(
                            attempt=action_recovery_state.block_events,
                            reasons=(required_change_block_reason(),),
                        )
                    if (
                        action_recovery_state.calls
                        >= self.action_recovery_limit
                    ):
                        reason = action_recovery_stuck_reason(
                            action_recovery_state.calls
                        )
                        change_reason = required_change_block_reason()
                        self.task_manager.stuck((reason, change_reason))
                        self.messages[:] = request_messages
                        yield TurnCompleted(
                            result=TurnResult(
                                text=reason,
                                usage=completed_usage,
                                last_request_usage=request_usage,
                                model_calls=iteration,
                                tool_calls=tuple(all_tool_calls),
                                status='stuck',
                                changed_paths=(),
                                verification=verification_state.latest,
                                completion_reasons=(
                                    reason,
                                    change_reason,
                                ),
                            )
                        )
                        return
                    request_messages.append(
                        build_action_recovery_feedback(
                            self.task_manager.system_suffix(),
                            action_recovery_state.calls,
                            self.action_recovery_limit,
                            read_used=action_recovery_state.read_used,
                        )
                    )
                    continue
            completion_ready = (
                not protocol_failure
                and await self.completion_checker.can_finalize_after_stagnation(
                    mutation_attempted=mutation_attempted,
                    verification=verification_state.latest,
                    mutation_failures=edit_recovery.failures,
                    reviewed_paths=completion_reviewed_paths,
                    evidence_paths=self.working_state.evidence_paths,
                )
            )
            if completion_ready:
                if self.workspace_tracker is None:
                    raise AssertionError(
                        'Completion readiness requires a workspace tracker.'
                    )
                revision = self.workspace_tracker.revision
                new_ready_revision = completion_ready_revision != revision
                if new_ready_revision:
                    completion_ready_revision = revision
                    completion_decision_calls = 0
                    completion_reviewed_paths.clear()
                    if synthesis_state.mode is SynthesisMode.CHECKPOINT:
                        synthesis_state.clear()
                    synthesis_retries = 0
                    if (
                        synthesis_state.mode
                        is SynthesisMode.STAGNATION_FINAL
                    ):
                        synthesis_state.clear()
                if not new_ready_revision and not new_reviews:
                    completion_decision_calls += 1
                completion_ready_context = render_completion_ready_context(
                    self.workspace_tracker.changed_paths,
                    verification_state.latest,
                    completion_decision_calls,
                    self.completion_decision_limit,
                    completion_reviewed_paths,
                )
                calls_without_progress = 0
                synthesis_state.mode = SynthesisMode.FINALIZATION
                request_messages.append(
                    build_finalization_recovery_feedback(
                        self.task_manager.system_suffix(),
                        self.working_state.system_suffix(),
                        self.workspace_tracker.changed_paths,
                        verification_state.latest,
                    )
                )
                continue
            completion_ready_revision = None
            completion_decision_calls = 0
            completion_ready_context = ''
            completion_reviewed_paths.clear()
            progress = evaluate_progress(
                workspace_progressed=workspace_progressed,
                task_progressed=batch.task_progressed,
                evidence_progressed=batch.evidence_progressed,
                verification_progressed=batch.verification_progressed,
                review_progressed=bool(new_reviews),
                protocol_failure=protocol_failure,
                mutation_recovery_active=bool(edit_recovery.failures),
                requires_change=change_required,
                task_scope_patterns=(
                    self._static_task_scope_patterns(task_contract)
                ),
                changed_paths=(
                    self.workspace_tracker.changed_paths
                    if self.workspace_tracker is not None
                    else ()
                ),
                evidence_paths=tuple(
                    sorted(
                        set(self.working_state.evidence_paths)
                        - evidence_paths_before_batch
                    )
                ),
                review_paths=tuple(sorted(new_reviews)),
                repair_target_paths=(
                    verification_state.repair_target.paths
                    if verification_state.repair_target is not None
                    else ()
                ),
            )
            if progress.progressed:
                calls_without_progress = 0
                if synthesis_state.mode is SynthesisMode.CHECKPOINT:
                    synthesis_state.clear()
                synthesis_retries = 0
            elif protocol_failure:
                # Malformed tool arguments are a protocol-recovery problem,
                # not evidence that the task itself is stuck.
                request_messages.append(
                    build_tool_protocol_feedback(
                        tool_protocol_failures,
                        self.task_manager.system_suffix(),
                        batch.results,
                    )
                )
                if (
                    tool_protocol_failures
                    >= self.max_tool_protocol_recoveries
                ):
                    reason = (
                        'Stopped after repeated malformed or schema-invalid '
                        'tool requests. The repository task may still be '
                        'solvable, but this agent trajectory is stuck.'
                    )
                    self.task_manager.stuck((reason,))
                    self.messages[:] = request_messages
                    yield TurnCompleted(
                        result=TurnResult(
                            text=reason,
                            usage=completed_usage,
                            last_request_usage=request_usage,
                            model_calls=iteration,
                            tool_calls=tuple(all_tool_calls),
                            status='stuck',
                            changed_paths=(
                                self.workspace_tracker.changed_paths
                                if self.workspace_tracker is not None
                                else ()
                            ),
                            verification=verification_state.latest,
                            completion_reasons=(reason,),
                        )
                    )
                    return
            elif edit_recovery.failures:
                # Edit Recovery exclusively owns progress limits while a
                # workspace-write failure remains unresolved. Reads and
                # searches may guide the corrected edit without also
                # consuming the global Stagnation budget.
                calls_without_progress = 0
            else:
                calls_without_progress += 1
            if calls_without_progress == self.stagnation_warning:
                if synthesis_state.mode is SynthesisMode.NORMAL:
                    synthesis_state.mode = SynthesisMode.CHECKPOINT
                request_messages.append(
                    build_stagnation_feedback(
                        calls_without_progress,
                        self.task_manager.system_suffix(),
                        self.working_state.system_suffix(),
                    )
                )
            elif calls_without_progress >= self.stagnation_limit:
                if (
                    not edit_recovery.failures
                    and self._pending_required_change(
                        change_required,
                        mutation_attempted=mutation_attempted,
                    )
                    and _can_enter_pre_mutation_action_recovery(
                        task_contract
                    )
                ):
                    self.agent_controller.enter_targeted_analysis()
                    action_recovery_state.calls = 0
                    action_recovery_state.read_used = False
                    if synthesis_state.mode is SynthesisMode.CHECKPOINT:
                        synthesis_state.clear()
                    synthesis_retries = 0
                    calls_without_progress = 0
                    action_recovery_state.block_events += 1
                    yield CompletionBlocked(
                        attempt=action_recovery_state.block_events,
                        reasons=(required_change_block_reason(),),
                    )
                    request_messages.append(
                        build_action_recovery_feedback(
                            self.task_manager.system_suffix(),
                            action_recovery_state.calls,
                            self.action_recovery_limit,
                            read_used=action_recovery_state.read_used,
                        )
                    )
                    continue
                if await self.completion_checker.can_finalize_after_stagnation(
                    mutation_attempted=mutation_attempted,
                    verification=verification_state.latest,
                    mutation_failures=edit_recovery.failures,
                    evidence_paths=self.working_state.evidence_paths,
                ):
                    synthesis_state.mode = SynthesisMode.FINALIZATION
                    request_messages.append(
                        build_finalization_recovery_feedback(
                            self.task_manager.system_suffix(),
                            self.working_state.system_suffix(),
                            self.workspace_tracker.changed_paths,
                            verification_state.latest,
                        )
                    )
                    continue
                if (
                    self.workspace_tracker is not None
                    and self.workspace_tracker.changed_paths
                ):
                    decision = await self.completion_checker.evaluate(
                        verification_state.latest,
                        mutation_attempted=mutation_attempted,
                        reviewed_paths=completion_reviewed_paths,
                        evidence_paths=self.working_state.evidence_paths,
                    )
                    if not decision.allowed:
                        last_completion_reasons = decision.reasons
                        completion_blocks += 1
                        yield CompletionBlocked(
                            attempt=completion_blocks,
                            reasons=decision.reasons,
                        )
                        if completion_blocks >= self.max_completion_blocks:
                            if only_verification_blocked(decision.reasons):
                                verification_state.recovery_calls += 1
                                if (
                                    verification_state.recovery_calls
                                    > self.action_recovery_limit
                                ):
                                    reason = (
                                        'ForgeCode stopped after repeated '
                                        'verification recovery requests did '
                                        'not produce current verify evidence.'
                                    )
                                    reasons = (reason, *decision.reasons)
                                    self.task_manager.stuck(reasons)
                                    self.messages[:] = request_messages
                                    yield TurnCompleted(
                                        result=TurnResult(
                                            text=reason,
                                            usage=completed_usage,
                                            last_request_usage=request_usage,
                                            model_calls=iteration,
                                            tool_calls=tuple(all_tool_calls),
                                            status='stuck',
                                            changed_paths=(
                                                self.workspace_tracker.changed_paths
                                            ),
                                            verification=verification_state.latest,
                                            completion_reasons=reasons,
                                        )
                                    )
                                    return
                                verification_failure_blocked = any(
                                    marker in reason
                                    for marker in (
                                        'latest verification failed',
                                        'latest verification timed out',
                                        'latest verification command was invalid',
                                    )
                                    for reason in decision.reasons
                                )
                                verification_needs_repair = (
                                    verification_failure_blocked
                                    and verification_state.latest is not None
                                    and not verification_state.latest.success
                                )
                                verification_state.failed_revision = (
                                    verification_state.latest.workspace_revision
                                    if verification_needs_repair
                                    and verification_state.latest is not None
                                    else None
                                )
                                if verification_needs_repair:
                                    self.agent_controller.enter_fix_required()
                                else:
                                    self.agent_controller.enter_ready_to_verify()
                                calls_without_progress = 0
                                if (
                                    synthesis_state.mode
                                    is SynthesisMode.CHECKPOINT
                                ):
                                    synthesis_state.clear()
                                synthesis_retries = 0
                                if (
                                    synthesis_state.mode
                                    is SynthesisMode.STAGNATION_FINAL
                                ):
                                    synthesis_state.clear()
                                request_messages.append(
                                    build_completion_feedback(
                                        decision.reasons,
                                        task_context=(
                                            self.task_manager.system_suffix()
                                        ),
                                    )
                                )
                                continue
                            reason = (
                                'ForgeCode stopped after repeated recovery '
                                'requests did not satisfy the current '
                                'completion checks.'
                            )
                            reasons = (reason, *decision.reasons)
                            self.task_manager.stuck(reasons)
                            self.messages[:] = request_messages
                            yield TurnCompleted(
                                result=TurnResult(
                                    text=reason,
                                    usage=completed_usage,
                                    last_request_usage=request_usage,
                                    model_calls=iteration,
                                    tool_calls=tuple(all_tool_calls),
                                    status='stuck',
                                    changed_paths=(
                                        self.workspace_tracker.changed_paths
                                    ),
                                    verification=verification_state.latest,
                                    completion_reasons=reasons,
                                )
                            )
                            return
                        calls_without_progress = 0
                        if synthesis_state.mode is SynthesisMode.CHECKPOINT:
                            synthesis_state.clear()
                        synthesis_retries = 0
                        if (
                            synthesis_state.mode
                            is SynthesisMode.STAGNATION_FINAL
                        ):
                            synthesis_state.clear()
                        request_messages.append(
                            build_completion_feedback(
                                decision.reasons,
                                task_context=(
                                    self.task_manager.system_suffix()
                                ),
                            )
                        )
                        continue
                if synthesis_state.mode is not SynthesisMode.STAGNATION_FINAL:
                    synthesis_state.mode = SynthesisMode.STAGNATION_FINAL
                    request_messages.append(
                        build_stagnation_final_recovery_feedback(
                            self.task_manager.system_suffix(),
                            self.working_state.system_suffix(),
                            calls_without_progress,
                        )
                    )
                    continue
                reason = (
                    'Stopped after '
                    f'{calls_without_progress} model calls without new '
                    'workspace, plan, or repository evidence.'
                )
                self.task_manager.stuck((reason,))
                self.messages[:] = request_messages
                yield TurnCompleted(
                    result=TurnResult(
                        text=reason,
                        usage=completed_usage,
                        last_request_usage=request_usage,
                        model_calls=iteration,
                        tool_calls=tuple(all_tool_calls),
                        status='stuck',
                        changed_paths=(
                            self.workspace_tracker.changed_paths
                            if self.workspace_tracker is not None
                            else ()
                        ),
                        verification=verification_state.latest,
                        completion_reasons=(reason,),
                    )
                )
                return

        if self.max_iterations is not None:
            raise AgentLoopLimitError(
                f'Agent Loop exceeded {self.max_iterations} model calls.'
            )
        raise AssertionError('Unlimited Agent Loop stopped unexpectedly.')

    def _system_prompt_with_task(
        self,
        *,
        include_tool_availability: bool = True,
    ) -> str:
        task_context = self.task_manager.system_suffix()
        self._last_task_context = task_context
        parts = [self.system_prompt]
        if task_context:
            parts.append(task_context)
        working_context = self.working_state.system_suffix()
        if working_context:
            parts.append(working_context)
        if self.tools and include_tool_availability:
            parts.append(
                '[Runtime Tool Availability]\n'
                'The tools included with this model request are currently '
                'available. Decide from the user goal whether to answer, '
                'inspect, modify, or verify. If earlier conversation text '
                'claimed tools were unavailable, that claim is stale for '
                'this request. Use tools directly whenever your chosen '
                'approach requires repository actions.'
            )
        parts.append(render_interaction_mode_context(self.interaction_mode))
        return '\n\n'.join(parts)

    def _initial_change_required(self, prompt: str) -> bool:
        return self._initial_task_contract(prompt).requires_change

    def _initial_task_contract(self, prompt: str) -> TaskContract:
        return infer_task_contract(
            prompt,
            interaction_mode=self.interaction_mode,
            workspace_available=self.workspace_tracker is not None,
            policy_requires_change=self.completion_checker.requires_changes,
        )

    async def _resolved_task_contract(self, prompt: str) -> TaskContract:
        contract = self._initial_task_contract(prompt)
        return await refine_task_contract_async(
            prompt,
            contract,
            self.intent_classifier,
        )

    def _plan_mode_tools(self) -> list[dict[str, Any]] | None:
        if self.tools is None:
            return None
        selected: list[dict[str, Any]] = []
        for definition in self.tools:
            name = str(definition.get('name', ''))
            if name in PLAN_MODE_TOOLS:
                selected.append(definition)
                continue
            if (
                name.startswith('mcp_')
                and self.registry is not None
                and self.registry.effect(name) == 'read_only'
            ):
                selected.append(definition)
        return selected

    def _pending_required_change(
        self,
        change_required: bool,
        *,
        mutation_attempted: bool,
    ) -> bool:
        tracker = self.workspace_tracker
        return bool(
            change_required
            and tracker is not None
            and getattr(tracker, 'available', True)
            and (not mutation_attempted or not tracker.changed_paths)
        )

    def _static_task_scope_patterns(
        self,
        contract: TaskContract,
    ) -> tuple[str, ...]:
        patterns = self.completion_checker.task_scope_patterns(
            evidence_paths=(),
        )
        if patterns:
            return patterns
        return infer_task_scope(
            contract.goal,
            scope_hints=contract.allowed_paths,
        ).patterns

    async def compact(self) -> CompactionReport:
        '''Manually summarize committed history for the /compact command.'''
        if not self.messages:
            return CompactionReport(
                success=True,
                automatic=False,
                before_characters=0,
                after_characters=0,
                transcript_path=None,
                reason='conversation history is empty',
            )
        report = await self.context.compact_history(
            self.messages,
            self.client,
            force=True,
        )
        if report is None:
            raise AssertionError('Forced compaction did not return a report.')
        return report

    def remember(self, name: str, content: str) -> str:
        record = self.context.remember(name, content)
        return f'Remembered {record.name} in {record.path.as_posix()}'

    def memory_list(self) -> str:
        records = self.context.repository.memory.list()
        if not records:
            return 'No repository memories.'
        return '\n'.join(
            f'- {record.name} [{record.memory_type}]: {record.description}'
            for record in records
        )

    def memory_show(self, name: str) -> str:
        record = self.context.repository.memory.get(name)
        if record is None:
            return f'Memory not found: {name}'
        return (
            f'{record.name} [{record.memory_type}]\n'
            f'source: {record.source}\n'
            f'created_at: {record.created_at}\n'
            f'updated_at: {record.updated_at}\n'
            f'{record.description}\n\n{record.content}'
        )

    def memory_forget(self, name: str) -> str:
        removed = self.context.repository.memory.forget(name)
        return f'Forgot {name}.' if removed else f'Memory not found: {name}'

    def memory_rebuild(self) -> str:
        path = self.context.repository.memory.rebuild_index()
        return f'Rebuilt memory index: {path.as_posix()}'

    def memory_consolidate(self) -> str:
        removed = self.context.repository.memory.consolidate()
        return f'Consolidated memory; removed {removed} duplicate(s).'

    def task_show(self) -> str:
        return self.task_manager.describe()

    def task_history(self) -> str:
        return self.task_manager.history()

    def task_resume(self, task_id: str) -> str:
        task = self.task_manager.resume(task_id)
        self._last_task_context = self.task_manager.system_suffix()
        return f'Resumed {task.id}: {task.goal}'

    def mcp_status(self) -> str:
        tool_names: tuple[str, ...]
        if self.registry is not None:
            tool_names = tuple(
                name for name in self.registry.names if name.startswith('mcp_')
            )
        else:
            tool_names = tuple(
                str(definition.get('name', ''))
                for definition in (self.tools or [])
                if str(definition.get('name', '')).startswith('mcp_')
            )
        return render_mcp_status(self.task_manager.root, tool_names)

    def hooks_status(self) -> str:
        return self.hook_registry.describe()

    def todo_status(self) -> str:
        return self.todo_list.render()

    def mode_show(self) -> str:
        return render_mode_notice(self.interaction_mode)

    def mode_set(self, mode: str) -> str:
        normalized = normalize_interaction_mode(mode)
        self.interaction_mode = normalized
        return render_mode_notice(normalized)

    def permission_show(self) -> str:
        return render_permission_notice(self.permission.mode)

    def permission_set(self, mode: str) -> str:
        normalized = normalize_permission_mode(mode)
        self.permission.set_mode(normalized)
        return render_permission_notice(normalized)

    def model_show(self) -> str:
        current = getattr(self.client, 'model', 'configured model')
        choices = ', '.join(SUPPORTED_MODEL_IDS)
        return f'Model: {current}.\nAvailable: {choices}.'

    def model_set(self, model_id: str) -> str:
        normalized = normalize_supported_model_id(model_id)
        config_path = update_user_model_id(normalized)
        if isinstance(self.client, AnthropicModelClient):
            current = ForgeConfig.from_env(cwd=self.config_root)
            config = ForgeConfig(
                api_key=current.api_key,
                model_id=normalized,
                base_url=current.base_url,
                max_tokens=current.max_tokens,
                context_window=current.context_window,
                request_timeout_seconds=current.request_timeout_seconds,
            )
            self.client = AnthropicModelClient.from_config(config=config)
        elif hasattr(self.client, 'model'):
            setattr(self.client, 'model', normalized)
        self.model_runner = ModelRunner(self.client)
        if not self._intent_classifier_overridden:
            self.intent_classifier = (
                None
                if getattr(self.client, 'provider', '') == 'fake'
                else ModelSemanticTaskClassifier(self.client)
            )
        return (
            f'Model: {normalized}.\n'
            f'Global default updated: {config_path}'
        )

    def set_permission_approver(self, approver: Any | None) -> None:
        self.permission.approver = approver

    def enable_rollout_persistence(self) -> None:
        '''Persist replayable session state after every runtime event.'''
        self._rollout_enabled = True

    async def run_stop_hooks(self, result: TurnResult) -> None:
        await self.hook_registry.run(
            HookContext(
                event='stop',
                root=self.task_manager.root,
                turn_result=result,
                permission_mode=self.permission.mode,
            )
        )

    def save_session(self) -> str:
        snapshot = self.session_manager.save(
            self.messages,
            session_id=self.session_id,
            active_task=self.task_manager.active,
            interaction_mode=self.interaction_mode,
            permission_mode=self.permission.mode,
        )
        self.session_id = snapshot.id
        return snapshot.id

    def resume_session(self, session_id: str | None = None) -> str:
        snapshot = self.session_manager.load(session_id)
        warnings = self.session_manager.consistency_warnings(snapshot)
        self._apply_session_snapshot(snapshot)
        notice = (
            f'Resumed {snapshot.id}: '
            f'{len(snapshot.messages)} message(s), updated {snapshot.updated_at}'
        )
        if warnings:
            notice += '\nWorkspace warning: ' + '; '.join(warnings)
        return notice

    def fork_session(self, session_id: str | None = None) -> str:
        snapshot = self.session_manager.fork(session_id)
        self._apply_session_snapshot(snapshot)
        return (
            f'Forked {snapshot.parent_session_id} into {snapshot.id}: '
            f'{len(snapshot.messages)} message(s)'
        )

    def _apply_session_snapshot(self, snapshot: Any) -> None:
        self.messages[:] = snapshot.messages
        self.session_id = snapshot.id
        self.task_manager.active = snapshot.active_task
        self.interaction_mode = normalize_interaction_mode(
            snapshot.interaction_mode
        )
        self.permission.set_mode(
            normalize_permission_mode(snapshot.permission_mode)
        )
        self._last_task_context = self.task_manager.system_suffix()
        self._last_repository_context = self.context.repository.system_suffix('')

    def _persist_rollout_event(self, event: ConversationEvent) -> None:
        if not self._rollout_enabled or isinstance(
            event,
            (
                ModelTextDelta,
                ModelToolCallArgumentsDelta,
                ModelUsageUpdate,
            ),
        ):
            return
        snapshot = self.session_manager.record_event(
            event,
            self._inflight_messages or self.messages,
            session_id=self.session_id,
            active_task=self.task_manager.active,
            interaction_mode=self.interaction_mode,
            permission_mode=self.permission.mode,
            update_workspace=isinstance(
                event,
                (WorkspaceChanged, TurnCompleted),
            ),
        )
        self.session_id = snapshot.id

    def _persist_rollout_interruption(self, error: Exception) -> None:
        if not self._rollout_enabled:
            return
        try:
            self.session_manager.store.record_event(
                TurnInterrupted(
                    error_type=type(error).__name__,
                    message=str(error),
                ),
                self._inflight_messages or self.messages,
                session_id=self.session_id,
                active_task=self.task_manager.active,
                interaction_mode=self.interaction_mode,
                permission_mode=self.permission.mode,
                update_workspace=True,
            )
        except OSError:
            pass

    def session_history(self) -> str:
        return self.session_manager.history()

    def session_choices(self) -> tuple[tuple[str, str, str], ...]:
        return self.session_manager.choices()

    def subagent_worktrees(self) -> str:
        from forge.runtime.worktree import SubagentWorktreeManager

        return SubagentWorktreeManager(self.task_manager.root).describe()


def tool_call_signature(tool_call: ToolCall, revision: int) -> str:
    '''Identify an exact tool request within one workspace revision.'''
    arguments = json.dumps(
        normalize_tool_arguments(tool_call.name, tool_call.arguments),
        ensure_ascii=False,
        sort_keys=True,
        separators=(',', ':'),
        default=str,
    )
    digest = hashlib.sha256(arguments.encode('utf-8')).hexdigest()[:24]
    return f'{revision}:{tool_call.name}:{digest}'


def early_mutation_relevance_failure(
    tool_call: ToolCall,
    *,
    tool_effect: str | None,
    change_required: bool,
    task_scope_patterns: tuple[str, ...],
) -> ToolResult | None:
    '''Block statically obvious off-goal workspace edits before execution.'''
    if (
        not change_required
        or tool_effect != 'workspace_write'
        or not task_scope_patterns
    ):
        return None
    targets = mutation_target_paths(tool_call)
    if not targets:
        return None
    relevance_targets = _scope_probe_paths(targets)
    relevance = evaluate_change_relevance(
        relevance_targets,
        TaskScope(patterns=task_scope_patterns),
    )
    if relevance.relevant:
        return None
    return ToolResult.fail(
        'irrelevant_mutation_target',
        (
            f'{tool_call.name} targets paths outside the current task scope: '
            + ', '.join(targets)
            + '. Choose a task-relevant edit target instead.'
        ),
        details={
            'targets': list(targets),
            'task_scope_patterns': list(task_scope_patterns[:16]),
            'reasons': list(relevance.reasons),
        },
        metadata={'irrelevant_mutation_target': True},
    )


def _can_enter_pre_mutation_action_recovery(
    contract: TaskContract,
) -> bool:
    if contract.kind != 'implement':
        return True
    return bool(
        contract.allowed_paths
        or contract.context_hints
        or '.' in contract.goal
    )


def _scope_probe_paths(paths: tuple[str, ...]) -> tuple[str, ...]:
    expanded: list[str] = []
    for path in paths:
        normalized = path.replace('\\', '/').rstrip('/')
        expanded.append(normalized)
        name = normalized.rsplit('/', 1)[-1]
        if '.' not in name:
            expanded.append(f'{normalized}/__forge_scope_probe__')
    return tuple(dict.fromkeys(expanded))


def normalize_tool_arguments(
    tool_name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    normalized = dict(arguments)
    defaults: dict[str, Any] = {}
    if tool_name == 'read_file':
        defaults = {'start_line': 1, 'end_line': None}
    elif tool_name == 'list_directory':
        defaults = {'path': '.', 'max_results': 1000}
    elif tool_name == 'find_files':
        defaults = {'path': '.', 'max_results': 200}
    elif tool_name == 'grep':
        defaults = {
            'path': '.',
            'file_types': [],
            'case_sensitive': True,
            'regex': True,
            'max_results': 200,
        }
    elif tool_name == 'verify':
        defaults = {
            'target': 'auto',
            'command_id': '',
            'command': '',
            'cwd': '.',
            'timeout_seconds': 120.0,
        }
    elif tool_name == 'git_diff':
        defaults = {'path': None, 'cached': False}

    for key, value in defaults.items():
        normalized.setdefault(key, value)
    for key in ('path', 'cwd'):
        if key in normalized and normalized[key] is not None:
            normalized[key] = normalize_signature_path(str(normalized[key]))
    if tool_name == 'grep' and isinstance(normalized.get('file_types'), list):
        normalized['file_types'] = sorted(
            normalize_file_type(str(item))
            for item in normalized['file_types']
        )
    return normalized


def normalize_signature_path(path: str) -> str:
    rendered = path.strip().replace('\\', '/')
    while rendered.startswith('./'):
        rendered = rendered[2:]
    rendered = rendered.rstrip('/')
    return rendered or '.'


def normalize_file_type(value: str) -> str:
    lowered = value.strip().casefold()
    if not lowered:
        return lowered
    return lowered if lowered.startswith('.') else f'.{lowered}'


def required_change_block_reason() -> str:
    return (
        'This turn requires a real task-local workspace change, but no file '
        'differs from the workspace snapshot captured when the turn began.'
    )


def render_mcp_status(root: Path, tool_names: tuple[str, ...]) -> str:
    sources = mcp_config_sources(
        root,
        home=forge_home(),
        app_root=forge_app_root(),
    )
    existing = [
        (path, cwd) for path, cwd in sources if path.is_file()
    ]
    if not existing:
        return (
            'Config sources:\n'
            + ''.join(f'- {path.as_posix()}\n' for path, _ in sources)
            +
            'Status: no MCP config file found.\n'
            'Servers: 0\n'
            f'Tools: {len(tool_names)}'
        )
    configs_by_name = {}
    try:
        for path, cwd in existing:
            data = json.loads(path.read_text(encoding='utf-8'))
            for config in parse_mcp_config(data, cwd):
                configs_by_name[config.name] = config
    except (
        OSError,
        json.JSONDecodeError,
        MCPConfigurationError,
    ) as error:
        return (
            'Config sources:\n'
            + ''.join(f'- {path.as_posix()}\n' for path, _ in sources)
            +
            'Status: invalid MCP config.\n'
            f'Error: {error}\n'
            f'Tools registered before error: {len(tool_names)}'
        )

    lines = [
        'Config sources:',
        *[
            f'- {path.as_posix()}'
            for path, _ in sources
        ],
        'Status: configured',
        f'Servers: {len(configs_by_name)}',
    ]
    for config in configs_by_name.values():
        command = ' '.join((config.command, *config.args)).strip()
        lines.append(f'- {config.name}: stdio `{command}`')
    lines.append(f'Tools: {len(tool_names)}')
    lines.extend(f'- {name}' for name in tool_names)
    return '\n'.join(lines)


def normalize_interaction_mode(mode: str) -> InteractionMode:
    normalized = mode.strip().casefold()
    if normalized not in {'auto', 'plan', 'code'}:
        raise ValueError('Mode must be one of: auto, plan, code.')
    return normalized  # type: ignore[return-value]


def render_mode_notice(mode: InteractionMode) -> str:
    if mode == 'auto':
        return (
            'Mode: auto. ForgeCode infers whether a turn needs edits; '
            'planning, checklist, suggestion, and analysis requests do not '
            'require a workspace Diff.'
        )
    if mode == 'plan':
        return (
            'Mode: plan. ForgeCode will only use read-only planning tools and '
            'will not require or perform workspace edits. Switch to /code '
            'when you want the plan implemented.'
        )
    return (
        'Mode: code. ForgeCode treats user turns as authorized implementation '
        'work and requires a real workspace Diff before completion.'
    )


def render_interaction_mode_context(mode: InteractionMode) -> str:
    if mode == 'plan':
        return (
            '[ForgeCode Interaction Mode]\n'
            'Mode: plan. The user wants planning, analysis, or a repair '
            'checklist only. Do not modify files. Only read-only planning '
            'tools are available. A final answer should present a clear plan '
            'or prioritized checklist and mention that the user can switch to '
            '/code to implement it. No workspace Diff is required.'
        )
    if mode == 'code':
        return (
            '[ForgeCode Interaction Mode]\n'
            'Mode: code. The user has authorized implementation. Make the '
            'necessary workspace edits instead of stopping at a plan. After '
            'modifying files, run or recommend an appropriate verification; '
            'a real workspace Diff is required before completion.'
        )
    return (
        '[ForgeCode Interaction Mode]\n'
        'Mode: auto. Infer whether the user wants an answer/plan or actual '
        'implementation. Planning, checklist, suggestion, analysis, and '
        'proposal requests should be answered without forcing a workspace '
        'Diff. Only require edits for high-confidence implementation asks.'
    )



