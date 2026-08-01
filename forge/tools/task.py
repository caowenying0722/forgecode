'''Optional planning tools backed by the current TaskManager.'''

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from forge.runtime.acceptance import (
    AcceptanceEvidence,
    AcceptanceLedger,
    evidence_from_payload,
)
from forge.runtime.task_scope import scope_patterns_from_hints
from forge.tasks.manager import ExistingPlanError, TaskManager
from forge.tasks.state import TaskSpecDigest
from forge.tools.base import Tool, ToolExecutionError, ToolInput, ToolResult


class TaskGetInput(ToolInput):
    pass


class TaskGetTool(Tool[TaskGetInput]):
    name = 'task_get'
    description = (
        'Return the current ForgeCode task and optional plan. Use only when '
        'you need to inspect current active-goal state; the current goal is '
        'already injected into every model request. This is not the durable '
        'project task graph.'
    )
    input_model = TaskGetInput

    def __init__(self, root: Path, manager: TaskManager) -> None:
        super().__init__(root)
        self.manager = manager

    async def execute(self, arguments: TaskGetInput) -> ToolResult:
        del arguments
        return ToolResult.ok(
            'Read the current task.',
            content=self.manager.describe(),
        )


class TaskPlanStepInput(BaseModel):
    model_config = ConfigDict(extra='forbid')

    title: str = Field(min_length=1, max_length=1_000)
    deliverables: list[str] = Field(default_factory=list, max_length=20)
    criterion_ids: list[str] = Field(default_factory=list, max_length=20)


class SourceSectionInput(BaseModel):
    model_config = ConfigDict(extra='forbid')

    path: str = Field(min_length=1, max_length=1_000)
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    title: str = Field(default='', max_length=1_000)

    @model_validator(mode='after')
    def validate_line_range(self) -> SourceSectionInput:
        if self.end_line < self.start_line:
            raise ValueError('end_line must be greater than or equal to start_line.')
        return self


class TaskSpecDigestInput(BaseModel):
    model_config = ConfigDict(extra='forbid')

    source_paths: list[str] = Field(min_length=1, max_length=20)
    goal: str = Field(min_length=1, max_length=20_000)
    requirements: list[str] = Field(default_factory=list, max_length=100)
    acceptance_criteria: list[str] = Field(default_factory=list, max_length=100)
    required_commands: list[str] = Field(default_factory=list, max_length=50)
    required_modules: list[str] = Field(default_factory=list, max_length=50)
    forbidden_changes: list[str] = Field(default_factory=list, max_length=50)
    relevant_sections: list[SourceSectionInput] = Field(
        default_factory=list,
        max_length=50,
    )


class TaskPlanInput(ToolInput):
    steps: list[str | TaskPlanStepInput] = Field(min_length=1, max_length=20)
    constraints: list[str] = Field(default_factory=list, max_length=20)
    scope_hints: list[str] = Field(
        default_factory=list,
        max_length=20,
        description=(
            'Optional workspace-relative path or glob hints only. Examples: '
            'src/**, tests/**, package.json. Do not put prose or task '
            'descriptions here.'
        ),
    )
    replace: bool = False
    operation: Literal['create', 'append', 'replace'] = 'create'
    task_spec: TaskSpecDigestInput | None = None

    @model_validator(mode='after')
    def validate_step_count(self) -> TaskPlanInput:
        operation = 'replace' if self.replace else self.operation
        if operation != 'append' and len(self.steps) < 2:
            raise ValueError(
                'A created or replaced plan requires at least two steps.'
            )
        return self


class TaskPlanTool(Tool[TaskPlanInput]):
    name = 'task_plan'
    description = (
        'Create one active-goal linear plan for complex work with multiple '
        'dependent steps, multiple files, or implementation plus verification '
        'inside the current conversation. Do not use for questions, directory '
        'listings, one command, one file read, or a small focused edit. Do '
        'not use this for durable project task queues; use task-graph tools '
        'only when persistent dependency tracking is explicitly needed. A '
        'For an existing plan use operation="append", task_update to update '
        'or complete a step, or operation="replace" (replace=true remains a '
        'compatibility alias).'
    )
    input_model = TaskPlanInput

    def __init__(self, root: Path, manager: TaskManager) -> None:
        super().__init__(root)
        self.manager = manager

    async def execute(self, arguments: TaskPlanInput) -> ToolResult:
        titles, step_deliverables, step_criterion_ids = _plan_step_links(
            arguments.steps
        )
        scope_patterns = scope_patterns_from_hints(tuple(arguments.scope_hints))
        ignored_scope_hints = [
            value
            for value in arguments.scope_hints
            if not scope_patterns_from_hints((value,))
        ]
        operation = 'replace' if arguments.replace else arguments.operation
        digest = (
            TaskSpecDigest.from_dict(arguments.task_spec.model_dump())
            if arguments.task_spec is not None
            else None
        )
        try:
            if operation == 'append':
                task = self.manager.append_steps(
                    titles,
                    constraints=arguments.constraints,
                    scope_hints=list(scope_patterns),
                    task_spec_digest=digest,
                    step_deliverables=step_deliverables,
                    step_criterion_ids=step_criterion_ids,
                )
            else:
                task = self.manager.plan(
                    titles,
                    constraints=arguments.constraints,
                    scope_hints=list(scope_patterns),
                    step_deliverables=step_deliverables,
                    step_criterion_ids=step_criterion_ids,
                    replace_existing=operation == 'replace',
                    task_spec_digest=digest,
                )
        except ExistingPlanError as error:
            raise ToolExecutionError(
                'task_plan_rejected',
                str(error),
                details=error.details,
            ) from error
        except ValueError as error:
            raise ToolExecutionError('task_plan_rejected', str(error)) from error
        return ToolResult.ok(
            (
                f'Appended {len(titles)} step(s) to the task plan.'
                if operation == 'append'
                else f'Created a {len(task.steps)}-step task plan.'
            ),
            content=self.manager.describe(),
            metadata={
                'task_id': task.id,
                'step_count': len(task.steps),
                'operation': operation,
                'ignored_scope_hints': ignored_scope_hints,
                'step_criterion_ids': {
                    step.id: list(step.criterion_ids)
                    for step in task.steps
                    if step.criterion_ids
                },
            },
        )


class TaskUpdateEvidenceInput(BaseModel):
    model_config = ConfigDict(extra='forbid')

    criterion_id: str = ''
    evidence_type: Literal[
        'source_change',
        'test_result',
        'typecheck',
        'build',
        'lint',
        'smoke',
        'symbol_evidence',
        'runtime_integration',
        'configuration',
        'review',
        'manual_limitation',
    ] = 'review'
    evidence_paths: list[str] = Field(default_factory=list, max_length=50)
    symbols: list[str] = Field(default_factory=list, max_length=50)
    verification_record_ids: list[str] = Field(
        default_factory=list,
        max_length=20,
    )
    source_revision: int = 0
    producer: Literal['tool', 'model', 'runtime', 'test'] = 'model'
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    explanation: str = Field(default='', max_length=2_000)
    status: Literal[
        'pending',
        'partially_satisfied',
        'satisfied',
        'blocked',
    ] | None = None


class TaskUpdateInput(ToolInput):
    step_id: str = Field(min_length=1)
    status: Literal['pending', 'in_progress', 'completed', 'blocked']
    evidence: list[str] = Field(default_factory=list, max_length=20)
    deliverables: list[str] = Field(default_factory=list, max_length=20)
    criterion_ids: list[str] = Field(default_factory=list, max_length=20)
    evidence_paths: list[str] = Field(default_factory=list, max_length=50)
    symbols: list[str] = Field(default_factory=list, max_length=50)
    verification_record_ids: list[str] = Field(
        default_factory=list,
        max_length=20,
    )
    acceptance_evidence: list[TaskUpdateEvidenceInput] = Field(
        default_factory=list,
        max_length=50,
    )


class TaskUpdateTool(Tool[TaskUpdateInput]):
    name = 'task_update'
    description = (
        'Update one step of the current active-goal linear plan. Use only '
        'when a step actually starts, completes, or becomes blocked. This '
        'tool cannot complete the whole task; ForgeCode completion checks own '
        'that state. This is not for claiming or completing durable '
        'task-graph items.'
    )
    input_model = TaskUpdateInput

    def __init__(
        self,
        root: Path,
        manager: TaskManager,
        *,
        acceptance_ledger: AcceptanceLedger | None = None,
        workspace_tracker: Any | None = None,
    ) -> None:
        super().__init__(root)
        self.manager = manager
        self.acceptance_ledger = acceptance_ledger
        self.workspace_tracker = workspace_tracker

    async def execute(self, arguments: TaskUpdateInput) -> ToolResult:
        previous = self.manager.active
        previous_step = (
            next(
                (
                    step
                    for step in previous.steps
                    if step.id == arguments.step_id
                ),
                None,
            )
            if previous is not None
            else None
        )
        try:
            task = self.manager.update_step(
                arguments.step_id,
                arguments.status,
                evidence=arguments.evidence,
            )
        except ValueError as error:
            raise ToolExecutionError('task_update_rejected', str(error)) from error
        current = task.current_step.title if task.current_step else 'none'
        source_revision = _source_revision(self.workspace_tracker)
        criterion_ids = tuple(
            dict.fromkeys(
                (
                    *(
                        previous_step.criterion_ids
                        if previous_step is not None
                        else ()
                    ),
                    *arguments.criterion_ids,
                )
            )
        )
        attached = _acceptance_evidence(
            arguments,
            criterion_ids=criterion_ids,
            source_revision=source_revision,
            ledger=self.acceptance_ledger,
        )
        completed_criteria = (
            self.acceptance_ledger.record_many(attached)
            if self.acceptance_ledger is not None
            else ()
        )
        evidence_valid = bool(attached) and all(
            item.criterion_id
            and (
                item.evidence_paths
                or item.symbols
                or item.verification_record_ids
                or item.evidence_type in {'review', 'manual_limitation'}
            )
            for item in attached
        )
        completed_plan_step = (
            arguments.status == 'completed' and evidence_valid
        )
        return ToolResult.ok(
            f'Updated {arguments.step_id} to {arguments.status}.',
            content=f'Current step: {current}',
            metadata={
                'task_id': task.id,
                'step_id': arguments.step_id,
                'status': arguments.status,
                'completed_plan_step': completed_plan_step,
                'evidence_valid': evidence_valid,
                'completed_criterion_ids': list(completed_criteria),
                'acceptance_evidence': [
                    item.as_dict() for item in attached
                ],
            },
        )


def create_task_tools(
    root: Path,
    manager: TaskManager,
    *,
    acceptance_ledger: AcceptanceLedger | None = None,
    workspace_tracker: Any | None = None,
) -> tuple[TaskGetTool, TaskPlanTool, TaskUpdateTool]:
    return (
        TaskGetTool(root, manager),
        TaskPlanTool(root, manager),
        TaskUpdateTool(
            root,
            manager,
            acceptance_ledger=acceptance_ledger,
            workspace_tracker=workspace_tracker,
        ),
    )


def _plan_step_links(
    steps: list[str | TaskPlanStepInput],
) -> tuple[list[str], dict[str, tuple[str, ...]], dict[str, tuple[str, ...]]]:
    titles: list[str] = []
    deliverables: dict[str, tuple[str, ...]] = {}
    criterion_ids: dict[str, tuple[str, ...]] = {}
    for index, item in enumerate(steps, start=1):
        step_id = f'step-{index}'
        if isinstance(item, str):
            titles.append(item)
            continue
        titles.append(item.title)
        deliverables[step_id] = tuple(item.deliverables)
        criterion_ids[step_id] = tuple(item.criterion_ids)
    return titles, deliverables, criterion_ids


def _acceptance_evidence(
    arguments: TaskUpdateInput,
    *,
    criterion_ids: tuple[str, ...],
    source_revision: int,
    ledger: AcceptanceLedger | None,
) -> tuple[AcceptanceEvidence, ...]:
    payloads: list[dict[str, Any]] = [
        item.model_dump(exclude_none=True)
        for item in arguments.acceptance_evidence
    ]
    if not payloads and criterion_ids and (
        arguments.evidence_paths
        or arguments.symbols
        or arguments.verification_record_ids
    ):
        payloads.extend(
            {
                'criterion_id': criterion_id,
                'evidence_type': (
                    'test_result'
                    if arguments.verification_record_ids
                    else 'source_change'
                ),
                'evidence_paths': arguments.evidence_paths,
                'symbols': arguments.symbols,
                'verification_record_ids': arguments.verification_record_ids,
                'source_revision': source_revision,
                'producer': 'model',
                'confidence': 0.8,
                'explanation': '; '.join(arguments.evidence),
            }
            for criterion_id in criterion_ids
        )
    evidence: list[AcceptanceEvidence] = []
    for payload in payloads:
        criterion = str(payload.get('criterion_id', ''))
        criterion_text = (
            ledger.criterion_text(criterion)
            if ledger is not None
            else ''
        )
        evidence.append(
            evidence_from_payload(
                payload,
                criterion_text=criterion_text,
                source_revision=source_revision,
                plan_step_id=arguments.step_id,
            )
        )
    return tuple(evidence)


def _source_revision(workspace_tracker: Any | None) -> int:
    if workspace_tracker is None:
        return 0
    return int(
        getattr(
            workspace_tracker,
            'source_revision',
            getattr(workspace_tracker, 'revision', 0),
        )
    )
