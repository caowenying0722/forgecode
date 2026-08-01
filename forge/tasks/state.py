'''Provider-neutral task state kept outside conversation history.'''

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal


TaskStatus = Literal['in_progress', 'completed', 'blocked', 'stuck']
StepStatus = Literal['pending', 'in_progress', 'completed', 'blocked']


@dataclass(frozen=True, slots=True)
class SourceSection:
    '''A stable link back to an authoritative task document section.'''

    path: str
    start_line: int
    end_line: int
    title: str = ''

    def __post_init__(self) -> None:
        if self.start_line < 1 or self.end_line < self.start_line:
            raise ValueError('Source section lines must form a positive range.')

    @property
    def reference(self) -> str:
        return f'{self.path}:{self.start_line}-{self.end_line}'

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SourceSection:
        return cls(
            path=str(data['path']),
            start_line=int(data['start_line']),
            end_line=int(data['end_line']),
            title=str(data.get('title', '')),
        )


@dataclass(frozen=True, slots=True)
class TaskSpecDigest:
    '''Structured navigation aid for an authoritative task specification.'''

    source_paths: tuple[str, ...]
    goal: str
    requirements: tuple[str, ...] = ()
    acceptance_criteria: tuple[str, ...] = ()
    required_commands: tuple[str, ...] = ()
    required_modules: tuple[str, ...] = ()
    forbidden_changes: tuple[str, ...] = ()
    relevant_sections: tuple[SourceSection, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskSpecDigest:
        def strings(name: str) -> tuple[str, ...]:
            return tuple(str(item) for item in data.get(name, []))

        return cls(
            source_paths=strings('source_paths'),
            goal=str(data.get('goal', '')),
            requirements=strings('requirements'),
            acceptance_criteria=strings('acceptance_criteria'),
            required_commands=strings('required_commands'),
            required_modules=strings('required_modules'),
            forbidden_changes=strings('forbidden_changes'),
            relevant_sections=tuple(
                SourceSection.from_dict(item)
                for item in data.get('relevant_sections', [])
                if isinstance(item, dict)
            ),
        )

    def render(self) -> str:
        lines = [
            '[TaskSpec Digest]',
            'This digest is a navigation aid. The linked authoritative source '
            'documents remain controlling.',
            'Source documents:',
            *[f'- {path}' for path in self.source_paths],
            'TaskSpec goal:',
            self.goal,
        ]
        for label, values in (
            ('Requirements', self.requirements),
            ('Acceptance criteria', self.acceptance_criteria),
            ('Required commands', self.required_commands),
            ('Required modules', self.required_modules),
            ('Forbidden changes', self.forbidden_changes),
        ):
            if values:
                lines.extend([f'{label}:', *[f'- {item}' for item in values]])
        if self.relevant_sections:
            lines.append('Relevant authoritative source sections:')
            lines.extend(
                f'- {section.reference}'
                + (f' ({section.title})' if section.title else '')
                for section in self.relevant_sections
            )
        return '\n'.join(lines)


@dataclass(frozen=True, slots=True)
class TaskStep:
    id: str
    title: str
    status: StepStatus = 'pending'
    evidence: tuple[str, ...] = ()
    deliverables: tuple[str, ...] = ()
    criterion_ids: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskStep:
        return cls(
            id=str(data['id']),
            title=str(data['title']),
            status=str(data.get('status', 'pending')),  # type: ignore[arg-type]
            evidence=tuple(str(item) for item in data.get('evidence', [])),
            deliverables=tuple(
                str(item) for item in data.get('deliverables', [])
            ),
            criterion_ids=tuple(
                str(item) for item in data.get('criterion_ids', [])
            ),
        )


@dataclass(frozen=True, slots=True)
class ActiveTask:
    id: str
    goal: str
    status: TaskStatus = 'in_progress'
    planned: bool = False
    current_step_id: str | None = None
    steps: tuple[TaskStep, ...] = ()
    constraints: tuple[str, ...] = ()
    scope_hints: tuple[str, ...] = ()
    blocked_reasons: tuple[str, ...] = ()
    task_spec_digest: TaskSpecDigest | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ActiveTask:
        return cls(
            id=str(data['id']),
            goal=str(data['goal']),
            status=str(data.get('status', 'in_progress')),  # type: ignore[arg-type]
            planned=bool(data.get('planned', False)),
            current_step_id=(
                str(data['current_step_id'])
                if data.get('current_step_id') is not None
                else None
            ),
            steps=tuple(
                TaskStep.from_dict(item)
                for item in data.get('steps', [])
                if isinstance(item, dict)
            ),
            constraints=tuple(
                str(item) for item in data.get('constraints', [])
            ),
            scope_hints=tuple(
                str(item) for item in data.get('scope_hints', [])
            ),
            blocked_reasons=tuple(
                str(item) for item in data.get('blocked_reasons', [])
            ),
            task_spec_digest=(
                TaskSpecDigest.from_dict(data['task_spec_digest'])
                if isinstance(data.get('task_spec_digest'), dict)
                else None
            ),
        )

    @property
    def current_step(self) -> TaskStep | None:
        return next(
            (
                step
                for step in self.steps
                if step.id == self.current_step_id
            ),
            None,
        )
