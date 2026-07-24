'''Persistent repository task graph with dependency-aware claiming.'''

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
import json
from pathlib import Path
import re
from uuid import uuid4
from typing import Any, Literal


GraphTaskStatus = Literal['pending', 'in_progress', 'completed', 'blocked']


@dataclass(frozen=True, slots=True)
class GraphTask:
    id: str
    subject: str
    description: str = ''
    status: GraphTaskStatus = 'pending'
    owner: str | None = None
    blocked_by: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()
    created_at: str = ''
    updated_at: str = ''

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GraphTask:
        return cls(
            id=str(data['id']),
            subject=str(data['subject']),
            description=str(data.get('description', '')),
            status=str(data.get('status', 'pending')),  # type: ignore[arg-type]
            owner=(
                str(data['owner']) if data.get('owner') is not None else None
            ),
            blocked_by=tuple(
                str(item)
                for item in data.get('blocked_by', data.get('blockedBy', []))
            ),
            evidence=tuple(str(item) for item in data.get('evidence', [])),
            created_at=str(data.get('created_at', '')),
            updated_at=str(data.get('updated_at', '')),
        )


class TaskGraphStore:
    '''Store one JSON file per graph task under .forge/task-graph.'''

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.directory = self.root / '.forge' / 'task-graph'

    def create(
        self,
        subject: str,
        *,
        description: str = '',
        blocked_by: list[str] | None = None,
    ) -> GraphTask:
        clean_subject = clean_text(subject, name='subject', maximum=500)
        clean_description = clean_text(
            description,
            name='description',
            maximum=10_000,
            allow_empty=True,
        )
        dependencies = clean_task_ids(blocked_by or [])
        missing = [task_id for task_id in dependencies if not self.exists(task_id)]
        if missing:
            raise ValueError(f'Blocked-by task IDs do not exist: {", ".join(missing)}')
        task_id = self._new_id()
        task = GraphTask(
            id=task_id,
            subject=clean_subject,
            description=clean_description,
            blocked_by=tuple(dependencies),
            created_at=now_utc(),
            updated_at=now_utc(),
        )
        self._ensure_acyclic(task)
        self.save(task)
        return task

    def list(self) -> tuple[GraphTask, ...]:
        if not self.directory.exists():
            return ()
        tasks: list[GraphTask] = []
        for path in sorted(self.directory.glob('graph-task-*.json')):
            try:
                tasks.append(self._read(path))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
        return tuple(tasks)

    def load(self, task_id: str) -> GraphTask:
        validate_task_id(task_id)
        return self._read(self._path(task_id))

    def exists(self, task_id: str) -> bool:
        try:
            validate_task_id(task_id)
        except ValueError:
            return False
        return self._path(task_id).exists()

    def can_start(self, task_id: str) -> bool:
        task = self.load(task_id)
        return not self.blocking_dependencies(task)

    def blocking_dependencies(self, task: GraphTask) -> tuple[str, ...]:
        blocked: list[str] = []
        for dependency_id in task.blocked_by:
            if not self.exists(dependency_id):
                blocked.append(dependency_id)
                continue
            dependency = self.load(dependency_id)
            if dependency.status != 'completed':
                blocked.append(dependency_id)
        return tuple(blocked)

    def claim(self, task_id: str, *, owner: str) -> GraphTask:
        task = self.load(task_id)
        if task.status != 'pending':
            raise ValueError(f'Task {task_id} is {task.status}, cannot claim.')
        blocked = self.blocking_dependencies(task)
        if blocked:
            raise ValueError(f'Task {task_id} is blocked by: {", ".join(blocked)}')
        clean_owner = clean_text(owner, name='owner', maximum=200)
        updated = replace(
            task,
            status='in_progress',
            owner=clean_owner,
            updated_at=now_utc(),
        )
        self.save(updated)
        return updated

    def complete(
        self,
        task_id: str,
        *,
        evidence: list[str] | None = None,
    ) -> tuple[GraphTask, tuple[GraphTask, ...]]:
        task = self.load(task_id)
        if task.status != 'in_progress':
            raise ValueError(
                f'Task {task_id} is {task.status}, cannot complete.'
            )
        additions = clean_texts(evidence or [], name='evidence', maximum=20)
        updated = replace(
            task,
            status='completed',
            evidence=tuple(dict.fromkeys((*task.evidence, *additions))),
            updated_at=now_utc(),
        )
        self.save(updated)
        unblocked = tuple(
            candidate
            for candidate in self.list()
            if candidate.status == 'pending'
            and candidate.blocked_by
            and self.can_start(candidate.id)
        )
        return updated, unblocked

    def block(self, task_id: str, *, reason: str) -> GraphTask:
        task = self.load(task_id)
        clean_reason = clean_text(reason, name='reason', maximum=1_000)
        updated = replace(
            task,
            status='blocked',
            evidence=tuple(dict.fromkeys((*task.evidence, clean_reason))),
            updated_at=now_utc(),
        )
        self.save(updated)
        return updated

    def save(self, task: GraphTask) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self._path(task.id)
        serialized = json.dumps(
            task.as_dict(),
            ensure_ascii=False,
            indent=2,
        )
        temporary = path.with_suffix(path.suffix + '.tmp')
        temporary.write_text(serialized + '\n', encoding='utf-8')
        temporary.replace(path)
        return path

    def _new_id(self) -> str:
        while True:
            task_id = f'graph-task-{uuid4().hex[:12]}'
            if not self._path(task_id).exists():
                return task_id

    def _path(self, task_id: str) -> Path:
        validate_task_id(task_id)
        return self.directory / f'{task_id}.json'

    @staticmethod
    def _read(path: Path) -> GraphTask:
        data = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(data, dict):
            raise ValueError(f'Invalid task graph file: {path}')
        return GraphTask.from_dict(data)

    def _ensure_acyclic(self, candidate: GraphTask) -> None:
        tasks = {task.id: task for task in self.list()}
        tasks[candidate.id] = candidate

        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(task_id: str) -> None:
            if task_id in visited:
                return
            if task_id in visiting:
                raise ValueError('Task dependencies must not contain a cycle.')
            task = tasks.get(task_id)
            if task is None:
                return
            visiting.add(task_id)
            for dependency_id in task.blocked_by:
                visit(dependency_id)
            visiting.remove(task_id)
            visited.add(task_id)

        for task_id in tasks:
            visit(task_id)


def validate_task_id(task_id: str) -> None:
    if re.fullmatch(r'graph-task-[0-9a-f]{12}', task_id) is None:
        raise ValueError(f'Invalid task ID: {task_id}')


def now_utc() -> str:
    return datetime.now(UTC).isoformat()


def clean_task_ids(values: list[str]) -> list[str]:
    cleaned = clean_texts(values, name='blocked_by', maximum=50)
    for task_id in cleaned:
        validate_task_id(task_id)
    return cleaned


def clean_texts(
    values: list[str],
    *,
    name: str,
    maximum: int,
) -> list[str]:
    if len(values) > maximum:
        raise ValueError(f'{name} may contain at most {maximum} items.')
    return list(
        dict.fromkeys(
            clean_text(value, name=name, maximum=1_000) for value in values
        )
    )


def clean_text(
    value: str,
    *,
    name: str,
    maximum: int,
    allow_empty: bool = False,
) -> str:
    cleaned = str(value).strip()
    if not cleaned and not allow_empty:
        raise ValueError(f'{name} must not be empty.')
    if len(cleaned) > maximum:
        raise ValueError(f'{name} is limited to {maximum} characters.')
    return cleaned
