'''Persistent executable task graph with dependency and resource analysis.'''

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
import json
import os
from pathlib import Path, PurePosixPath
import re
import time
from typing import Any, Literal
from uuid import uuid4


GraphTaskStatus = Literal['pending', 'in_progress', 'completed', 'blocked']
ExecutionMode = Literal['parallel', 'cautious', 'serial']
ConflictKind = Literal[
    'read_only_overlap',
    'same_file_different_symbols',
    'write_overlap',
    'serial_constraint',
]


@dataclass(frozen=True, slots=True)
class ResourceScope:
    '''A predicted file/subtree scope and optional symbol ownership.'''

    path: str
    symbols: tuple[str, ...] = ()
    logical_area: str = ''

    @classmethod
    def from_value(cls, value: object) -> ResourceScope:
        if isinstance(value, str):
            return cls(path=clean_scope_path(value))
        if not isinstance(value, dict):
            raise ValueError('resource scope must be a path string or object.')
        symbols = clean_texts(
            [str(item) for item in value.get('symbols', [])],
            name='symbols',
            maximum=50,
        )
        return cls(
            path=clean_scope_path(str(value.get('path', ''))),
            symbols=tuple(symbols),
            logical_area=clean_text(
                str(value.get('logical_area', value.get('logicalArea', ''))),
                name='logical_area',
                maximum=500,
                allow_empty=True,
            ),
        )


@dataclass(frozen=True, slots=True)
class TaskConflict:
    left_task_id: str
    right_task_id: str
    kind: ConflictKind
    resources: tuple[str, ...] = ()
    blocks_parallel: bool = True

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class GraphTask:
    id: str
    subject: str
    description: str = ''
    status: GraphTaskStatus = 'pending'
    owner: str | None = None
    blocked_by: tuple[str, ...] = ()
    acceptance_criteria: tuple[str, ...] = ()
    read_scope: tuple[ResourceScope, ...] = ()
    write_scope: tuple[ResourceScope, ...] = ()
    execution: ExecutionMode = 'parallel'
    verification: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()
    created_at: str = ''
    updated_at: str = ''

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GraphTask:
        execution = str(data.get('execution', 'parallel'))
        if execution not in {'parallel', 'cautious', 'serial'}:
            execution = 'parallel'
        return cls(
            id=str(data['id']),
            subject=str(data['subject']),
            description=str(data.get('description', '')),
            status=str(data.get('status', 'pending')),  # type: ignore[arg-type]
            owner=str(data['owner']) if data.get('owner') is not None else None,
            blocked_by=tuple(
                str(item)
                for item in data.get('blocked_by', data.get('blockedBy', []))
            ),
            acceptance_criteria=tuple(
                str(item)
                for item in data.get(
                    'acceptance_criteria', data.get('acceptanceCriteria', [])
                )
            ),
            read_scope=tuple(
                ResourceScope.from_value(item)
                for item in data.get('read_scope', data.get('readScope', []))
            ),
            write_scope=tuple(
                ResourceScope.from_value(item)
                for item in data.get('write_scope', data.get('writeScope', []))
            ),
            execution=execution,  # type: ignore[arg-type]
            verification=tuple(str(item) for item in data.get('verification', [])),
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
        acceptance_criteria: list[str] | None = None,
        read_scope: list[object] | None = None,
        write_scope: list[object] | None = None,
        execution: ExecutionMode = 'parallel',
        verification: list[str] | None = None,
    ) -> GraphTask:
        if execution not in {'parallel', 'cautious', 'serial'}:
            raise ValueError('execution must be parallel, cautious, or serial.')
        dependencies = clean_task_ids(blocked_by or [])
        missing = [task_id for task_id in dependencies if not self.exists(task_id)]
        if missing:
            raise ValueError(f'Blocked-by task IDs do not exist: {", ".join(missing)}')
        task_id = self._new_id()
        timestamp = now_utc()
        task = GraphTask(
            id=task_id,
            subject=clean_text(subject, name='subject', maximum=500),
            description=clean_text(
                description, name='description', maximum=10_000, allow_empty=True
            ),
            blocked_by=tuple(dependencies),
            acceptance_criteria=tuple(
                clean_texts(
                    acceptance_criteria or [],
                    name='acceptance_criteria',
                    maximum=50,
                )
            ),
            read_scope=clean_scopes(read_scope or [], name='read_scope'),
            write_scope=clean_scopes(write_scope or [], name='write_scope'),
            execution=execution,
            verification=tuple(
                clean_texts(
                    verification or [], name='verification', maximum=50
                )
            ),
            created_at=timestamp,
            updated_at=timestamp,
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
        return tuple(sorted(tasks, key=lambda task: (task.created_at, task.id)))

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
        return not self.blocking_dependencies(task) and not self.active_conflicts(task)

    def blocking_dependencies(self, task: GraphTask) -> tuple[str, ...]:
        blocked: list[str] = []
        for dependency_id in task.blocked_by:
            if not self.exists(dependency_id):
                blocked.append(dependency_id)
                continue
            if self.load(dependency_id).status != 'completed':
                blocked.append(dependency_id)
        return tuple(blocked)

    def conflict(self, left: GraphTask, right: GraphTask) -> TaskConflict | None:
        if left.id == right.id:
            return None
        if left.execution == 'serial' or right.execution == 'serial':
            return TaskConflict(left.id, right.id, 'serial_constraint')

        hard_resources: set[str] = set()
        symbol_resources: set[str] = set()
        read_resources: set[str] = set()
        for left_scope, left_mode in task_scopes(left):
            for right_scope, right_mode in task_scopes(right):
                logical_overlap = (
                    left_scope.logical_area
                    and left_scope.logical_area == right_scope.logical_area
                )
                path_overlap = scope_paths_overlap(left_scope.path, right_scope.path)
                if not logical_overlap and not path_overlap:
                    continue
                label = (
                    f'logical:{left_scope.logical_area}'
                    if logical_overlap
                    else common_scope_label(left_scope.path, right_scope.path)
                )
                if left_mode == right_mode == 'read':
                    read_resources.add(label)
                    continue
                if (
                    left_mode == right_mode == 'write'
                    and left_scope.path == right_scope.path
                    and left_scope.symbols
                    and right_scope.symbols
                    and set(left_scope.symbols).isdisjoint(right_scope.symbols)
                    and not logical_overlap
                ):
                    symbol_resources.add(label)
                    continue
                hard_resources.add(label)

        if hard_resources:
            return TaskConflict(
                left.id,
                right.id,
                'write_overlap',
                tuple(sorted(hard_resources)),
            )
        if symbol_resources:
            blocks = left.execution != 'cautious' or right.execution != 'cautious'
            return TaskConflict(
                left.id,
                right.id,
                'same_file_different_symbols',
                tuple(sorted(symbol_resources)),
                blocks_parallel=blocks,
            )
        if read_resources:
            return TaskConflict(
                left.id,
                right.id,
                'read_only_overlap',
                tuple(sorted(read_resources)),
                blocks_parallel=False,
            )
        return None

    def active_conflicts(self, task: GraphTask) -> tuple[TaskConflict, ...]:
        conflicts: list[TaskConflict] = []
        for active in self.list():
            if active.status != 'in_progress' or active.id == task.id:
                continue
            conflict = self.conflict(task, active)
            if conflict is not None and conflict.blocks_parallel:
                conflicts.append(conflict)
        return tuple(conflicts)

    def ready_tasks(self) -> tuple[GraphTask, ...]:
        return tuple(
            task
            for task in self.list()
            if task.status == 'pending'
            and not task.owner
            and not self.blocking_dependencies(task)
            and not self.active_conflicts(task)
        )

    def ready_wave(self) -> tuple[GraphTask, ...]:
        '''Return a deterministic maximal batch safe to start concurrently.'''
        selected: list[GraphTask] = []
        for task in self.ready_tasks():
            conflicts = [self.conflict(task, other) for other in selected]
            if any(item is not None and item.blocks_parallel for item in conflicts):
                continue
            selected.append(task)
        return tuple(selected)

    def analysis(self, *, include_completed: bool = True) -> dict[str, Any]:
        tasks = tuple(
            task
            for task in self.list()
            if include_completed or task.status != 'completed'
        )
        conflicts: list[TaskConflict] = []
        for index, left in enumerate(tasks):
            for right in tasks[index + 1 :]:
                if left.status == 'completed' or right.status == 'completed':
                    continue
                conflict = self.conflict(left, right)
                if conflict is not None:
                    conflicts.append(conflict)
        return {
            'nodes': [task.as_dict() for task in tasks],
            'dependency_edges': [
                {'from': dependency, 'to': task.id}
                for task in tasks
                for dependency in task.blocked_by
            ],
            'resource_conflicts': [item.as_dict() for item in conflicts],
            'ready_wave': [task.id for task in self.ready_wave()],
        }

    def claim(self, task_id: str, *, owner: str) -> GraphTask:
        with self._mutation_lock():
            return self._claim_locked(task_id, owner=owner)

    def _claim_locked(self, task_id: str, *, owner: str) -> GraphTask:
        task = self.load(task_id)
        if task.status != 'pending':
            raise ValueError(f'Task {task_id} is {task.status}, cannot claim.')
        blocked = self.blocking_dependencies(task)
        if blocked:
            raise ValueError(f'Task {task_id} is blocked by: {", ".join(blocked)}')
        conflicts = self.active_conflicts(task)
        if conflicts:
            task_ids = sorted({item.right_task_id for item in conflicts})
            raise ValueError(
                f'Task {task_id} conflicts with active tasks: {", ".join(task_ids)}'
            )
        updated = replace(
            task,
            status='in_progress',
            owner=clean_text(owner, name='owner', maximum=200),
            updated_at=now_utc(),
        )
        self.save(updated)
        return updated

    def complete(
        self, task_id: str, *, evidence: list[str] | None = None
    ) -> tuple[GraphTask, tuple[GraphTask, ...]]:
        with self._mutation_lock():
            return self._complete_locked(task_id, evidence=evidence)

    def _complete_locked(
        self, task_id: str, *, evidence: list[str] | None = None
    ) -> tuple[GraphTask, tuple[GraphTask, ...]]:
        task = self.load(task_id)
        if task.status != 'in_progress':
            raise ValueError(f'Task {task_id} is {task.status}, cannot complete.')
        additions = clean_texts(evidence or [], name='evidence', maximum=20)
        if task.verification and not task.evidence and not additions:
            raise ValueError(
                f'Task {task_id} declares verification requirements; '
                'completion evidence is required.'
            )
        combined_evidence = tuple(dict.fromkeys((*task.evidence, *additions)))
        missing_requirements = tuple(
            requirement
            for requirement in task.verification
            if not any(
                requirement.casefold() in item.casefold()
                for item in combined_evidence
            )
        )
        if missing_requirements:
            raise ValueError(
                'Completion evidence does not reference planned verification: '
                + ', '.join(missing_requirements)
            )
        updated = replace(
            task,
            status='completed',
            evidence=combined_evidence,
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
        with self._mutation_lock():
            return self._block_locked(task_id, reason=reason)

    def _block_locked(self, task_id: str, *, reason: str) -> GraphTask:
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

    @contextmanager
    def _mutation_lock(self) -> Iterator[None]:
        '''Serialize graph state transitions across agents and processes.'''
        self.directory.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.directory.parent / 'task-graph.lock'
        deadline = time.monotonic() + 5.0
        descriptor: int | None = None
        while descriptor is None:
            try:
                descriptor = os.open(
                    lock_path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                )
            except FileExistsError:
                try:
                    stale = time.time() - lock_path.stat().st_mtime > 30.0
                    if stale:
                        lock_path.unlink(missing_ok=True)
                        continue
                except FileNotFoundError:
                    continue
                if time.monotonic() >= deadline:
                    raise ValueError('Timed out waiting for task-graph mutation lock.')
                time.sleep(0.01)
        try:
            os.write(descriptor, f'{os.getpid()}\n'.encode())
            yield
        finally:
            os.close(descriptor)
            lock_path.unlink(missing_ok=True)

    def save(self, task: GraphTask) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self._path(task.id)
        serialized = json.dumps(task.as_dict(), ensure_ascii=False, indent=2)
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


def task_scopes(task: GraphTask) -> tuple[tuple[ResourceScope, str], ...]:
    return tuple((item, 'read') for item in task.read_scope) + tuple(
        (item, 'write') for item in task.write_scope
    )


def scope_paths_overlap(left: str, right: str) -> bool:
    if left == right:
        return True
    left_prefix = left[:-3].rstrip('/') if left.endswith('/**') else None
    right_prefix = right[:-3].rstrip('/') if right.endswith('/**') else None
    if left_prefix is not None and (right == left_prefix or right.startswith(left_prefix + '/')):
        return True
    if right_prefix is not None and (left == right_prefix or left.startswith(right_prefix + '/')):
        return True
    return False


def common_scope_label(left: str, right: str) -> str:
    if left == right:
        return left
    return left if left.endswith('/**') else right


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


def clean_scopes(values: list[object], *, name: str) -> tuple[ResourceScope, ...]:
    if len(values) > 50:
        raise ValueError(f'{name} may contain at most 50 items.')
    scopes = [ResourceScope.from_value(value) for value in values]
    return tuple(dict.fromkeys(scopes))


def clean_scope_path(value: str) -> str:
    cleaned = value.strip().replace('\\', '/')
    if not cleaned:
        raise ValueError('resource scope path must not be empty.')
    path = PurePosixPath(cleaned)
    if path.is_absolute() or '..' in path.parts or re.match(r'^[A-Za-z]:', cleaned):
        raise ValueError('resource scope path must be repository-relative.')
    if len(cleaned) > 1_000:
        raise ValueError('resource scope path is limited to 1000 characters.')
    return path.as_posix()


def clean_texts(values: list[str], *, name: str, maximum: int) -> list[str]:
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
