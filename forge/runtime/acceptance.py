'''Generic acceptance evidence tracking for task completion.'''

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Literal

from forge.runtime.intent import TaskContract
from forge.runtime.state import VerificationEvidence
from forge.runtime.verification_ledger import VerificationRecord


AcceptanceStatus = Literal[
    'pending',
    'partially_satisfied',
    'satisfied',
    'blocked',
]
AcceptanceEvidenceType = Literal[
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
]
AcceptanceProducer = Literal['tool', 'model', 'runtime', 'test']


@dataclass(frozen=True, slots=True)
class AcceptanceEvidence:
    criterion_id: str
    criterion_text: str
    status: AcceptanceStatus
    evidence_type: AcceptanceEvidenceType
    evidence_paths: tuple[str, ...] = ()
    symbols: tuple[str, ...] = ()
    plan_step_id: str = ''
    verification_record_ids: tuple[str, ...] = ()
    source_revision: int = 0
    producer: AcceptanceProducer = 'runtime'
    confidence: float = 0.0
    explanation: str = ''

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'AcceptanceEvidence':
        return cls(
            criterion_id=str(data.get('criterion_id', '')),
            criterion_text=str(data.get('criterion_text', '')),
            status=_literal_status(data.get('status')),
            evidence_type=_literal_evidence_type(data.get('evidence_type')),
            evidence_paths=_string_tuple(data.get('evidence_paths')),
            symbols=_string_tuple(data.get('symbols')),
            plan_step_id=str(data.get('plan_step_id', '')),
            verification_record_ids=_string_tuple(
                data.get('verification_record_ids')
            ),
            source_revision=_int_value(data.get('source_revision')),
            producer=_literal_producer(data.get('producer')),
            confidence=_confidence(data.get('confidence')),
            explanation=str(data.get('explanation', '')),
        )


@dataclass(frozen=True, slots=True)
class AcceptanceCriterionState:
    criterion_id: str
    criterion_text: str
    status: AcceptanceStatus = 'pending'
    evidence: tuple[AcceptanceEvidence, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data['evidence'] = [item.as_dict() for item in self.evidence]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'AcceptanceCriterionState':
        return cls(
            criterion_id=str(data.get('criterion_id', '')),
            criterion_text=str(data.get('criterion_text', '')),
            status=_literal_status(data.get('status')),
            evidence=tuple(
                AcceptanceEvidence.from_dict(item)
                for item in data.get('evidence', [])
                if isinstance(item, dict)
            ),
        )


@dataclass(slots=True)
class AcceptanceLedger:
    criteria: dict[str, AcceptanceCriterionState] = field(
        default_factory=dict
    )
    current_source_revision: int = 0

    def configure(self, contract: TaskContract) -> None:
        self.criteria = {
            criterion_id(index): AcceptanceCriterionState(
                criterion_id=criterion_id(index),
                criterion_text=text,
            )
            for index, text in enumerate(contract.acceptance_criteria, start=1)
        }
        self.current_source_revision = 0

    @classmethod
    def from_contract(cls, contract: TaskContract) -> 'AcceptanceLedger':
        ledger = cls()
        ledger.configure(contract)
        return ledger

    def as_dict(self) -> dict[str, Any]:
        return {
            'current_source_revision': self.current_source_revision,
            'criteria': [
                self.criteria[key].as_dict()
                for key in sorted(self.criteria, key=_criterion_sort_key)
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'AcceptanceLedger':
        ledger = cls(
            current_source_revision=_int_value(
                data.get('current_source_revision')
            )
        )
        ledger.criteria = {
            item.criterion_id: item
            for item in (
                AcceptanceCriterionState.from_dict(raw)
                for raw in data.get('criteria', [])
                if isinstance(raw, dict)
            )
            if item.criterion_id
        }
        return ledger

    def criterion_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.criteria, key=_criterion_sort_key))

    def criterion_text(self, criterion_id: str) -> str:
        state = self.criteria.get(criterion_id)
        return state.criterion_text if state is not None else ''

    def satisfied_ids(self) -> tuple[str, ...]:
        return tuple(
            key
            for key in self.criterion_ids()
            if self.criteria[key].status == 'satisfied'
        )

    def partial_ids(self) -> tuple[str, ...]:
        return tuple(
            key
            for key in self.criterion_ids()
            if self.criteria[key].status == 'partially_satisfied'
        )

    def pending_ids(self) -> tuple[str, ...]:
        return tuple(
            key
            for key in self.criterion_ids()
            if self.criteria[key].status in {'pending', 'blocked'}
        )

    @property
    def acceptance_verified(self) -> bool:
        return bool(self.criteria) and all(
            state.status == 'satisfied' for state in self.criteria.values()
        )

    def evidence_snapshot(
        self,
        *,
        criterion_ids: tuple[str, ...] = (),
        source_revision: int | None = None,
    ) -> tuple[AcceptanceEvidence, ...]:
        selected = criterion_ids or self.criterion_ids()
        evidence: list[AcceptanceEvidence] = []
        for criterion_id in selected:
            state = self.criteria.get(criterion_id)
            if state is None:
                continue
            evidence.extend(
                item
                for item in state.evidence
                if (
                    source_revision is None
                    or item.source_revision == source_revision
                )
            )
        return tuple(evidence)

    def record_evidence(
        self,
        evidence: AcceptanceEvidence,
    ) -> tuple[str, ...]:
        if evidence.criterion_id not in self.criteria:
            return ()
        previous = self.criteria[evidence.criterion_id]
        merged = (*previous.evidence, evidence)
        status = _status_for_evidence(
            previous.criterion_text,
            merged,
            current_source_revision=self.current_source_revision,
        )
        self.criteria[evidence.criterion_id] = replace(
            previous,
            status=status,
            evidence=merged,
        )
        if previous.status != 'satisfied' and status == 'satisfied':
            return (evidence.criterion_id,)
        return ()

    def record_many(
        self,
        evidence: tuple[AcceptanceEvidence, ...],
    ) -> tuple[str, ...]:
        completed: list[str] = []
        for item in evidence:
            completed.extend(self.record_evidence(item))
        return tuple(dict.fromkeys(completed))

    def observe_source_change(
        self,
        paths: tuple[str, ...],
        *,
        source_revision: int,
        criterion_ids: tuple[str, ...] = (),
        plan_step_id: str = '',
        symbols: tuple[str, ...] = (),
    ) -> tuple[str, ...]:
        if source_revision > self.current_source_revision:
            self.current_source_revision = source_revision
            self.refresh_statuses()
        target_ids = criterion_ids or tuple(
            criterion_id
            for criterion_id, state in self.criteria.items()
            if _criterion_accepts_source_change(state.criterion_text)
            or _required_evidence_types(state.criterion_text)
        )
        evidence = tuple(
            AcceptanceEvidence(
                criterion_id=item,
                criterion_text=self.criterion_text(item),
                status=_source_evidence_status(self.criterion_text(item)),
                evidence_type='source_change',
                evidence_paths=paths,
                symbols=symbols,
                plan_step_id=plan_step_id,
                source_revision=source_revision,
                producer='runtime',
                confidence=0.7,
                explanation='Task-relevant source or configuration changed.',
            )
            for item in target_ids
            if item in self.criteria
        )
        return self.record_many(evidence)

    def observe_plan(
        self,
        *,
        step_ids: tuple[str, ...],
    ) -> tuple[str, ...]:
        target_ids = tuple(
            criterion_id
            for criterion_id, state in self.criteria.items()
            if 'plan' in state.criterion_text.casefold()
            or '计划' in state.criterion_text
        )
        evidence = tuple(
            AcceptanceEvidence(
                criterion_id=item,
                criterion_text=self.criterion_text(item),
                status='satisfied',
                evidence_type='review',
                plan_step_id=','.join(step_ids),
                producer='tool',
                confidence=0.9,
                explanation='A structured task plan was created.',
            )
            for item in target_ids
            if item in self.criteria
        )
        return self.record_many(evidence)

    def observe_verification(
        self,
        verification: VerificationEvidence | VerificationRecord,
    ) -> tuple[str, ...]:
        source_revision = int(
            getattr(
                verification,
                'source_revision',
                getattr(verification, 'workspace_revision', 0),
            )
            or 0
        )
        if source_revision > self.current_source_revision:
            self.current_source_revision = source_revision
        if not bool(getattr(verification, 'success', False)):
            self.refresh_statuses()
            return ()
        evidence_type = verification_evidence_type(
            str(
                getattr(
                    verification,
                    'verification_type',
                    getattr(verification, 'kind', 'auto'),
                )
            )
        )
        record_id = verification_record_id(verification)
        target_ids = tuple(
            criterion_id
            for criterion_id, state in self.criteria.items()
            if _criterion_accepts_verification(
                state.criterion_text,
                evidence_type,
            )
        )
        evidence = tuple(
            AcceptanceEvidence(
                criterion_id=item,
                criterion_text=self.criterion_text(item),
                status='satisfied',
                evidence_type=evidence_type,
                verification_record_ids=(record_id,),
                source_revision=source_revision,
                producer='test',
                confidence=0.95,
                explanation='Verification passed for the current source revision.',
            )
            for item in target_ids
            if item in self.criteria
        )
        completed = self.record_many(evidence)
        self.refresh_statuses()
        return completed

    def refresh_statuses(self) -> None:
        for criterion_id, state in tuple(self.criteria.items()):
            self.criteria[criterion_id] = replace(
                state,
                status=_status_for_evidence(
                    state.criterion_text,
                    state.evidence,
                    current_source_revision=self.current_source_revision,
                ),
            )

    def gap_report(self) -> dict[str, object]:
        return {
            'satisfied_criteria': tuple(
                state.criterion_text
                for state in self.criteria.values()
                if state.status == 'satisfied'
            ),
            'partially_satisfied_criteria': tuple(
                state.criterion_text
                for state in self.criteria.values()
                if state.status == 'partially_satisfied'
            ),
            'missing_criteria': tuple(
                state.criterion_text
                for state in self.criteria.values()
                if state.status in {'pending', 'blocked'}
            ),
            'missing_evidence': tuple(
                _missing_evidence_for(state.criterion_text)
                for state in self.criteria.values()
                if state.status != 'satisfied'
            ),
        }


def criterion_id(index: int) -> str:
    return f'criterion-{index}'


def evidence_from_payload(
    payload: dict[str, Any],
    *,
    criterion_text: str,
    status: AcceptanceStatus | None = None,
    source_revision: int = 0,
    plan_step_id: str = '',
) -> AcceptanceEvidence:
    evidence_type = _literal_evidence_type(payload.get('evidence_type'))
    resolved_status = status or _payload_status(payload, evidence_type)
    return AcceptanceEvidence(
        criterion_id=str(payload.get('criterion_id', '')),
        criterion_text=criterion_text,
        status=resolved_status,
        evidence_type=evidence_type,
        evidence_paths=_string_tuple(payload.get('evidence_paths')),
        symbols=_string_tuple(payload.get('symbols')),
        plan_step_id=str(payload.get('plan_step_id') or plan_step_id),
        verification_record_ids=_string_tuple(
            payload.get('verification_record_ids')
        ),
        source_revision=_int_value(
            payload.get('source_revision'),
            fallback=source_revision,
        ),
        producer=_literal_producer(payload.get('producer')),
        confidence=_confidence(payload.get('confidence')),
        explanation=str(payload.get('explanation', '')),
    )


def verification_evidence_type(kind: str) -> AcceptanceEvidenceType:
    normalized = kind.casefold()
    if 'type' in normalized:
        return 'typecheck'
    if 'test' in normalized or 'pytest' in normalized:
        return 'test_result'
    if 'build' in normalized:
        return 'build'
    if 'lint' in normalized:
        return 'lint'
    if 'smoke' in normalized:
        return 'smoke'
    return 'test_result'


def verification_record_id(
    verification: VerificationEvidence | VerificationRecord,
) -> str:
    command_id = str(getattr(verification, 'command_id', '') or '')
    source_revision = int(
        getattr(
            verification,
            'source_revision',
            getattr(verification, 'workspace_revision', 0),
        )
        or 0
    )
    kind = str(getattr(verification, 'verification_type', 'auto'))
    return ':'.join(part for part in (command_id, kind, str(source_revision)) if part)


def _status_for_evidence(
    criterion_text: str,
    evidence: tuple[AcceptanceEvidence, ...],
    *,
    current_source_revision: int,
) -> AcceptanceStatus:
    if any(item.status == 'blocked' for item in evidence):
        return 'blocked'
    valid = tuple(
        item
        for item in evidence
        if _evidence_current(item, current_source_revision)
    )
    if not valid:
        return 'pending'
    required = _required_evidence_types(criterion_text)
    if required:
        present = {item.evidence_type for item in valid}
        if present & required:
            return 'satisfied'
        return 'partially_satisfied'
    if any(item.status == 'satisfied' for item in valid):
        return 'satisfied'
    if any(
        item.evidence_type
        in {
            'source_change',
            'configuration',
            'review',
            'symbol_evidence',
            'runtime_integration',
        }
        for item in valid
    ):
        return 'satisfied'
    return 'partially_satisfied'


def _required_evidence_types(text: str) -> set[AcceptanceEvidenceType]:
    normalized = text.casefold()
    required: set[AcceptanceEvidenceType] = set()
    if 'typecheck' in normalized or 'type check' in normalized or '类型' in normalized:
        required.add('typecheck')
    if 'build' in normalized or '构建' in normalized:
        required.add('build')
    if 'lint' in normalized:
        required.add('lint')
    if 'smoke' in normalized or 'smoke test' in normalized:
        required.add('smoke')
    if 'test' in normalized or 'pytest' in normalized or '测试' in normalized:
        required.add('test_result')
    if 'verification' in normalized or 'verified' in normalized or '验证' in normalized:
        required.update({'test_result', 'typecheck', 'build', 'lint', 'smoke'})
    return required


def _criterion_accepts_source_change(text: str) -> bool:
    normalized = text.casefold()
    return any(
        token in normalized
        for token in (
            'diff',
            'change',
            'changed',
            'source',
            'config',
            'workspace',
            'document',
            'doc',
            '源码',
            '代码',
            '配置',
            '文档',
            '变更',
        )
    )


def _criterion_accepts_verification(
    text: str,
    evidence_type: AcceptanceEvidenceType,
) -> bool:
    required = _required_evidence_types(text)
    if required:
        return evidence_type in required
    return False


def _source_evidence_status(text: str) -> AcceptanceStatus:
    return 'partially_satisfied' if _required_evidence_types(text) else 'satisfied'


def _missing_evidence_for(text: str) -> str:
    required = _required_evidence_types(text)
    if required:
        return ' or '.join(sorted(required))
    if _criterion_accepts_source_change(text):
        return 'source_change'
    return 'structured acceptance evidence'


def _evidence_current(
    evidence: AcceptanceEvidence,
    current_source_revision: int,
) -> bool:
    if evidence.evidence_type in {
        'test_result',
        'typecheck',
        'build',
        'lint',
        'smoke',
    }:
        return evidence.source_revision >= current_source_revision
    return True


def _payload_status(
    payload: dict[str, Any],
    evidence_type: AcceptanceEvidenceType,
) -> AcceptanceStatus:
    if 'status' in payload:
        return _literal_status(payload.get('status'))
    if evidence_type == 'manual_limitation':
        return 'partially_satisfied'
    return 'satisfied'


def _string_tuple(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,) if value else ()
    if not isinstance(value, list | tuple):
        return ()
    return tuple(str(item) for item in value if str(item).strip())


def _int_value(value: object, *, fallback: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _confidence(value: object) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _literal_status(value: object) -> AcceptanceStatus:
    return (
        value
        if value in {'pending', 'partially_satisfied', 'satisfied', 'blocked'}
        else 'pending'
    )  # type: ignore[return-value]


def _literal_evidence_type(value: object) -> AcceptanceEvidenceType:
    allowed = {
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
    }
    return (value if value in allowed else 'review')  # type: ignore[return-value]


def _literal_producer(value: object) -> AcceptanceProducer:
    return (
        value if value in {'tool', 'model', 'runtime', 'test'} else 'runtime'
    )  # type: ignore[return-value]


def _criterion_sort_key(value: str) -> tuple[int, str]:
    try:
        return (int(value.rsplit('-', 1)[-1]), value)
    except ValueError:
        return (10_000, value)
