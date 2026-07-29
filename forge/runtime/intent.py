'''Task-envelope inference used by the Agent Loop controller.'''

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Literal, Protocol


TurnKind = Literal[
    'answer',
    'inspect',
    'plan',
    'implement',
    'fix',
    'refactor',
    'verify',
    'status',
]
CompletionContract = Literal['none', 'inspection', 'change', 'verified_change']
InitialPhase = Literal['answering', 'exploring', 'planning', 'implementing']
InitialToolSurface = Literal['none', 'read_only', 'all']
IntentConfidence = Literal['low', 'medium', 'high']
VerificationPolicyKind = Literal['none', 'optional', 'required']
SemanticClassificationMode = Literal['not_needed', 'recommended']


@dataclass(frozen=True, slots=True)
class TurnIntent:
    kind: TurnKind
    confidence: IntentConfidence
    reason: str


@dataclass(frozen=True, slots=True)
class VerificationPolicy:
    kind: VerificationPolicyKind = 'none'
    required: bool = False
    allow_manual_summary: bool = True
    preferred_commands: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ModelBudget:
    max_semantic_classification_calls: int = 0
    max_planning_calls: int = 0


@dataclass(frozen=True, slots=True)
class ToolBudget:
    initial_surface: InitialToolSurface = 'read_only'
    max_initial_read_calls: int | None = None
    max_recovery_read_calls: int = 1


class SemanticTaskClassifier(Protocol):
    '''Optional semantic classifier used only after low-confidence routing.'''

    def classify_task(self, prompt: str, baseline: 'TaskContract') -> 'TaskContract':
        ...


@dataclass(frozen=True, slots=True)
class TaskContract:
    '''Structured task envelope with backward-compatible contract fields.'''

    intent: TurnIntent
    requires_change: bool
    requires_plan: bool
    completion_contract: CompletionContract
    initial_phase: InitialPhase
    initial_tool_surface: InitialToolSurface
    kind: TurnKind | None = None
    goal: str = ''
    deliverables: tuple[str, ...] = ()
    acceptance_criteria: tuple[str, ...] = ()
    context_hints: tuple[str, ...] = ()
    allowed_paths: tuple[str, ...] = ()
    forbidden_paths: tuple[str, ...] = ()
    verification_policy: VerificationPolicy = field(
        default_factory=VerificationPolicy
    )
    model_budget: ModelBudget = field(default_factory=ModelBudget)
    tool_budget: ToolBudget | None = None
    confidence: IntentConfidence | None = None
    semantic_classification: SemanticClassificationMode = 'not_needed'

    def __post_init__(self) -> None:
        if self.kind is None:
            object.__setattr__(self, 'kind', self.intent.kind)
        if not self.goal:
            object.__setattr__(self, 'goal', self.intent.reason)
        if self.confidence is None:
            object.__setattr__(self, 'confidence', self.intent.confidence)
        if self.tool_budget is None:
            object.__setattr__(
                self,
                'tool_budget',
                ToolBudget(initial_surface=self.initial_tool_surface),
            )

    @property
    def requires_inspection_evidence(self) -> bool:
        return self.completion_contract == 'inspection'

    @property
    def is_low_confidence(self) -> bool:
        return self.intent.confidence == 'low'


_CHANGE_VERBS_ZH = (
    '修复|修好|解决|修改|改|实现|实施|执行|落地|处理|新增|添加|'
    '新建|删除|移除|创建|编写|写入|重写|重构|优化|更新|调整|调高|'
    '调低|改进|完成|替换|继续|开始|完善|开发'
)
_DIRECT_CHANGE_ZH = re.compile(
    rf'^\s*(?:(?:请你?|帮我|麻烦你?|你直接|直接)\s*)?'
    rf'(?:{_CHANGE_VERBS_ZH})'
)
_SCOPED_CHANGE_ZH = re.compile(
    rf'(?:帮我|请你|麻烦你|需要你|我希望你|我想让你|你直接)'
    rf'[^，。；！？\n]{{0,40}}(?:{_CHANGE_VERBS_ZH})'
)
_OBJECT_CHANGE_ZH = re.compile(
    rf'(?:把|将)\s*[^，。；！？\n]{{1,60}}(?:{_CHANGE_VERBS_ZH})'
)
_COMBINED_CHANGE_ZH = re.compile(
    rf'(?:检查|排查|分析|定位)'
    rf'[^，。；！？\n]{{0,30}}(?:并|然后|后)'
    rf'[^，。；！？\n]{{0,20}}(?:{_CHANGE_VERBS_ZH})'
)
_PRIORITY_FIX_ZH = re.compile(
    rf'按\s*(?:最高)?\s*优先级\s*[Pp]0\b[^，。；！？\n]{{0,20}}(?:进行|执行|开始|实施|修复|处理|解决|优化|完成|实现|修改|改|新增|添加|重写|重构|落地)'
)
_EXECUTE_PLAN_ZH = re.compile(
    r'(?:按|按照|根据).{0,40}(?:方案|计划|上述|刚才).{0,20}'
    r'(?:执行|实施|实现|落地)'
)
_TASK_SPEC_CHANGE_ZH = re.compile(
    r'(?:按|按照|根据).{0,40}(?:task\.md|任务|需求|说明).{0,50}'
    r'(?:完善|实现|完成|继续|优化|开发|落地|改进|修改|补齐)'
)
_START_TASK_WORK_ZH = re.compile(
    r'(?:阅读|读取|查看|明确|理解).{0,40}(?:任务|task\.md|需求)'
    r'.{0,30}(?:后|并|然后|开始).{0,20}'
    r'(?:开始工作|开始实现|执行|实施|实现|落地|完成)'
)
_NEGATED_CHANGE_ZH = re.compile(
    rf'(?:不要|别|无需|不用|暂时不|先不|禁止)'
    rf'[^，。；！？\n]{{0,30}}(?:{_CHANGE_VERBS_ZH})'
)
_READ_ONLY_ZH = re.compile(
    r'(?:^\s*(?:为什么|为何|如何|怎么|(?:帮我|请你?)?'
    r'(?:解释|说明|介绍)|查看|告诉我|'
    r'列出|总结|回顾|分析)|'
    r'(?:清单|列表)|'
    r'(?:出|给|给出|制定|写|编写).{0,30}'
    r'(?:清单|列表|方案|计划|建议|规划|roadmap)|'
    r'(?:修复|改动|修改|优化).{0,16}'
    r'(?:清单|列表|方案|计划|建议|规划)|'
    r'(?:完成|实现|修复|更新|修改|优化|开始|继续)(?:了)?'
    r'(?:吗|没有|了吗|没|呢)|'
    r'(?:方案|计划|建议)(?:是什么|有哪些|怎么样|呢|吗)|'
    r'(?:优化|修改).{0,12}(?:方案|计划|建议)|'
    r'^\s*继续(?:解释|介绍|说明|分析|查看|讨论|回答)|'
    r'(?:更新|介绍|查看|告诉我).{0,12}(?:进度|状态|情况)|'
    r'(?:给出|制定|写|编写).{0,20}(?:方案|计划|建议|plan)|'
    r'我再决定|先进行规划)'
)
_CLAUSE_SPLIT_ZH = re.compile(
    r'[，,。；;！!？?\n]+|(?:然后|接着|随后|并(?:且)?)'
)

_DIRECT_CHANGE_EN = re.compile(
    r'^\s*'
    r'(?:(?:please|kindly)\s+)?'
    r'(?:fix|implement|modify|update|add|remove|delete|create|write|'
    r'make|build|complete|refactor|optimize|change|resolve|rewrite|'
    r'execute|apply|continue|start)\b',
    re.IGNORECASE,
)
_REQUESTED_CHANGE_EN = re.compile(
    r'\b(?:'
    r'(?:can|could|would)\s+you\s+(?:please\s+)?|'
    r'help\s+me\s+|'
    r'i\s+need\s+you\s+to\s+'
    r')'
    r'(?:fix|implement|modify|update|add|remove|delete|create|write|'
    r'make|build|complete|refactor|optimize|change|resolve|rewrite|'
    r'execute|apply|continue|start)\b',
    re.IGNORECASE,
)
_COMBINED_CHANGE_EN = re.compile(
    r'(?:inspect|review|investigate|analyze|find)'
    r'.{0,40}\b(?:and|then)\b.{0,30}'
    r'(?:fix|modify|change|resolve|implement|rewrite)\b',
    re.IGNORECASE,
)
_NEGATED_CHANGE_EN = re.compile(
    r'(?:do\s+not|don.t|without|no\s+need\s+to|must\s+not)'
    r'.{0,40}'
    r'(?:fix|modify|change|write|implement|update|edit|create|apply)',
    re.IGNORECASE,
)
_READ_ONLY_EN = re.compile(
    r'(?:^\s*(?:why|how|what|explain|describe|tell\s+me|show\s+me|'
    r'list|summarize|review|analyze|inspect)\b|'
    r'\b(?:p0/p1/p2|priority|priorities|checklist|roadmap)\b|'
    r'\bupdate\s+me\b|'
    r'\b(?:status|progress)\b|'
    r'\b(?:write|create|draft|give|provide)\b.{0,30}'
    r'\b(?:plan|proposal|suggestion|explanation|checklist|roadmap)\b|'
    r'\b(?:fix|change|edit|implementation)\b.{0,20}'
    r'\b(?:plan|proposal|suggestion|checklist|roadmap)\b)',
    re.IGNORECASE,
)
_STATUS_ZH = re.compile(r'(?:进度|状态|情况|完成了吗|完成了?没有)')
_STATUS_EN = re.compile(r'\b(?:status|progress|update me|done yet)\b', re.IGNORECASE)
_PLAN_ZH = re.compile(r'(?:方案|计划|建议|规划|清单|列表|checklist|roadmap)', re.IGNORECASE)
_PLAN_EN = re.compile(r'\b(?:plan|proposal|suggestion|checklist|roadmap)\b', re.IGNORECASE)
_INSPECT_ZH = re.compile(
    r'^\s*(?:只\s*)?(?:查看|列出|分析|审计|检查|阅读|读取|总结|扫描)'
)
_INSPECT_EN = re.compile(r'^\s*(?:inspect|review|analyze|understand|list|show|read|scan|summarize)\b', re.IGNORECASE)
_FIX_ZH = re.compile(r'(?:修复|修好|解决|bug|报错|错误|失败)')
_FIX_EN = re.compile(r'\b(?:fix|resolve|bug|error|failure|failing|broken)\b', re.IGNORECASE)
_REFACTOR_ZH = re.compile(r'(?:重构|迁移|架构)')
_REFACTOR_EN = re.compile(r'\b(?:refactor|migrate|architecture)\b', re.IGNORECASE)
_VERIFY_ZH = re.compile(r'(?:验证|测试|构建|跑测试|检查构建)')
_VERIFY_EN = re.compile(r'\b(?:verify|test|build|lint|typecheck|type-check)\b', re.IGNORECASE)
_CLAUSE_SPLIT_EN = re.compile(
    r'[\n!?,;]+|\b(?:then|and\s+then|however|but)\b',
    re.IGNORECASE,
)
_PATH_HINT = re.compile(
    r'(?<![\w@.-])(?:@)?([A-Za-z0-9_.-]+(?:[\\/][A-Za-z0-9_.-]+)+|'
    r'[A-Za-z0-9_.-]+\.(?:py|js|ts|tsx|jsx|md|json|toml|yaml|yml|css|'
    r'html|rs|go|java|kt|c|cc|cpp|h|hpp|cs|php|rb|sh|ps1|sql))'
)
_MULTI_MODULE_SCOPE = re.compile(
    r'(?:多模块|多个模块|跨模块|整体|系统|架构|architecture|multi[- ]module)',
    re.IGNORECASE,
)


def infer_change_required(prompt: str) -> bool:
    '''Return true only for high-confidence requests to change the workspace.

    This is an execution-contract boundary, not a semantic task classifier:
    it decides whether an empty Diff may satisfy the turn, never what code the
    model should write.
    '''
    text = prompt.strip()
    if not text:
        return False
    clauses = [
        clause.strip()
        for part in _CLAUSE_SPLIT_ZH.split(text)
        for clause in _CLAUSE_SPLIT_EN.split(part)
        if clause.strip()
    ]
    for clause in clauses:
        if _NEGATED_CHANGE_ZH.search(clause):
            continue
        if _NEGATED_CHANGE_EN.search(clause):
            continue
        if (
            _COMBINED_CHANGE_ZH.search(clause)
            or _COMBINED_CHANGE_EN.search(clause)
            or _PRIORITY_FIX_ZH.search(clause)
            or _EXECUTE_PLAN_ZH.search(clause)
            or _TASK_SPEC_CHANGE_ZH.search(clause)
            or _START_TASK_WORK_ZH.search(clause)
        ):
            return True
        if _READ_ONLY_ZH.search(clause) or _READ_ONLY_EN.search(clause):
            continue
        if any(
            pattern.search(clause) is not None
            for pattern in (
                _DIRECT_CHANGE_ZH,
                _SCOPED_CHANGE_ZH,
                _OBJECT_CHANGE_ZH,
                _DIRECT_CHANGE_EN,
                _REQUESTED_CHANGE_EN,
            )
        ):
            return True
    return False


def infer_task_contract(
    prompt: str,
    *,
    interaction_mode: str = 'auto',
    workspace_available: bool = True,
    policy_requires_change: bool = False,
) -> TaskContract:
    '''Infer a conservative executable contract for one user turn.'''
    if interaction_mode == 'plan':
        return _task_contract(
            prompt,
            intent=TurnIntent('plan', 'high', 'explicit plan mode'),
            requires_change=False,
            requires_plan=True,
            completion_contract='none',
            initial_phase='planning',
            initial_tool_surface='read_only',
        )
    if interaction_mode == 'code':
        return _task_contract(
            prompt,
            intent=TurnIntent('implement', 'high', 'explicit code mode'),
            requires_change=workspace_available,
            requires_plan=False,
            completion_contract='change' if workspace_available else 'none',
            initial_phase='implementing',
            initial_tool_surface='all',
        )

    text = prompt.strip()
    inferred_change = infer_change_required(text)
    wants_change = bool(policy_requires_change or inferred_change)
    requires_change = bool(policy_requires_change or (
        workspace_available and inferred_change
    ))
    if wants_change:
        kind: TurnKind = 'implement'
        reason = 'explicit workspace-change request'
        if _FIX_ZH.search(text) or _FIX_EN.search(text):
            kind = 'fix'
            reason = 'explicit fix request'
        elif _REFACTOR_ZH.search(text) or _REFACTOR_EN.search(text):
            kind = 'refactor'
            reason = 'explicit refactor request'
        elif _VERIFY_ZH.search(text) or _VERIFY_EN.search(text):
            kind = 'verify'
            reason = 'change request includes verification language'
        requires_plan = _requires_structured_plan(text)
        return _task_contract(
            prompt,
            intent=TurnIntent(kind, 'high', reason),
            requires_change=requires_change,
            requires_plan=requires_plan,
            completion_contract='change' if requires_change else 'none',
            initial_phase='implementing',
            initial_tool_surface='all',
        )

    if _STATUS_ZH.search(text) or _STATUS_EN.search(text):
        return read_only_contract(
            'status',
            'high',
            'status request',
            prompt=prompt,
        )
    if _PLAN_ZH.search(text) or _PLAN_EN.search(text):
        return _task_contract(
            prompt,
            intent=TurnIntent('plan', 'high', 'plan/checklist request'),
            requires_change=False,
            requires_plan=True,
            completion_contract='none',
            initial_phase='planning',
            initial_tool_surface='read_only',
        )
    if _INSPECT_ZH.search(text) or _INSPECT_EN.search(text):
        return read_only_contract(
            'inspect',
            'medium',
            'inspection request',
            prompt=prompt,
        )
    if _extract_path_hints(text) and (
        _READ_ONLY_ZH.search(text) or _READ_ONLY_EN.search(text)
    ):
        return read_only_contract(
            'inspect',
            'medium',
            'path-scoped inspection request',
            prompt=prompt,
        )
    return read_only_contract(
        'answer',
        'low',
        'default non-change request',
        prompt=prompt,
    )


def refine_task_contract(
    prompt: str,
    contract: TaskContract,
    classifier: SemanticTaskClassifier | None = None,
) -> TaskContract:
    '''Apply optional semantic routing only when deterministic confidence is low.'''
    if (
        classifier is None
        or contract.semantic_classification != 'recommended'
        or contract.model_budget.max_semantic_classification_calls < 1
    ):
        return contract
    return classifier.classify_task(prompt, contract)


def read_only_contract(
    kind: TurnKind,
    confidence: IntentConfidence,
    reason: str,
    *,
    prompt: str = '',
) -> TaskContract:
    completion: CompletionContract = (
        'inspection' if kind == 'inspect' else 'none'
    )
    phase: InitialPhase = 'exploring' if kind == 'inspect' else 'answering'
    surface: InitialToolSurface = 'read_only' if kind in {
        'inspect',
        'plan',
        'status',
    } else 'none'
    return _task_contract(
        prompt,
        intent=TurnIntent(kind, confidence, reason),
        requires_change=False,
        requires_plan=False,
        completion_contract=completion,
        initial_phase=phase,
        initial_tool_surface=surface,
    )


def _task_contract(
    prompt: str,
    *,
    intent: TurnIntent,
    requires_change: bool,
    requires_plan: bool,
    completion_contract: CompletionContract,
    initial_phase: InitialPhase,
    initial_tool_surface: InitialToolSurface,
) -> TaskContract:
    goal = prompt.strip()
    paths = _extract_path_hints(goal)
    verification = _verification_policy(
        completion_contract,
        requires_change=requires_change,
    )
    semantic_classification: SemanticClassificationMode = (
        'recommended' if intent.confidence == 'low' else 'not_needed'
    )
    return TaskContract(
        intent=intent,
        kind=intent.kind,
        goal=goal,
        requires_change=requires_change,
        requires_plan=requires_plan,
        completion_contract=completion_contract,
        initial_phase=initial_phase,
        initial_tool_surface=initial_tool_surface,
        deliverables=_deliverables(intent.kind, requires_change),
        acceptance_criteria=_acceptance_criteria(
            completion_contract,
            requires_change=requires_change,
            requires_plan=requires_plan,
        ),
        context_hints=paths,
        allowed_paths=paths,
        forbidden_paths=(),
        verification_policy=verification,
        model_budget=ModelBudget(
            max_semantic_classification_calls=(
                1 if semantic_classification == 'recommended' else 0
            ),
            max_planning_calls=1 if requires_plan else 0,
        ),
        tool_budget=ToolBudget(initial_surface=initial_tool_surface),
        confidence=intent.confidence,
        semantic_classification=semantic_classification,
    )


def _extract_path_hints(prompt: str) -> tuple[str, ...]:
    paths: list[str] = []
    for match in _PATH_HINT.finditer(prompt):
        path = match.group(1).replace('\\', '/')
        if path not in paths:
            paths.append(path)
    return tuple(paths)


def _requires_structured_plan(prompt: str) -> bool:
    return bool(
        _MULTI_MODULE_SCOPE.search(prompt)
        or re.search(
            r'\b(?:p0|p1|p2|priority|priorities|roadmap)\b',
            prompt,
            re.IGNORECASE,
        )
        or re.search(r'(?:优先级|规划|计划)', prompt)
    )


def _verification_policy(
    completion_contract: CompletionContract,
    *,
    requires_change: bool,
) -> VerificationPolicy:
    if completion_contract == 'verified_change':
        return VerificationPolicy(kind='required', required=True)
    if requires_change:
        return VerificationPolicy(kind='required', required=True)
    if completion_contract == 'inspection':
        return VerificationPolicy(kind='none', required=False)
    return VerificationPolicy(kind='none', required=False)


def _deliverables(
    kind: TurnKind,
    requires_change: bool,
) -> tuple[str, ...]:
    if requires_change:
        return ('workspace changes', 'concise change summary')
    if kind == 'inspect':
        return ('repository-grounded analysis',)
    if kind == 'plan':
        return ('plan or checklist',)
    return ('direct answer',)


def _acceptance_criteria(
    completion_contract: CompletionContract,
    *,
    requires_change: bool,
    requires_plan: bool,
) -> tuple[str, ...]:
    criteria: list[str] = []
    if requires_plan:
        criteria.append('A working plan exists before broad implementation.')
    if requires_change:
        criteria.extend(
            [
                'A task-local workspace diff exists.',
                'Changed paths are relevant to the user goal.',
                'Verification is current or any limitation is stated.',
            ]
        )
    elif completion_contract == 'inspection':
        criteria.extend(
            [
                'The answer is grounded in repository evidence.',
                'No workspace edit is made.',
            ]
        )
    else:
        criteria.append('The response directly addresses the user request.')
    return tuple(criteria)
