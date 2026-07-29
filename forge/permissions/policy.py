'''Deterministic permission modes, rules, approvals, and audit decisions.'''

from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatch
import json
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal


PermissionMode = Literal[
    'plan',
    'supervised',
    'auto',
]
PermissionAction = Literal['allow', 'ask', 'deny']
PermissionScope = Literal['user', 'project', 'session']
ApprovalChoice = Literal[
    'allow_once',
    'allow_session',
    'allow_project',
    'deny',
]


@dataclass(frozen=True, slots=True)
class PermissionRequest:
    tool_name: str
    capability: str
    risk: Literal['low', 'medium', 'high', 'critical']
    targets: tuple[str, ...] = ()
    reason: str = ''
    preview: str = ''
    hard_deny: bool = False

    @property
    def signature(self) -> str:
        target = self.targets[0] if len(self.targets) == 1 else '*'
        return f'{self.capability}:{target}'


@dataclass(frozen=True, slots=True)
class PermissionRule:
    action: PermissionAction
    capability: str = '*'
    target: str = '*'
    scope: PermissionScope = 'session'

    def __post_init__(self) -> None:
        if self.action not in {'allow', 'ask', 'deny'}:
            raise ValueError(f'Invalid permission action: {self.action}')
        if self.scope not in {'user', 'project', 'session'}:
            raise ValueError(f'Invalid permission scope: {self.scope}')

    def matches(self, request: PermissionRequest) -> bool:
        if not fnmatch(request.capability, self.capability):
            return False
        targets = request.targets or ('',)
        return any(fnmatch(target, self.target) for target in targets)

    @property
    def specificity(self) -> tuple[int, int]:
        fixed_target = len(self.target.replace('*', '').replace('?', ''))
        fixed_capability = len(self.capability.replace('*', ''))
        return fixed_target, fixed_capability


@dataclass(frozen=True, slots=True)
class ApprovalResponse:
    choice: ApprovalChoice
    reason: str = ''


@dataclass(frozen=True, slots=True)
class PermissionDecision:
    action: Literal['allow', 'deny']
    request: PermissionRequest
    reason: str
    source: str


ApprovalHandler = Callable[
    [PermissionRequest],
    Awaitable[ApprovalResponse],
]


class PermissionManager:
    '''Evaluate hard rules, stored rules, mode defaults, and user approvals.'''

    def __init__(
        self,
        root: Path,
        *,
        mode: PermissionMode = 'auto',
        approval_handler: ApprovalHandler | None = None,
        user_path: Path | None = None,
    ) -> None:
        self.root = root.resolve()
        self.mode = mode
        self.approval_handler = approval_handler
        self.user_path = (
            user_path or Path.home() / '.forge' / 'permissions.json'
        )
        self.project_path = self.root / '.forge' / 'permissions.json'
        self.session_rules: list[PermissionRule] = []
        self.user_rules = self._load_rules(self.user_path, 'user')
        self.project_rules = self._load_rules(self.project_path, 'project')

    def set_mode(self, mode: str) -> PermissionMode:
        normalized = normalize_permission_mode(mode)
        if normalized != self.mode:
            self.session_rules.clear()
        self.mode = normalized
        return self.mode

    def describe(self) -> str:
        return render_permission_notice(self.mode)

    async def authorize(
        self,
        request: PermissionRequest,
    ) -> PermissionDecision:
        if request.hard_deny:
            return self._decision(
                'deny',
                request,
                request.reason or 'This operation is forbidden.',
                'hard_deny',
            )

        matching = [
            rule
            for rule in (
                *self.user_rules,
                *self.project_rules,
                *self.session_rules,
            )
            if rule.matches(request)
        ]
        denied = [rule for rule in matching if rule.action == 'deny']
        if denied:
            return self._decision(
                'deny',
                request,
                'A matching deny rule applies.',
                'rule',
            )
        if matching:
            scope_rank = {'user': 0, 'project': 1, 'session': 2}
            rule = max(
                matching,
                key=lambda item: (
                    *item.specificity,
                    scope_rank[item.scope],
                ),
            )
            if rule.action == 'allow':
                return self._decision(
                    'allow',
                    request,
                    'Allowed by stored rule.',
                    rule.scope,
                )
            return await self._request_approval(request, source=rule.scope)

        default = self._mode_default(request)
        if default == 'allow':
            return self._decision(
                'allow',
                request,
                'Allowed by permission mode.',
                self.mode,
            )
        if default == 'deny':
            return self._decision(
                'deny',
                request,
                'Denied by permission mode.',
                self.mode,
            )
        return await self._request_approval(request, source=self.mode)

    def _mode_default(
        self,
        request: PermissionRequest,
    ) -> PermissionAction:
        read_only = request.capability in {
            'file.read',
            'repository.read',
        }
        if self.mode == 'plan':
            return 'allow' if read_only else 'deny'
        if self.mode == 'supervised':
            return 'allow' if read_only else 'ask'
        if request.risk == 'low':
            return 'allow'
        return 'ask'

    async def _request_approval(
        self,
        request: PermissionRequest,
        *,
        source: str,
    ) -> PermissionDecision:
        if self.approval_handler is None:
            return self._decision(
                'deny',
                request,
                (
                    'Approval is required but no interactive approval handler '
                    'is available.'
                ),
                'approval_unavailable',
            )
        response = await self.approval_handler(request)
        if request.capability == 'file.delete' and (
            response.choice == 'allow_project'
            or (
                response.choice == 'allow_session'
                and len(request.targets) != 1
            )
        ):
            response = ApprovalResponse('allow_once', response.reason)
        if response.choice == 'allow_session':
            self.session_rules.append(
                PermissionRule(
                    'allow',
                    request.capability,
                    self._rule_target(request),
                    'session',
                )
            )
        elif response.choice == 'allow_project':
            rule = PermissionRule(
                'allow',
                request.capability,
                self._rule_target(request),
                'project',
            )
            self.project_rules.append(rule)
            self._save_rules(self.project_path, self.project_rules)
        if response.choice == 'deny':
            return self._decision(
                'deny',
                request,
                response.reason or 'Denied by user.',
                'user',
            )
        return self._decision(
            'allow',
            request,
            response.reason or 'Approved by user.',
            'user',
        )

    @staticmethod
    def _rule_target(request: PermissionRequest) -> str:
        return request.targets[0] if len(request.targets) == 1 else '*'

    def _decision(
        self,
        action: Literal['allow', 'deny'],
        request: PermissionRequest,
        reason: str,
        source: str,
    ) -> PermissionDecision:
        return PermissionDecision(action, request, reason, source)

    @staticmethod
    def _load_rules(path: Path, scope: PermissionScope) -> list[PermissionRule]:
        if not path.is_file():
            return []
        try:
            raw = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            return []
        if not isinstance(raw, list):
            return []
        rules: list[PermissionRule] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            try:
                rules.append(
                    PermissionRule(
                        action=str(item.get('action', 'ask')),
                        capability=str(item.get('capability', '*')),
                        target=str(item.get('target', '*')),
                        scope=scope,
                    )
                )
            except ValueError:
                continue
        return rules

    @staticmethod
    def _save_rules(path: Path, rules: list[PermissionRule]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: list[dict[str, str]] = [
            {
                'action': rule.action,
                'capability': rule.capability,
                'target': rule.target,
            }
            for rule in rules
            if rule.scope == 'project'
        ]
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + '\n',
            encoding='utf-8',
        )


def normalize_permission_mode(mode: str) -> PermissionMode:
    normalized = mode.strip().casefold()
    if normalized not in {'plan', 'supervised', 'auto'}:
        raise ValueError(
            'Permission mode must be one of: plan, supervised, auto.'
        )
    return normalized  # type: ignore[return-value]


def render_permission_notice(mode: str) -> str:
    normalized = normalize_permission_mode(mode)
    if normalized == 'auto':
        return (
            'Permission: Auto. Read-only operations, workspace edits, and '
            'low-risk local commands run automatically; high-risk commands '
            'ask for confirmation.'
        )
    if normalized == 'plan':
        return (
            'Permission: Plan. Read-only repository inspection is allowed; '
            'writes and process commands are blocked.'
        )
    return (
        'Permission: Supervised. Read-only tools may run directly; writes '
        'and process commands ask for confirmation before execution.'
    )
