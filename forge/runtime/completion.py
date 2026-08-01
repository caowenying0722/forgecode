'''Deterministic completion checks for code-changing tasks.'''

from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path

from forge.runtime.state import VerificationEvidence
from forge.runtime.workspace import WorkspaceTracker
from forge.runtime.process import run_process


CODE_SUFFIXES = frozenset(
    {
        '.c', '.cc', '.cpp', '.cs', '.go', '.h', '.hpp', '.java', '.js',
        '.jsx', '.kt', '.kts', '.php', '.py', '.rb', '.rs', '.scala',
        '.svelte', '.swift', '.ts', '.tsx', '.vue',
    }
)
PROJECT_VALIDATION_MARKERS = (
    ' test',
    'build',
    'cargo ',
    'cmake',
    'dotnet ',
    'eslint',
    'gradle',
    'jest',
    'lint',
    'make ',
    'mocha',
    'mvn ',
    'mypy',
    'ninja',
    'playwright',
    'pytest',
    'pyright',
    'ruff',
    'tsc',
    'typecheck',
    'unittest',
    'vet',
    'vitest',
)


@dataclass(frozen=True, slots=True)
class TaskPolicy:
    '''Explicit requirements supplied by a caller or evaluation case.'''

    # Deprecated compatibility alias. It is honored by CompletionGate for
    # already change-like evaluations, but Conversation no longer uses it to
    # infer that every new turn must edit the workspace.
    require_changes: bool = False
    require_changes_for_change_turns: bool = True
    require_verification_for_change_turns: bool = True
    forbid_changes_for_read_only_turns: bool = True
    require_verification: bool = False
    require_change_verification: bool = True
    require_diff_review: bool = False
    allowed_paths: tuple[str, ...] = ()
    forbidden_paths: tuple[str, ...] = (
        'tests/hidden/**',
        '**/tests/hidden/**',
    )


@dataclass(frozen=True, slots=True)
class CompletionDecision:
    allowed: bool
    reasons: tuple[str, ...] = ()


class CompletionGate:
    '''Reject final answers that violate the active task policy.'''

    def __init__(
        self,
        root: Path,
        policy: TaskPolicy | None = None,
    ) -> None:
        self.root = root.resolve()
        self.policy = policy or TaskPolicy()

    async def evaluate(
        self,
        tracker: WorkspaceTracker,
        verification: VerificationEvidence | None,
        *,
        mutation_attempted: bool,
        reviewed_paths: set[str] | None = None,
    ) -> CompletionDecision:
        changed_paths = tracker.changed_paths
        # ``mutation_attempted`` is the caller's turn-local declaration that
        # this is a change turn. The deprecated policy.require_changes flag is
        # intentionally not allowed to turn read-only/advisory turns into code
        # tasks by itself.
        code_task = (
            mutation_attempted
            or self.policy.require_verification
            or bool(changed_paths)
            or verification is not None
        )
        if not code_task:
            return CompletionDecision(allowed=True)

        reasons: list[str] = []
        if not tracker.available:
            reasons.append(
                'Git workspace tracking is unavailable for this task.'
            )
        if (
            mutation_attempted
            and self.policy.require_changes_for_change_turns
            and not changed_paths
        ):
            reasons.append(
                'The task requires a code change, but the final Diff is empty.'
            )

        reasons.extend(self._path_violations(changed_paths))
        if (
            verification is not None
            and verification.bound_source_revision
            == _tracker_source_revision(tracker)
            and verification.verification_side_effect_paths
        ):
            reasons.append(
                'Verification modified undeclared source or workspace paths: '
                + ', '.join(
                    verification.verification_side_effect_paths
                )
            )

        verification_required = self.policy.require_verification or (
            self.policy.require_verification_for_change_turns
            and self.policy.require_change_verification
            and (mutation_attempted or bool(changed_paths))
        )
        if verification_required:
            if verification is None:
                reasons.append(
                    'The current code has not been verified with the verify tool.'
                )
            elif not verification.success:
                if verification.status == 'invalid':
                    reasons.append(
                        'The latest verification command was invalid; use a '
                        'discovered non-interactive project validation command.'
                    )
                elif verification.status == 'unavailable':
                    reasons.append(
                        'Project verification is unavailable for the current '
                        'workspace.'
                    )
                elif verification.status == 'timed_out':
                    reasons.append(
                        'The latest verification timed out.'
                    )
                else:
                    reasons.append(
                        f'The latest verification failed with exit code '
                        f'{verification.exit_code}.'
                    )
            elif verification.bound_source_revision != _tracker_source_revision(tracker):
                reasons.append(
                    'The code changed after verification; run verify again for '
                    f'source revision {_tracker_source_revision(tracker)}.'
                )
            elif self._needs_project_validation(changed_paths) and not (
                command_is_project_validation(verification.command)
            ):
                reasons.append(
                    'Source code changed, but the verification command does '
                    'not run a recognizable test, build, lint, or type-check '
                    'for this repository. A whitespace check, syntax-only '
                    'check, or arbitrary command is insufficient.'
                )
        elif (
            verification is not None
            and verification.bound_source_revision == _tracker_source_revision(tracker)
            and not verification.success
        ):
            reasons.append(
                f'The latest verification failed with status '
                f'{verification.status}.'
            )

        if (
            self.policy.require_diff_review
            and changed_paths
            and reviewed_paths is not None
        ):
            reviewed = {path.replace('\\', '/') for path in reviewed_paths}
            unreviewed = tuple(
                path
                for path in changed_paths
                if path.replace('\\', '/') not in reviewed
            )
            if unreviewed:
                reasons.append(
                    'The final Diff has not been reviewed for: '
                    + ', '.join(unreviewed)
                    + '. Run git_diff for these current-revision paths.'
                )

        if changed_paths and tracker.available:
            reasons.extend(
                await self._diff_check_reasons(changed_paths)
            )

        return CompletionDecision(
            allowed=not reasons,
            reasons=tuple(dict.fromkeys(reasons)),
        )

    async def _diff_check_reasons(
        self,
        changed_paths: tuple[str, ...],
    ) -> list[str]:
        tracked: list[str] = []
        untracked: list[str] = []
        for path in changed_paths:
            listed = await run_process(
                ['git', 'ls-files', '--error-unmatch', '--', path],
                cwd=self.root,
                timeout_seconds=30,
            )
            if listed.exit_code == 0:
                tracked.append(path)
            elif (self.root / path).is_file():
                untracked.append(path)

        reasons: list[str] = []
        if tracked:
            diff_check = await run_process(
                [
                    'git',
                    'diff',
                    'HEAD',
                    '--check',
                    '--',
                    *tracked,
                ],
                cwd=self.root,
                timeout_seconds=30,
            )
            if diff_check.exit_code != 0:
                reasons.append(
                    'git diff --check found a deterministic Patch error.'
                )

        for path in untracked:
            diff_check = await run_process(
                [
                    'git',
                    'diff',
                    '--no-index',
                    '--check',
                    '--',
                    '/dev/null',
                    path,
                ],
                cwd=self.root,
                timeout_seconds=30,
            )
            if (
                diff_check.exit_code not in {0, 1}
                or diff_check.stdout.strip()
            ):
                reasons.append(
                    'Git whitespace checking found a deterministic Patch '
                    f'error in untracked file: {path}.'
                )
        return reasons

    def _path_violations(self, paths: tuple[str, ...]) -> list[str]:
        reasons: list[str] = []
        forbidden = tuple(
            path
            for path in paths
            if matches_any(path, self.policy.forbidden_paths)
        )
        if forbidden:
            reasons.append(
                'Forbidden paths were modified: ' + ', '.join(forbidden)
            )

        if self.policy.allowed_paths:
            outside = tuple(
                path
                for path in paths
                if not matches_any(path, self.policy.allowed_paths)
            )
            if outside:
                reasons.append(
                    'Paths outside the allowed scope were modified: '
                    + ', '.join(outside)
                )
        return reasons

    def _needs_project_validation(self, paths: tuple[str, ...]) -> bool:
        if not any(Path(path).suffix.casefold() in CODE_SUFFIXES for path in paths):
            return False
        markers = (
            'Cargo.toml',
            'go.mod',
            'package.json',
            'pom.xml',
            'pyproject.toml',
            'setup.cfg',
            'tox.ini',
        )
        return any((self.root / marker).exists() for marker in markers) or any(
            (self.root / directory).is_dir()
            for directory in ('test', 'tests', '__tests__')
        )


def command_is_project_validation(command: str) -> bool:
    normalized = f' {command.strip().casefold()} '
    return any(marker in normalized for marker in PROJECT_VALIDATION_MARKERS)


def _tracker_source_revision(tracker: WorkspaceTracker) -> int:
    return int(getattr(tracker, 'source_revision', tracker.revision))


def matches_any(path: str, patterns: tuple[str, ...]) -> bool:
    candidate = path.replace('\\', '/')
    return any(fnmatchcase(candidate, pattern) for pattern in patterns)
