'''Evidence matching for final synthesized answers.'''

from __future__ import annotations

from dataclasses import dataclass

from forge.context.working import answer_mentions_any_path
from forge.runtime.state import VerificationEvidence, VerificationLevel


@dataclass(frozen=True, slots=True)
class CompletionEvidence:
    '''Authoritative completion facts available after completion gates pass.'''

    changed_paths: tuple[str, ...] = ()
    verification_command: str = ''
    verification_status: str = ''
    verification_exit_code: int | None = None
    verification_source_revision: int | None = None
    repository_paths: tuple[str, ...] = ()
    verification_levels: tuple[VerificationLevel, ...] = ()

    @classmethod
    def from_runtime(
        cls,
        *,
        changed_paths: tuple[str, ...],
        verification: VerificationEvidence | None,
        repository_paths: tuple[str, ...] = (),
        acceptance_verified: bool = False,
    ) -> 'CompletionEvidence':
        levels = list(
            verification.verification_levels
            if verification is not None
            else ()
        )
        if acceptance_verified:
            levels.append('acceptance_verified')
        return cls(
            changed_paths=changed_paths,
            verification_command=(
                verification.command if verification is not None else ''
            ),
            verification_status=(
                verification.status if verification is not None else ''
            ),
            verification_exit_code=(
                verification.exit_code if verification is not None else None
            ),
            verification_source_revision=(
                verification.bound_source_revision
                if verification is not None
                else None
            ),
            repository_paths=repository_paths,
            verification_levels=tuple(dict.fromkeys(levels)),
        )

    @property
    def has_authoritative_items(self) -> bool:
        return bool(
            self.changed_paths
            or self.verification_command
            or self.repository_paths
        )


def answer_mentions_completion_evidence(
    text: str,
    evidence: CompletionEvidence,
) -> bool:
    '''Return whether final text cites authoritative completion facts.'''
    if evidence.changed_paths and answer_mentions_any_path(
        text,
        evidence.changed_paths,
    ):
        return True
    if evidence.repository_paths and answer_mentions_any_path(
        text,
        evidence.repository_paths,
    ):
        return True
    if evidence.verification_command and _contains_phrase(
        text,
        evidence.verification_command,
    ):
        return True
    if _mentions_verification_result(text, evidence):
        return True
    return False


def completion_evidence_candidates(
    evidence: CompletionEvidence,
) -> tuple[str, ...]:
    candidates: list[str] = []
    candidates.extend(evidence.changed_paths)
    if evidence.verification_command:
        candidates.append(evidence.verification_command)
    if evidence.verification_status:
        candidates.append(f'verification status: {evidence.verification_status}')
    if evidence.verification_exit_code is not None:
        candidates.append(f'exit code: {evidence.verification_exit_code}')
    if evidence.verification_source_revision is not None:
        candidates.append(
            f'source revision: {evidence.verification_source_revision}'
        )
    candidates.extend(evidence.repository_paths)
    return tuple(dict.fromkeys(candidates))


def render_completion_evidence_requirements(
    evidence: CompletionEvidence,
) -> str:
    sections: list[str] = []
    if evidence.changed_paths:
        sections.append(
            'Mention at least one changed path:\n'
            + '\n'.join(f'- {path}' for path in evidence.changed_paths)
        )
    if evidence.verification_command:
        details = [f'- {evidence.verification_command}']
        if evidence.verification_status:
            details.append(f'- status: {evidence.verification_status}')
        if evidence.verification_exit_code is not None:
            details.append(f'- exit code: {evidence.verification_exit_code}')
        if evidence.verification_source_revision is not None:
            details.append(
                f'- source revision: {evidence.verification_source_revision}'
            )
        sections.append('Mention the verification:\n' + '\n'.join(details))
    if evidence.repository_paths:
        sections.append(
            'Repository evidence that may also be cited:\n'
            + '\n'.join(f'- {path}' for path in evidence.repository_paths)
        )
    if evidence.verification_levels:
        sections.append(
            'Do not claim a stronger verification level than these recorded '
            'levels:\n'
            + '\n'.join(f'- {item}' for item in evidence.verification_levels)
        )
    if (
        'build_verified' in evidence.verification_levels
        and 'browser_smoke_verified' not in evidence.verification_levels
    ):
        sections.append(
            'The project passed its build, but browser runtime and interaction '
            'behavior have not been demonstrated.'
        )
    return '\n\n'.join(sections)


def render_repository_evidence_requirements(paths: tuple[str, ...]) -> str:
    if not paths:
        return 'No repository evidence paths were recorded.'
    return (
        'Mention at least one collected repository evidence path:\n'
        + '\n'.join(f'- {path}' for path in paths)
    )


def build_completion_fallback_summary(
    evidence: CompletionEvidence,
) -> str:
    lines = ['Task completed.']
    if evidence.changed_paths:
        lines.extend(['', 'Changed:'])
        lines.extend(f'- {path}' for path in evidence.changed_paths)
    if evidence.verification_command:
        lines.extend(['', 'Verification:', f'- {evidence.verification_command}'])
        if evidence.verification_exit_code is not None:
            lines.append(f'- exit code: {evidence.verification_exit_code}')
        if evidence.verification_status:
            lines.append(f'- status: {evidence.verification_status}')
        if evidence.verification_source_revision is not None:
            lines.append(
                f'- source revision: {evidence.verification_source_revision}'
            )
    lines.extend(['', 'Limitations:'])
    if (
        'build_verified' in evidence.verification_levels
        and 'browser_smoke_verified' not in evidence.verification_levels
    ):
        lines.append(
            '- Browser runtime and interaction behavior were not verified.'
        )
    else:
        lines.append(
            '- No additional semantic or visual verification evidence was recorded.'
        )
    return '\n'.join(lines)


def _contains_phrase(text: str, phrase: str) -> bool:
    normalized_text = ' '.join(text.casefold().replace('\\', '/').split())
    normalized_phrase = ' '.join(phrase.casefold().replace('\\', '/').split())
    return bool(normalized_phrase) and normalized_phrase in normalized_text


def _mentions_verification_result(
    text: str,
    evidence: CompletionEvidence,
) -> bool:
    normalized = ' '.join(text.casefold().split())
    if evidence.verification_exit_code is not None:
        code = evidence.verification_exit_code
        exit_phrases = (
            f'exit code {code}',
            f'exit code: {code}',
            f'exit_code {code}',
            f'exit_code: {code}',
            f'退出码 {code}',
            f'退出码为 {code}',
            f'退出代码 {code}',
            f'退出代码为 {code}',
        )
        if any(phrase in normalized for phrase in exit_phrases):
            return True
    if evidence.verification_status == 'passed':
        return any(
            phrase in normalized
            for phrase in (
                'verification passed',
                'verify passed',
                '验证通过',
                '校验通过',
                '检查通过',
            )
        )
    if evidence.verification_status:
        return f'status: {evidence.verification_status}' in normalized
    return False
