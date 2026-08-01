'''Tests for evidence-backed verification levels.'''

from forge.runtime.browser_smoke import BrowserSmokeResult
from forge.runtime.completion_checker import verification_from_result
from forge.runtime.state import VerificationEvidence
from forge.runtime.synthesis_evidence import (
    CompletionEvidence,
    render_completion_evidence_requirements,
)
from forge.runtime.verification_executor import verification_levels_for
from forge.tools.base import ToolResult


def test_build_only_reports_build_verified_not_runtime_verified() -> None:
    evidence = VerificationEvidence(
        command='npm run build',
        cwd='.',
        exit_code=0,
        duration_seconds=0.2,
        timed_out=False,
        workspace_revision=3,
        verification_type='build',
        verification_levels=('build_verified',),
    )

    assert evidence.build_verified is True
    assert evidence.runtime_verified is False


def test_browser_smoke_evidence_allows_runtime_verified_claim() -> None:
    result = BrowserSmokeResult(
        status='passed',
        url='http://127.0.0.1:43123',
        http_status=200,
        canvas_count=1,
        server_terminated=True,
    )

    evidence = result.to_evidence(source_revision=4, duration_seconds=0.3)

    assert evidence.success is True
    assert evidence.runtime_verified is True
    assert 'browser_smoke_verified' in evidence.verification_levels


def test_executor_levels_survive_tool_result_conversion() -> None:
    levels = verification_levels_for(('typecheck', 'build'), command='npm run build')
    result = ToolResult.ok(
        'passed',
        metadata={
            'verification': True,
            'verification_status': 'passed',
            'command': 'npm run build',
            'cwd': '.',
            'exit_code': 0,
            'duration_seconds': 0.1,
            'timed_out': False,
            'workspace_revision': 2,
            'source_revision': 2,
            'verification_type': 'build',
            'verification_levels': list(levels),
        },
    )

    evidence = verification_from_result(result)

    assert evidence is not None
    assert evidence.verification_levels == (
        'typecheck_verified',
        'build_verified',
    )
    assert evidence.runtime_verified is False


def test_build_only_finalization_states_runtime_limitation() -> None:
    verification = VerificationEvidence(
        command='npm run build',
        cwd='.',
        exit_code=0,
        duration_seconds=0.2,
        timed_out=False,
        workspace_revision=1,
        verification_levels=('build_verified',),
    )
    evidence = CompletionEvidence.from_runtime(
        changed_paths=('src/main.ts',),
        verification=verification,
    )

    rendered = render_completion_evidence_requirements(evidence)

    assert 'browser runtime and interaction behavior have not been demonstrated' in rendered
