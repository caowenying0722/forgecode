'''Safe runtime identity diagnostics for CLI and status surfaces.'''

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, distribution
import json
from pathlib import Path
import re
import subprocess
import sys

from forge import __version__


RUNTIME_SCHEMA_VERSION = 1
_COMMIT_PATTERN = re.compile(r'[0-9a-f]{40}')


@dataclass(frozen=True, slots=True)
class RuntimeDiagnostics:
    '''Non-secret identity of the ForgeCode process and active workspace.'''

    version: str
    git_commit: str
    editable_install: bool
    runtime_schema_version: int
    python_executable: str
    package_path: str
    workspace: str
    actual_cwd: str

    def lines(self) -> tuple[str, ...]:
        return (
            f'ForgeCode version: {self.version}',
            f'Git commit: {self.git_commit}',
            f'Editable install: {"yes" if self.editable_install else "no"}',
            f'Runtime schema version: {self.runtime_schema_version}',
            f'Python executable: {self.python_executable}',
            f'ForgeCode package path: {self.package_path}',
            f'Workspace: {self.workspace}',
            f'Actual cwd: {self.actual_cwd}',
        )


def collect_runtime_diagnostics(
    workspace: Path,
    *,
    actual_cwd: Path | None = None,
) -> RuntimeDiagnostics:
    '''Collect deterministic process identity without reading credentials.'''
    package_path = Path(__file__).resolve().parents[1] / '__init__.py'
    source_root = package_path.parent.parent
    return RuntimeDiagnostics(
        version=__version__,
        git_commit=_git_commit(source_root),
        editable_install=_is_editable_install(),
        runtime_schema_version=RUNTIME_SCHEMA_VERSION,
        python_executable=str(Path(sys.executable).resolve()),
        package_path=str(package_path),
        workspace=str(workspace.resolve()),
        actual_cwd=str((actual_cwd or Path.cwd()).resolve()),
    )


def _git_commit(source_root: Path) -> str:
    try:
        completed = subprocess.run(
            ['git', '-C', str(source_root), 'rev-parse', 'HEAD'],
            capture_output=True,
            check=False,
            encoding='utf-8',
            errors='replace',
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return 'unknown'
    candidate = completed.stdout.strip().lower()
    if completed.returncode != 0 or _COMMIT_PATTERN.fullmatch(candidate) is None:
        return 'unknown'
    return candidate


def _is_editable_install() -> bool:
    try:
        direct_url = distribution('forge-code').read_text('direct_url.json')
    except (PackageNotFoundError, OSError):
        return False
    if not direct_url:
        return False
    try:
        payload = json.loads(direct_url)
    except (TypeError, json.JSONDecodeError):
        return False
    directory_info = payload.get('dir_info')
    return bool(
        isinstance(directory_info, dict)
        and directory_info.get('editable') is True
    )
