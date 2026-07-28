'''Keep user-level ForgeCode state isolated from the developer machine.'''

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolated_forge_home(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv('FORGE_HOME', str(tmp_path / 'forge-home'))
