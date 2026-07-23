'''Tests for repository instructions and durable memory.'''

from pathlib import Path

import pytest

from forge.context.repository import MemoryStore, RepositoryContext


def test_memory_round_trip_index_and_forget(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)

    record = store.remember(
        'testing',
        'Run uv run pytest after code changes.',
        description='Project test command',
    )

    assert record.path.exists()
    assert record.source == 'manual'
    assert record.created_at
    assert record.updated_at
    text = record.path.read_text(encoding='utf-8')
    assert 'source: "manual"' in text
    assert 'created_at:' in text
    assert 'updated_at:' in text
    assert store.get('testing') is not None
    assert MemoryStore(tmp_path).get('testing') is not None
    assert '[testing](testing.md)' in store.index_path.read_text(encoding='utf-8')
    assert store.forget('testing') is True
    assert store.get('testing') is None


def test_memory_update_preserves_created_at_and_updates_metadata(
    tmp_path: Path,
) -> None:
    store = MemoryStore(tmp_path)
    created = store.create(
        'testing',
        'Run pytest.',
        source='manual',
    )

    updated = store.update(
        'testing',
        'Run pytest -q.',
        source='model_memory_tool',
    )

    assert updated.created_at == created.created_at
    assert updated.updated_at >= created.updated_at
    assert updated.source == 'model_memory_tool'
    assert updated.content == 'Run pytest -q.'


def test_legacy_memory_without_metadata_still_parses(tmp_path: Path) -> None:
    directory = tmp_path / '.forge' / 'memory'
    directory.mkdir(parents=True)
    (directory / 'legacy.md').write_text(
        '---\n'
        'name: "legacy"\n'
        'description: "Old memory"\n'
        'type: "project"\n'
        '---\n\n'
        'Legacy content.\n',
        encoding='utf-8',
    )

    record = MemoryStore(tmp_path).get('legacy')

    assert record is not None
    assert record.source == 'legacy'
    assert record.created_at
    assert record.updated_at


def test_memory_selection_is_relevant_and_bounded(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path, max_selected=1, max_file_characters=40)
    store.remember('testing', 'Use pytest for calculator verification.')
    store.remember('style', 'Use single quotes in Python source.')

    selection = store.select('How should I run calculator tests?')

    assert [record.name for record in selection.records] == ['testing']
    assert 'pytest' in selection.rendered
    assert 'single quotes' not in selection.rendered


def test_memory_rejects_secrets(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)

    with pytest.raises(ValueError, match='secret'):
        store.remember('credentials', 'API_KEY=sk-supersecret123')


def test_repository_context_loads_rules_and_relevant_memory(
    tmp_path: Path,
) -> None:
    (tmp_path / 'AGENTS.md').write_text('Always run tests.', encoding='utf-8')
    rules = tmp_path / '.forge' / 'rules'
    rules.mkdir(parents=True)
    (rules / 'python.md').write_text('Use Python 3.12.', encoding='utf-8')
    context = RepositoryContext(tmp_path)
    context.memory.remember('calculator', 'Calculator tests use pytest.')

    suffix = context.system_suffix('fix calculator tests')

    assert 'Always run tests.' in suffix
    assert 'Use Python 3.12.' in suffix
    assert 'Calculator tests use pytest.' in suffix


def test_repository_context_loads_global_and_nested_agents_files(
    tmp_path: Path,
) -> None:
    home = tmp_path / 'home'
    global_forge = home / '.forge'
    global_forge.mkdir(parents=True)
    (global_forge / 'AGENTS.md').write_text(
        'Global instructions.',
        encoding='utf-8',
    )
    repo = tmp_path / 'repo'
    package = repo / 'src' / 'package'
    package.mkdir(parents=True)
    (repo / 'AGENTS.md').write_text('Root instructions.', encoding='utf-8')
    (repo / 'AGENTS.override.md').write_text(
        'Root override.',
        encoding='utf-8',
    )
    (repo / 'src' / 'AGENTS.md').write_text(
        'Source instructions.',
        encoding='utf-8',
    )
    (package / 'AGENTS.override.md').write_text(
        'Package override.',
        encoding='utf-8',
    )

    instructions = RepositoryContext(
        repo,
        cwd=package,
        home=home,
    ).instructions()

    assert instructions.index('Global instructions.') < instructions.index(
        'Root instructions.'
    )
    assert instructions.index('Root instructions.') < instructions.index(
        'Root override.'
    )
    assert instructions.index('Root override.') < instructions.index(
        'Source instructions.'
    )
    assert instructions.index('Source instructions.') < instructions.index(
        'Package override.'
    )
    assert '## ~/.forge/AGENTS.md' in instructions
    assert '## src/package/AGENTS.override.md' in instructions


def test_consolidation_removes_exact_duplicate_content(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)
    store.remember('first', 'Same durable fact.')
    store.remember('second', 'Same durable fact.')

    removed = store.consolidate()

    assert removed == 1
    assert len(store.list()) == 1


def test_explicit_user_memory_is_captured_but_secret_is_ignored(
    tmp_path: Path,
) -> None:
    from forge.context.manager import ContextManager

    manager = ContextManager([], tmp_path)

    saved = manager.capture_explicit_memory('请记住：项目测试使用 uv run pytest')
    rejected = manager.capture_explicit_memory('记住：API_KEY=sk-secret1234')

    assert saved is not None
    assert saved.source == 'explicit_user_prompt'
    assert rejected is None
    assert len(MemoryStore(tmp_path).list()) == 1
