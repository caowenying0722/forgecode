'''Tests for repository filesystem and search tools.'''

import asyncio
from pathlib import Path

from forge.tools.filesystem import (
    ListDirectoryTool,
    ReadFileTool,
    ReplaceTextTool,
    WriteFileTool,
)
from forge.tools.search import FindFilesTool, GrepTool
from forge.tools.base import ToolResult


def run(coroutine: object) -> ToolResult:
    return asyncio.run(coroutine)  # type: ignore[arg-type]


def create_repository(root: Path) -> None:
    (root / 'src').mkdir()
    (root / 'src' / 'app.py').write_text(
        'first\nTODO: repair parser\nthird\n',
        encoding='utf-8',
    )
    (root / 'src' / 'app.ts').write_text(
        '// TODO: TypeScript\n',
        encoding='utf-8',
    )
    (root / 'README.md').write_text('# Example\n', encoding='utf-8')
    (root / 'node_modules').mkdir()
    (root / 'node_modules' / 'ignored.py').write_text(
        'TODO: ignored\n',
        encoding='utf-8',
    )


def test_list_directory_sorts_directories_before_files(
    tmp_path: Path,
) -> None:
    create_repository(tmp_path)

    result = run(ListDirectoryTool(tmp_path).run({'path': '.'}))

    assert result.success is True
    assert result.content.splitlines() == [
        'node_modules/',
        'src/',
        'README.md',
    ]
    assert result.metadata['entry_count'] == 3


def test_read_file_supports_inclusive_line_ranges(tmp_path: Path) -> None:
    create_repository(tmp_path)

    result = run(
        ReadFileTool(tmp_path).run(
            {'path': 'src/app.py', 'start_line': 2, 'end_line': 3}
        )
    )

    assert result.success is True
    assert result.content == (
        '     2 | TODO: repair parser\n'
        '     3 | third'
    )
    assert result.metadata == {
        'path': 'src/app.py',
        'start_line': 2,
        'end_line': 3,
        'total_lines': 3,
    }


def test_read_file_rejects_an_inverted_range(tmp_path: Path) -> None:
    create_repository(tmp_path)

    result = run(
        ReadFileTool(tmp_path).run(
            {'path': 'src/app.py', 'start_line': 3, 'end_line': 2}
        )
    )

    assert result.success is False
    assert result.error is not None
    assert result.error.code == 'invalid_arguments'


def test_write_file_creates_and_atomically_replaces_small_text(
    tmp_path: Path,
) -> None:
    tool = WriteFileTool(tmp_path)

    created = run(tool.run({'path': 'game.html', 'content': 'first'}))
    replaced = run(tool.run({'path': 'game.html', 'content': 'second'}))

    assert created.success is True
    assert created.metadata['created'] is True
    assert replaced.success is True
    assert replaced.metadata['created'] is False
    assert (tmp_path / 'game.html').read_text(encoding='utf-8') == 'second'
    assert not list(tmp_path.glob('*.forge-tmp'))


def test_write_file_rejects_content_over_8000_characters(
    tmp_path: Path,
) -> None:
    result = run(
        WriteFileTool(tmp_path).run(
            {'path': 'large.html', 'content': 'x' * 8_001}
        )
    )

    assert result.success is False
    assert result.error is not None
    assert result.error.code == 'invalid_arguments'
    assert not (tmp_path / 'large.html').exists()


def test_replace_text_requires_one_exact_occurrence(tmp_path: Path) -> None:
    path = tmp_path / 'game.js'
    path.write_text('const gravity = 1;\n', encoding='utf-8')
    tool = ReplaceTextTool(tmp_path)

    replaced = run(
        tool.run(
            {
                'path': 'game.js',
                'old_text': 'gravity = 1',
                'new_text': 'gravity = 0.08',
            }
        )
    )
    missing = run(
        tool.run(
            {
                'path': 'game.js',
                'old_text': 'missing',
                'new_text': 'value',
            }
        )
    )

    assert replaced.success is True
    assert 'gravity = 0.08' in path.read_text(encoding='utf-8')
    assert missing.success is False
    assert missing.error is not None
    assert missing.error.code == 'text_not_unique'


def test_find_files_uses_globs_and_ignores_generated_directories(
    tmp_path: Path,
) -> None:
    create_repository(tmp_path)

    result = run(FindFilesTool(tmp_path).run({'pattern': '*.py'}))

    assert result.success is True
    assert result.content == 'src/app.py'
    assert result.metadata['truncated'] is False


def test_grep_supports_path_and_file_type_filters(tmp_path: Path) -> None:
    create_repository(tmp_path)

    result = run(
        GrepTool(tmp_path).run(
            {
                'pattern': 'todo:',
                'path': 'src',
                'file_types': ['py'],
                'case_sensitive': False,
            }
        )
    )

    assert result.success is True
    assert result.content == 'src/app.py:2:TODO: repair parser'
    assert result.metadata['match_count'] == 1


def test_grep_returns_invalid_regex_as_structured_error(
    tmp_path: Path,
) -> None:
    create_repository(tmp_path)

    result = run(GrepTool(tmp_path).run({'pattern': '['}))

    assert result.success is False
    assert result.error is not None
    assert result.error.code == 'invalid_pattern'
