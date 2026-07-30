'''Tests for task-local structured repository evidence.'''

from forge.context.working import FileEvidence, WorkingState
from forge.runtime.state import ToolCall
from forge.tools.base import ToolResult


def read_call(
    call_id: str,
    *,
    start_line: int = 1,
    end_line: int | None = None,
) -> ToolCall:
    arguments = {'path': 'play/js/player.js', 'start_line': start_line}
    if end_line is not None:
        arguments['end_line'] = end_line
    return ToolCall(
        index=0,
        id=call_id,
        name='read_file',
        arguments=arguments,
    )


def test_exact_read_replays_content_and_new_ranges_may_execute() -> None:
    state = WorkingState()
    initial = read_call('first', end_line=260)
    result = ToolResult.ok(
        'Read file.',
        content='\n'.join(
            f'{line:>6} | line {line}' for line in range(1, 252)
        ),
        metadata={
            'path': 'play/js/player.js',
            'start_line': 1,
            'end_line': 251,
            'total_lines': 251,
        },
    )

    assert state.observe(initial, result, 0, 'first') is True

    replay = state.preflight(initial, 0, 'first')

    assert replay is not None and replay.success is True
    assert replay.summary.startswith('Replayed covered read')
    assert '     1 | line 1' in replay.content
    assert '   251 | line 251' in replay.content
    assert replay.metadata['cache_hit'] is True
    subset = state.preflight(read_call('subset', end_line=120), 0, 'subset')
    extended = state.preflight(
        read_call('extended', end_line=320),
        0,
        'extended',
    )
    assert subset is not None and subset.metadata['evidence_replayed'] is True
    assert subset.summary.endswith('lines 1-120.')
    assert '     1 | line 1' in subset.content
    assert '   120 | line 120' in subset.content
    assert '   121 | line 121' not in subset.content
    assert extended is not None
    assert extended.metadata['end_line'] == 251
    assert state.preflight(read_call('new-revision', end_line=260), 1, 'new') is None


def test_adjacent_and_overlapping_segments_replay_requested_content() -> None:
    for second_start in (40, 41):
        state = WorkingState()
        for call_id, start_line, end_line in (
            ('first', 1, 40),
            ('second', second_start, 316),
        ):
            call = read_call(
                call_id,
                start_line=start_line,
                end_line=end_line,
            )
            result = ToolResult.ok(
                'Read file.',
                content='\n'.join(
                    f'{line:>6} | line {line}'
                    for line in range(start_line, end_line + 1)
                ),
                metadata={
                    'path': 'play/js/player.js',
                    'start_line': start_line,
                    'end_line': end_line,
                    'total_lines': 316,
                },
            )
            assert state.observe(call, result, 0, call_id) is True

        replay = state.preflight(
            read_call('combined', end_line=140),
            0,
            'combined',
        )

        assert replay is not None
        assert replay.metadata['evidence_replayed'] is True
        assert '     1 | line 1' in replay.content
        assert '   140 | line 140' in replay.content
        assert '   141 | line 141' not in replay.content


def test_read_replay_keeps_source_available_after_compaction_reference() -> None:
    state = WorkingState()
    call = read_call('first', end_line=3)
    state.observe(
        call,
        ToolResult.ok(
            'Read file.',
            content='     1 | alpha\n     2 | beta\n     3 | gamma',
            metadata={
                'path': 'play/js/player.js',
                'start_line': 1,
                'end_line': 3,
                'total_lines': 3,
            },
        ),
        0,
        'first',
    )

    assert 'replayable from cached source' in state.system_suffix()
    replay = state.preflight(read_call('after-compaction', end_line=3), 0)

    assert replay is not None
    assert replay.metadata['evidence_replayed'] is True
    assert replay.content == '     1 | alpha\n     2 | beta\n     3 | gamma'


def test_working_evidence_is_small_and_answer_check_uses_paths() -> None:
    state = WorkingState()
    call = read_call('first', end_line=20)
    state.observe(
        call,
        ToolResult.ok(
            'Read file.',
            metadata={
                'path': 'play/js/player.js',
                'start_line': 1,
                'end_line': 20,
                'total_lines': 251,
            },
        ),
        0,
        'first',
    )

    assert 'play/js/player.js' in state.system_suffix()
    assert state.answer_mentions_evidence(
        'player.js handles movement and collision.'
    )
    assert not state.answer_mentions_evidence('I am ForgeCode.')


def test_evidence_match_accepts_exact_path() -> None:
    state = WorkingState()
    state.files[(0, 'src/main.js')] = FileEvidence(
        path='src/main.js',
        revision=0,
        total_lines=1,
        covered_ranges=[(1, 1)],
    )

    assert state.answer_mentions_evidence('Updated src/main.js.')


def test_evidence_match_accepts_basename() -> None:
    state = WorkingState()
    state.files[(0, 'src/main.js')] = FileEvidence(
        path='src/main.js',
        revision=0,
        total_lines=1,
        covered_ranges=[(1, 1)],
    )

    assert state.answer_mentions_evidence('Updated main.js.')


def test_evidence_match_rejects_incidental_stem_substring() -> None:
    state = WorkingState()
    state.files[(0, 'src/main.js')] = FileEvidence(
        path='src/main.js',
        revision=0,
        total_lines=1,
        covered_ranges=[(1, 1)],
    )

    assert not state.answer_mentions_evidence('The main issue is validation.')


def test_short_file_stem_does_not_false_positive() -> None:
    state = WorkingState()
    state.files[(0, 'src/a.py')] = FileEvidence(
        path='src/a.py',
        revision=0,
        total_lines=1,
        covered_ranges=[(1, 1)],
    )
    state.files[(0, 'src/x.ts')] = FileEvidence(
        path='src/x.ts',
        revision=0,
        total_lines=1,
        covered_ranges=[(1, 1)],
    )

    assert not state.answer_mentions_evidence('A fix exists in the code.')
    assert not state.answer_mentions_evidence('X marks the failing branch.')
    assert state.answer_mentions_evidence('Updated a.py.')
    assert state.answer_mentions_evidence('Updated src/x.ts.')


def test_working_evidence_suffix_is_bounded() -> None:
    state = WorkingState()
    for index in range(75):
        state.files[(0, f'src/file_{index}.py')] = FileEvidence(
            path=f'src/file_{index}.py',
            revision=0,
            total_lines=1,
            covered_ranges=[(1, 1)],
        )
    for index in range(30):
        state.directories.add((0, f'src/dir_{index}'))

    suffix = state.system_suffix()

    assert '25 older file entries omitted' in suffix
    assert '10 older directory entries omitted' in suffix
    assert suffix.count('@ revision') == 70


def test_grep_of_fully_read_file_remains_available() -> None:
    state = WorkingState()
    state.observe(
        read_call('read', end_line=251),
        ToolResult.ok(
            'Read file.',
            metadata={
                'path': 'play/js/player.js',
                'start_line': 1,
                'end_line': 251,
                'total_lines': 251,
            },
        ),
        0,
        'read',
    )
    grep_call = ToolCall(
        index=0,
        id='grep',
        name='grep',
        arguments={'path': 'play/js/player.js', 'pattern': 'class'},
    )

    result = state.preflight(grep_call, 0)

    assert result is None


def test_grep_progress_requires_new_unread_match() -> None:
    state = WorkingState()
    call = ToolCall(
        index=0,
        id='grep',
        name='grep',
        arguments={'path': 'src', 'pattern': 'main'},
    )
    result = ToolResult.ok(
        'Found matches.',
        content='src/app.py:10:def main():\nsrc/lib.py:20:main()',
    )

    assert state.observe(call, result, 0, 'first') is True
    assert state.evidence_paths == ('src/app.py', 'src/lib.py')
    assert 'search hit src/app.py:10' in state.system_suffix()
    assert state.observe(call, result, 0, 'second') is False
    assert state.observe(
        call,
        ToolResult.ok('No matches.'),
        0,
        'empty',
    ) is False


def test_find_files_does_not_rediscover_an_already_read_path() -> None:
    state = WorkingState()
    state.observe(
        read_call('read', end_line=20),
        ToolResult.ok(
            'Read file.',
            metadata={
                'path': 'play/js/player.js',
                'start_line': 1,
                'end_line': 20,
                'total_lines': 20,
            },
        ),
        0,
        'read',
    )
    find_call = ToolCall(
        index=0,
        id='find',
        name='find_files',
        arguments={'path': 'play/js', 'pattern': '*.js'},
    )

    assert state.observe(
        find_call,
        ToolResult.ok(
            'Found one file.',
            content='play/js/player.js',
        ),
        0,
        'find-known',
    ) is False
    assert state.observe(
        find_call,
        ToolResult.ok(
            'Found one file.',
            content='play/js/world.js',
        ),
        0,
        'find-new',
    ) is True
    assert 'play/js/world.js' in state.evidence_paths
    assert 'discovered play/js/world.js' in state.system_suffix()


def test_empty_git_results_do_not_count_as_evidence_progress() -> None:
    state = WorkingState()

    for tool_name in ('git_diff', 'git_status'):
        call = ToolCall(
            index=0,
            id=tool_name,
            name=tool_name,
            arguments={},
        )
        assert state.observe(
            call,
            ToolResult.ok('No Git evidence.', content=''),
            0,
            tool_name,
        ) is False


def test_only_explicit_external_error_codes_create_a_blocker() -> None:
    state = WorkingState()
    call = ToolCall(0, 'failure', 'read_file', {'path': 'sample.txt'})

    state.observe(
        call,
        ToolResult.fail('invalid_arguments', 'Bad schema.'),
        0,
        'bad-schema',
    )
    assert state.has_external_blocker is False

    state.observe(
        call,
        ToolResult.fail('permission_denied', 'Approval is required.'),
        0,
        'permission',
    )
    assert state.has_external_blocker is True

    state.observe(
        call,
        ToolResult.ok('Recovered with an available repository path.'),
        0,
        'recovered',
    )
    assert state.has_external_blocker is False


def test_revision_change_invalidates_only_changed_file() -> None:
    state = WorkingState()
    for path in ('src/changed.py', 'src/unchanged.py'):
        call = ToolCall(
            index=0,
            id=path,
            name='read_file',
            arguments={'path': path},
        )
        state.observe(
            call,
            ToolResult.ok(
                'Read file.',
                metadata={
                    'path': path,
                    'start_line': 1,
                    'end_line': 20,
                    'total_lines': 20,
                },
            ),
            0,
            path,
        )

    state.advance_revision(1, ('src/changed.py',))

    changed = ToolCall(
        0,
        'changed',
        'read_file',
        {'path': 'src/changed.py'},
    )
    unchanged = ToolCall(
        0,
        'unchanged',
        'read_file',
        {'path': 'src/unchanged.py'},
    )
    assert state.preflight(changed, 1) is None
    replayed = state.preflight(unchanged, 1)
    assert replayed is not None
    assert replayed.metadata['evidence_replayed'] is True
    assert '@ revision 0' not in state.system_suffix()
