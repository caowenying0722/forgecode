'''Tests for loss-tolerant subprocess output decoding.'''

from forge.runtime.process import (
    ProcessResult,
    decode_process_output,
    process_metadata,
    render_process_output,
)


def test_windows_cp936_stderr_is_decoded() -> None:
    raw = '构建失败'.encode('cp936')

    decoded = decode_process_output(
        raw,
        stream='stderr',
        preferred_encoding='utf-8',
        windows=True,
    )

    assert decoded.text == '构建失败'
    assert decoded.encoding in {'cp936', 'gbk'}
    assert decoded.warning


def test_invalid_bytes_do_not_crash_process_rendering() -> None:
    decoded = decode_process_output(
        b'prefix\xff\xfe\x81suffix',
        stream='stderr',
        preferred_encoding='ascii',
        windows=False,
    )
    result = ProcessResult(
        9,
        '',
        decoded.text,
        0.1,
        stderr_encoding=decoded.encoding,
        decode_warnings=(decoded.warning,) if decoded.warning else (),
    )

    rendered = render_process_output(result)

    assert 'stderr:' in rendered
    assert 'prefix' in rendered
    assert '\ufffd' in rendered
    metadata = process_metadata(result)
    assert metadata['stderr_encoding'] == decoded.encoding
    assert metadata['decode_warnings']
