'''Architecture checks for the intentionally small model boundary.'''

from pathlib import Path


def test_anthropic_sdk_is_confined_to_model_client() -> None:
    forge_root = Path(__file__).resolve().parents[2] / 'forge'
    unexpected_imports: list[str] = []

    for path in forge_root.rglob('*.py'):
        if path.as_posix().endswith('runtime/model_client.py'):
            continue
        source = path.read_text(encoding='utf-8')
        if 'from anthropic' in source or 'import anthropic' in source:
            unexpected_imports.append(path.relative_to(forge_root).as_posix())

    assert unexpected_imports == []
