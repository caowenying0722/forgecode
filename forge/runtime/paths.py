'''Shared normalization for repository-relative runtime paths.'''

from pathlib import PurePosixPath


def normalize_workspace_path(path: str) -> str:
    return PurePosixPath(path.replace('\\', '/')).as_posix()
