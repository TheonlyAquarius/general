import os
from pathlib import Path
from typing import Optional, Union

PathLike = Union[str, Path]
OUTPUT_BASE_ENV = "EXPERIMENT_OUTPUT_BASE_DIR"

def _coerce_path(value: PathLike) -> Path:
    return Path(value).expanduser()

def project_root() -> Path:
    return Path(__file__).resolve().parent.parent

def get_base_dir(explicit_base: Optional[PathLike] = None) -> Path:
    if explicit_base is not None:
        return _coerce_path(explicit_base).resolve()
    env_dir = os.getenv(OUTPUT_BASE_ENV)
    if env_dir:
        return _coerce_path(env_dir).resolve()
    return project_root()

def resolve_path(path: PathLike, base_dir: Optional[PathLike] = None) -> Path:
    candidate = _coerce_path(path)
    if candidate.is_absolute():
        return candidate.resolve()
    base_path = get_base_dir(base_dir)
    return (base_path / candidate).resolve()

def ensure_directory(path: PathLike, base_dir: Optional[PathLike] = None) -> Path:
    directory = resolve_path(path, base_dir)
    directory.mkdir(parents=True, exist_ok=True)
    return directory

def ensure_parent_directory(path: PathLike, base_dir: Optional[PathLike] = None) -> Path:
    file_path = resolve_path(path, base_dir)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    return file_path
