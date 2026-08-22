from __future__ import annotations

LOW_VALUE_PATH_PARTS = {"examples", "docs", "doc", "test", "tests"}
SOURCE_PATH_PARTS = {"app", "lib", "package", "packages", "src"}
LOW_VALUE_FILENAMES = {"README.md", "setup.py"}


def normalized_path(path: str) -> str:
    return path.replace("\\", "/")


def is_low_value_path(path: str) -> bool:
    return normalized_path(path).rsplit("/", 1)[-1] in LOW_VALUE_FILENAMES


def has_low_value_part(path: str) -> bool:
    return bool(set(normalized_path(path).split("/")[:-1]) & LOW_VALUE_PATH_PARTS)


def has_source_part(path: str) -> bool:
    return bool(set(normalized_path(path).split("/")[:-1]) & SOURCE_PATH_PARTS)
