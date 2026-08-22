from __future__ import annotations

import ast
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class SymbolDefinition:
    name: str
    kind: str
    path: str
    line: int
    end_line: int
    qualname: str
    bases: tuple[str, ...] = ()
    parent: str = ""


@dataclass(frozen=True)
class ImportBinding:
    name: str
    module: str
    path: str
    line: int
    imported: str = ""


@dataclass
class SymbolIndex:
    definitions: list[SymbolDefinition] = field(default_factory=list)
    imports: list[ImportBinding] = field(default_factory=list)
    by_name: dict[str, list[SymbolDefinition]] = field(default_factory=lambda: defaultdict(list))
    subclasses: dict[str, list[SymbolDefinition]] = field(default_factory=lambda: defaultdict(list))
    callers: dict[str, list[SymbolDefinition]] = field(default_factory=lambda: defaultdict(list))
    definitions_by_module: dict[str, list[SymbolDefinition]] = field(
        default_factory=lambda: defaultdict(list)
    )

    def add_definition(self, definition: SymbolDefinition) -> None:
        self.definitions.append(definition)
        self.by_name[definition.name].append(definition)
        self.by_name[definition.qualname].append(definition)
        self.definitions_by_module[_module_name(definition.path)].append(definition)
        for base in definition.bases:
            self.subclasses[base].append(definition)

    def add_import(self, binding: ImportBinding) -> None:
        self.imports.append(binding)

    def add_call(self, symbol: str, caller: SymbolDefinition) -> None:
        self.callers[symbol].append(caller)

    def resolve_import(self, name: str, from_path: str) -> list[SymbolDefinition]:
        resolved = []
        for binding in self.imports:
            if binding.path != from_path or binding.name != name:
                continue
            module = _absolute_module(binding.module, from_path)
            candidates = self.definitions_by_module.get(module, [])
            if binding.imported:
                candidates = [item for item in candidates if item.name == binding.imported]
            resolved.extend(candidates)
        return list(dict.fromkeys(resolved))


def build_symbol_index(repo_root: Path, files: list[str]) -> SymbolIndex:
    index = SymbolIndex()
    for relative in files:
        if not relative.endswith(".py"):
            continue
        path = repo_root / relative
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(source)
        except (OSError, SyntaxError):
            continue
        visitor = _SymbolVisitor(relative, index)
        visitor.visit(tree)
    return index


class _SymbolVisitor(ast.NodeVisitor):
    def __init__(self, path: str, index: SymbolIndex) -> None:
        self.path = path
        self.index = index
        self.scope: list[SymbolDefinition] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        definition = SymbolDefinition(
            name=node.name,
            kind="class",
            path=self.path,
            line=node.lineno,
            end_line=getattr(node, "end_lineno", node.lineno),
            qualname=_qualname(self.scope, node.name),
            bases=tuple(_base_name(base) for base in node.bases if _base_name(base)),
            parent=self.scope[-1].qualname if self.scope else "",
        )
        self.index.add_definition(definition)
        self.scope.append(definition)
        self.generic_visit(node)
        self.scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node, "function")

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node, "function")

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.index.add_import(
                ImportBinding(
                    name=alias.asname or alias.name.split(".", 1)[0],
                    module=alias.name,
                    path=self.path,
                    line=node.lineno,
                )
            )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = "." * node.level + (node.module or "")
        for alias in node.names:
            self.index.add_import(
                ImportBinding(
                    name=alias.asname or alias.name,
                    module=module,
                    imported=alias.name,
                    path=self.path,
                    line=node.lineno,
                )
            )

    def visit_Call(self, node: ast.Call) -> None:
        caller = self.scope[-1] if self.scope else None
        symbol = _call_name(node.func)
        if caller and symbol:
            self.index.add_call(symbol, caller)
        self.generic_visit(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef, kind: str) -> None:
        definition = SymbolDefinition(
            name=node.name,
            kind=kind,
            path=self.path,
            line=node.lineno,
            end_line=getattr(node, "end_lineno", node.lineno),
            qualname=_qualname(self.scope, node.name),
            parent=self.scope[-1].qualname if self.scope else "",
        )
        self.index.add_definition(definition)
        self.scope.append(definition)
        self.generic_visit(node)
        self.scope.pop()


def _qualname(scope: list[SymbolDefinition], name: str) -> str:
    if not scope:
        return name
    return ".".join([scope[-1].qualname, name])


def _base_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _base_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    if isinstance(node, ast.Subscript):
        return _base_name(node.value)
    return ""


def _call_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _module_name(path: str) -> str:
    normalized = path.replace("\\", "/")
    if normalized.endswith("/__init__.py"):
        normalized = normalized[: -len("/__init__.py")]
    elif normalized.endswith(".py"):
        normalized = normalized[:-3]
    parts = normalized.split("/")
    if parts and parts[0] in {"src", "lib"}:
        parts = parts[1:]
    return ".".join(parts)


def _absolute_module(module: str, from_path: str) -> str:
    if not module.startswith("."):
        return module
    level = len(module) - len(module.lstrip("."))
    suffix = module[level:]
    package = _module_name(from_path).split(".")[:-1]
    keep = max(0, len(package) - level + 1)
    return ".".join([*package[:keep], *([suffix] if suffix else [])])
