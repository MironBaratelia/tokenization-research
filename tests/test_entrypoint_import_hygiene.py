"""Publication-facing hygiene checks for entrypoint scripts."""

from __future__ import annotations

import ast
from pathlib import Path


def _unused_imports(path: Path) -> list[tuple[str, int]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported: list[tuple[str, int]] = []
    loaded: set[str] = set()

    class Visitor(ast.NodeVisitor):
        def visit_Import(self, node: ast.Import) -> None:
            for alias in node.names:
                imported.append((alias.asname or alias.name.split(".")[0], node.lineno))

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            for alias in node.names:
                if alias.name != "*":
                    imported.append((alias.asname or alias.name, node.lineno))

        def visit_Name(self, node: ast.Name) -> None:
            if isinstance(node.ctx, ast.Load):
                loaded.add(node.id)

    Visitor().visit(tree)
    return [
        (name, lineno)
        for name, lineno in imported
        if name not in loaded and not name.startswith("_")
    ]


def test_entrypoint_scripts_have_no_confirmed_unused_imports() -> None:
    root = Path(__file__).resolve().parents[1]
    scripts_dir = root / "scripts"
    findings = {}
    for path in sorted(scripts_dir.glob("*.py")):
        unused = _unused_imports(path)
        if unused:
            findings[path.name] = unused
    assert findings == {}
