"""Read literal prompt assignments from external Python files without executing code."""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PromptSource:
    path: Path
    sha256: str
    strings: dict[str, str]

    @classmethod
    def load(cls, path: str | Path, required: tuple[str, ...]) -> PromptSource:
        path = Path(path)
        try:
            raw = path.read_bytes()
            tree = ast.parse(raw.decode("utf-8"), filename=str(path))
        except (OSError, UnicodeError, SyntaxError) as error:
            raise ValueError(f"external prompts {path}: {error}") from error
        strings = {}

        def assignments(body, prefix=""):
            for node in body:
                if isinstance(node, ast.ClassDef):
                    yield from assignments(node.body, prefix + node.name + ".")
                elif isinstance(node, ast.Assign):
                    yield prefix, node

        for prefix, node in assignments(tree.body):
            for target in node.targets:
                if not isinstance(target, ast.Name) or prefix + target.id not in required:
                    continue
                name = prefix + target.id
                if name in strings:
                    raise ValueError(f"external prompts {path}: repeated {name}")
                if not isinstance(node.value, ast.Constant) or not isinstance(
                    node.value.value, str
                ):
                    raise ValueError(f"external prompts {path}: {name} must be a literal string")
                strings[name] = node.value.value
        if missing := [name for name in required if not strings.get(name, "").strip()]:
            raise ValueError(f"external prompts {path}: missing literal strings {missing}")
        return cls(path, hashlib.sha256(raw).hexdigest(), strings)

    def metadata(self) -> dict:
        return {"path": str(self.path), "sha256": self.sha256, "loader": "ast_string_literals"}
