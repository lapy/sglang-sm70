"""Static check: every `from sglang... import ...` in fork-touched files resolves.

Catches upstream renames/removals (e.g. utils.common -> utils.device) without needing
torch on this machine.
"""

import ast
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(".")
PKG = ROOT / "python" / "sglang"


def changed_files(ref: str) -> list[str]:
    out = subprocess.run(
        ["git", "diff", "--name-only", f"{ref}", "HEAD", "--", "python/sglang"],
        capture_output=True, text=True, check=True).stdout.split()
    return [f for f in out if f.endswith(".py") and (ROOT / f).is_file()]


def module_exists(dotted: str) -> bool:
    parts = dotted.split(".")
    # a module file or a package dir (with __init__.py) must exist
    path = ROOT / "python" / pathlib.Path(*parts)
    return path.with_suffix(".py").is_file() or (path / "__init__.py").is_file()


def names_in(dotted: str) -> set[str]:
    parts = dotted.split(".")
    for i in range(len(parts), 0, -1):
        cand = ROOT / "python" / pathlib.Path(*parts[:i])
        f = cand.with_suffix(".py")
        if not f.is_file():
            f = cand / "__init__.py"
        if f.is_file():
            try:
                tree = ast.parse(f.read_text(errors="replace"))
            except SyntaxError:
                return set()
            names = set()
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    names.add(node.name)
                elif isinstance(node, ast.Assign):
                    for t in node.targets:
                        if isinstance(t, ast.Name):
                            names.add(t.id)
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    for a in node.names:
                        names.add(a.asname or a.name.split(".")[0])
            return names
    return set()


def main() -> int:
    ref = sys.argv[1] if len(sys.argv) > 1 else "upstream/main"
    files = changed_files(ref)
    bad = []
    for f in files:
        try:
            tree = ast.parse((ROOT / f).read_text(errors="replace"))
        except SyntaxError as e:
            bad.append((f, f"SYNTAX: {e}"))
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("sglang."):
                if not module_exists(node.module):
                    bad.append((f, f"missing module {node.module}"))
                    continue
                have = names_in(node.module)
                if have:
                    for a in node.names:
                        if a.name != "*" and a.name not in have:
                            bad.append((f, f"{node.module} has no {a.name}"))
    print(f"checked {len(files)} fork-touched files against {ref}")
    if not bad:
        print("all sglang imports resolve")
        return 0
    for f, why in bad[:60]:
        print(f"  {f}: {why}")
    print(f"issues: {len(bad)}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
