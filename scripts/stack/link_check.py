"""Check repo-relative links in the fork's docs resolve to files that exist."""

import pathlib
import re
import sys

ROOT = pathlib.Path(".")
DOCS = ["SM70.md", "README.md", "NOTICE", "docs/v100"]

LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")

def targets():
    out = []
    for d in DOCS:
        p = ROOT / d
        if p.is_file():
            out.append(p)
        elif p.is_dir():
            out.extend(p.rglob("*.md"))
    return out


def main() -> int:
    bad = []
    checked = 0
    for p in targets():
        text = p.read_text(encoding="utf-8")
        for m in LINK.finditer(text):
            url = m.group(1)
            if url.startswith(("http://", "https://", "mailto:", "#")):
                continue
            path = url.split("#")[0]
            if not path:
                continue
            checked += 1
            target = (p.parent / path).resolve()
            if not target.exists():
                bad.append((str(p), url, str(target.relative_to(ROOT.resolve()))))
    print(f"files={len(targets())} relative_links_checked={checked} broken={len(bad)}")
    for f, url, resolved in bad[:30]:
        print(f"  {f}: {url} -> {resolved}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
