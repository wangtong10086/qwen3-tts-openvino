from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_LINK_RE = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")


def test_example_json_files_are_valid():
    for path in sorted((ROOT / "examples").glob("*.json")):
        with open(path, "r", encoding="utf-8") as handle:
            json.load(handle)


def test_example_jsonl_files_are_valid():
    for path in sorted((ROOT / "examples").glob("*.jsonl")):
        with open(path, "r", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        assert rows, path
        for row in rows:
            assert row["mode"] in {"voice_design", "custom_voice", "voice_clone"}


def test_python_examples_parse():
    for path in sorted((ROOT / "examples" / "python").glob("*.py")):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_markdown_relative_links_exist():
    markdown_roots = [ROOT / "README.md", ROOT / "README.zh-CN.md", ROOT / "docs", ROOT / "examples"]
    markdown_files: list[Path] = []
    for item in markdown_roots:
        if item.is_file():
            markdown_files.append(item)
        else:
            markdown_files.extend(sorted(item.rglob("*.md")))

    missing: list[str] = []
    for path in markdown_files:
        text = path.read_text(encoding="utf-8")
        for raw_target in MARKDOWN_LINK_RE.findall(text):
            target = raw_target.strip()
            if not target or target.startswith(("#", "http://", "https://", "mailto:")):
                continue
            target = target.split("#", 1)[0]
            if not target:
                continue
            target_path = (path.parent / unquote(target)).resolve()
            try:
                target_path.relative_to(ROOT)
            except ValueError:
                continue
            if not target_path.exists():
                missing.append(f"{path.relative_to(ROOT)} -> {raw_target}")
    assert not missing, "\n".join(missing)
