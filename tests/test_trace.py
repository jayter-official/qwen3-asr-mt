"""Unit tests for the black-box trace writer. Pure stdlib, runs anywhere."""

import json
import os

from qasr_server.trace import write_trace


def test_writes_jsonl_lines(tmp_path):
    p = str(tmp_path / "t.jsonl")
    write_trace(p, {"a": 1, "txt": "中文與 echo"})
    write_trace(p, {"a": 2})
    lines = open(p, encoding="utf-8").read().strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0])["txt"] == "中文與 echo"   # non-ASCII preserved
    assert json.loads(lines[1])["a"] == 2


def test_creates_parent_dir(tmp_path):
    p = str(tmp_path / "sub" / "deep" / "t.jsonl")
    write_trace(p, {"ok": True})
    assert os.path.isfile(p)


def test_never_raises_on_bad_path():
    # NUL byte makes the path invalid -> error must be swallowed, not raised.
    write_trace("bad\x00path/t.jsonl", {"a": 1})


def test_never_raises_on_unserialisable():
    # A value json can't encode must not crash the caller.
    write_trace("ignored.jsonl", {"x": object()})
