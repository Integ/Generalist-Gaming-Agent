"""eval/jam.py reads score readers out of the dataset's source without executing it."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
import jam  # noqa: E402

SRC = '''
import os
_PLAY = "state: 'playing'"
os.system("echo should never run")
_reader("g1", "A/g1/main.js", f"return {{{_PLAY}, value: score}};", "note one")
_reader("g2", "A/g2/index.html", "return " "{state: 'playing', value: 1};", "two",
        replace=(("a", "b"),), html=True, more=(("A/g2/x.js", (("c", "d"),)),))
'''


def test_reader_specs_parse_without_running(tmp_path):
    p = tmp_path / "extra_games.py"
    p.write_text(SRC)
    s = jam.reader_specs(p)
    assert s["g1"]["body"] == "return {state: 'playing', value: score};"
    assert s["g2"]["html"] and s["g2"]["replace"] == (("a", "b"),)
    assert s["g2"]["more"] == (("A/g2/x.js", (("c", "d"),)),)


def test_unknown_expression_is_refused(tmp_path):
    p = tmp_path / "extra_games.py"
    p.write_text('_reader("g", "p", open("x").read(), "n")\n')
    with pytest.raises(ValueError):
        jam.reader_specs(p)


def test_normalise():
    spec = {"scoring": {"idle": 2, "reference": 12}}
    assert jam.normalise(spec, 7) == 0.5
    assert jam.normalise(spec, 50) == 1.0
    assert jam.normalise(spec, None) is None
    assert jam.normalise({"scoring": {"idle": 3, "reference": 3}}, 5) is None
