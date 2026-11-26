"""Off Line: score is the on-screen counter, floor(score / 10), where `score` counts ticks on the line.

script.js keeps its state in top-level globals: g.state is one of the functions play / end /
reset, and `score` resets to 0 on every attempt; the harness observer keeps the run's best.
"""
from __future__ import annotations

from gga.state import Hook, Reader, reader_js

SCRIPT = "Racing/off-line/web_build/src/js/script.js"
BODY = """
if (typeof g === 'undefined' || typeof score === 'undefined') return {state:'loading', value:0};
const s = g.state;
const state = (s === play || s === reset) ? 'playing' : (s === end ? 'terminal' : 'menu');
return {state, value: Math.floor(score / 10)};
"""

READER = Reader(
    game="off_line",
    hooks=(Hook(SCRIPT, "9a5bdb2ef5087c8f4d6561299d996147952642ac4634053018e41733282cce30",
                append=reader_js(BODY)),),
    note="approximation: the displayed counter",
)
