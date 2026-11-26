"""Asteroid Capitalist: dollars earned from kills, the bundle's module-private `x.score`.

The webpack bundle keeps the score in `var x={score:0}` inside a module closure; a hook
exposes that object, and the reader returns its score. A new run resets it to 0 (the
player's constructor), and the observer keeps the best.
"""
from __future__ import annotations

from gga.state import Hook, Reader, reader_js

SCRIPT = "Action/asteroid-capitalist/web_build/dist/app.js"
ANCHOR = "var x={score:0};"
BODY = """
const x = globalThis.__ggaAsteroidState;
if (!x || !Number.isFinite(Number(x.score))) return {state:'loading', value:0};
const value = Number(x.score);
return {state: value > 0 ? 'playing' : 'menu', value};
"""

READER = Reader(
    game="asteroid_capitalist",
    hooks=(Hook(SCRIPT, "9e2953a511d63e12c5d0171b58750d8dce6f6f8660d184d558fbcfb80057fefb",
                replace=((ANCHOR, ANCHOR + "globalThis.__ggaAsteroidState=x;"),),
                append=reader_js(BODY)),),
    note="approximation: the bundle's score object",
)
