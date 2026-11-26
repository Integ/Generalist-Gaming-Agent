"""Blarglike: score = floor number x nutrients gathered (the game's own global.gmlplayer_score).

GameMaker HTML5 keeps game globals on the top-level `global` object as gml<name>. A run ends
when health (the battery) hits 0; R starts a new run, and the observer keeps the best.
"""
from __future__ import annotations

from gga.state import Hook, Reader, reader_js

SCRIPT = "Adventure/blarglike/web_build/html5game/Blarglike.js"
BODY = """
const G = globalThis.global;
if (!G || G.gmlplayer_floor === undefined || G.gmlplayer_nutrients === undefined)
  return {state:'loading', value:0};
const value = Number(G.gmlplayer_floor) * Number(G.gmlplayer_nutrients);
return {state: Number(G.gmlplayer_health) <= 0 ? 'terminal' : 'playing', value};
"""

READER = Reader(
    game="blarglike",
    hooks=(Hook(SCRIPT, "99b7a2a6ed3fca7d63fd2b5b0fc235f1ffc8fa9362e118f80bc230706bd15a17",
                append=reader_js(BODY)),),
    note="approximation: floor x nutrients from the GameMaker globals",
)
