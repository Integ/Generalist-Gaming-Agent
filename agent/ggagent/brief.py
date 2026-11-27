"""Read a game's brief (obs.info description + controls) into the inputs it names.

Nothing here knows a game: it only looks for key names and pointer words in the text,
so the policy can limit itself to the inputs the game says it uses.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

ARROWS = ["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"]
WASD = ["w", "a", "s", "d"]

_NAMED = {
    "space": "Space", "spacebar": "Space", "enter": "Enter", "return": "Enter",
    "shift": "Shift", "escape": "Escape", "esc": "Escape", "tab": "Tab",
    "ctrl": "Control", "control": "Control",
}
_START_WORDS = ("play", "start", "begin", "continue", "ok", "new game", "restart", "retry")


@dataclass
class Brief:
    description: str = ""
    controls: str = ""
    move_keys: list[str] = field(default_factory=list)    # arrows and/or WASD
    action_keys: list[str] = field(default_factory=list)  # Space, letters the brief names, ...
    start_keys: list[str] = field(default_factory=list)   # keys the brief says start or restart the game
    uses_pointer: bool = False                             # click / mouse / drag / aim
    uses_hold_fire: bool = False                           # "hold the mouse button" style controls
    taps: bool = False                                     # one step per key press (grid / turn-based)
    start_words: list[str] = field(default_factory=list)  # button labels to look for ("Play", "Start")

    @property
    def keys(self) -> list[str]:
        out: list[str] = []
        for k in self.move_keys + self.action_keys:
            if k not in out:
                out.append(k)
        return out


def parse(info: dict) -> Brief:
    desc = str(info.get("description") or "")
    ctrl = str(info.get("controls") or "")
    text = ctrl.lower()
    b = Brief(description=desc, controls=ctrl)

    if re.search(r"arrow", text):
        b.move_keys += ARROWS
    if re.search(r"\bwasd\b|\bw\s*,?\s*a\s*,?\s*s\s*,?\s*(and\s*)?d\b", text):
        b.move_keys += WASD

    for word, key in _NAMED.items():
        if re.search(rf"\b{word}\b", text) and key not in b.action_keys:
            b.action_keys.append(key)
    # Single letters the brief names as keys: "press Q", "'R' to restart", "keys Z and X".
    for m in re.finditer(r"(?:press|key|keys|hit|tap|and|or|,)\s+'?\"?([A-Za-z0-9])'?\"?(?=[\s,.;)!]|$)", ctrl):
        k = m.group(1).lower()
        if k not in b.action_keys and k not in b.move_keys and k not in ("a",):
            b.action_keys.append(k)

    b.uses_pointer = bool(re.search(r"click|mouse|drag|aim|tap|pointer|cursor", text))
    b.taps = bool(re.search(r"per press|one (grid )?step|turn-based|turn based|each press", text + " " + desc.lower()))
    b.uses_hold_fire = bool(re.search(r"hold (the )?(left )?mouse|hold (the )?(left )?button|hold to fire", text))

    # Keys tied to starting: "Press Space to start", "press R to restart", "Press any key to begin".
    for m in re.finditer(r"press\s+([A-Za-z]+|'.'|\".\")[^.]*?\b(start|begin|restart|play|again)", ctrl, re.I):
        raw = m.group(1).strip("'\"")
        key = _NAMED.get(raw.lower(), raw.lower() if len(raw) == 1 else None)
        if raw.lower() == "any":
            key = "Space"
        if key and key not in b.start_keys:
            b.start_keys.append(key)
    b.start_words = [w for w in _START_WORDS if w in text]
    return b
