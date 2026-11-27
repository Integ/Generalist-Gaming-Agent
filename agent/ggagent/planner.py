"""The slow tier: a vision-language model reads the screen in a background thread and writes a Plan.

The game never waits for it. The agent hands the planner its newest frame every step
(observe() only stores it), and the planner thread asks the model whenever the previous
answer is in, at most once per MIN_GAP_S. A Plan is advice the CPU policy applies on its
next 200 ms steps: what screen this is, where to click to get into the game, which keys
move and act, whether to hold fire and where to aim. A plan older than TTL_S game seconds
is dropped, so a stale answer never steers a screen that has moved on.
"""
from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass, field

from PIL import Image

from . import brief as brief_mod

MIN_GAP_S = 1.0          # wall seconds between the end of one request and the next
TTL_S = 12.0             # game seconds a plan stays valid
MAX_TOKENS = 260
ERROR_BACKOFF_S = 5.0

SYSTEM = """You plan for a fast controller that plays a real-time browser game for a human player.
You see the current screenshot and the game's brief. The controller acts every 200 ms on its own;
your plan tells it what to do for the next few seconds. Coordinates are on a 0-1000 grid over the
screenshot (x from left, y from top).

Reply with ONE JSON object and nothing else:
{"screen": "menu" | "playing" | "game_over" | "loading" | "other",
 "click": [x, y] or null,      // what to click now to start or continue the game (a Play/Start/OK button); null while playing
 "move_keys": [...],           // keys that move the player, e.g. ["ArrowLeft","ArrowRight"]; [] if none
 "action_keys": [...],         // other keys worth pressing now, e.g. ["Space"]; [] if none
 "hold_fire": true | false,    // keep the left mouse button held (shooters with mouse fire)
 "aim": [x, y] or null,        // where the mouse should point (an enemy, a target); null if irrelevant
 "avoid": [x, y] or null,      // the most dangerous thing on screen, to move away from
 "goal": "...",                // one short sentence: what to do next
 "notes": "..."}               // one short sentence: what you learned about the rules"""


@dataclass
class Plan:
    made_at_s: float                       # game time of the frame it was made from
    screen: str = "other"
    click: tuple[int, int] | None = None   # viewport pixels
    move_keys: list[str] = field(default_factory=list)
    action_keys: list[str] = field(default_factory=list)
    hold_fire: bool = False
    aim: tuple[int, int] | None = None
    avoid: tuple[int, int] | None = None
    goal: str = ""
    notes: str = ""
    latency_s: float = 0.0

    def fresh(self, now_s: float) -> bool:
        return 0 <= now_s - self.made_at_s <= TTL_S


def _point(v, w: int, h: int) -> tuple[int, int] | None:
    if not (isinstance(v, (list, tuple)) and len(v) == 2):
        return None
    try:
        x, y = float(v[0]), float(v[1])
    except (TypeError, ValueError):
        return None
    if not (0 <= x <= 1000 and 0 <= y <= 1000):
        return None
    return min(w - 1, int(x * w / 1000)), min(h - 1, int(y * h / 1000))


def _keys(v) -> list[str]:
    from gga.protocol import valid_key, normalize_key
    if not isinstance(v, list):
        return []
    return [normalize_key(k) for k in v[:6] if isinstance(k, str) and valid_key(k)]


def parse_plan(text: str, made_at_s: float, w: int, h: int) -> Plan | None:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(d, dict):
        return None
    screen = d.get("screen") if d.get("screen") in ("menu", "playing", "game_over", "loading", "other") else "other"
    return Plan(made_at_s=made_at_s, screen=screen, click=_point(d.get("click"), w, h),
                move_keys=_keys(d.get("move_keys")), action_keys=_keys(d.get("action_keys")),
                hold_fire=bool(d.get("hold_fire")), aim=_point(d.get("aim"), w, h),
                avoid=_point(d.get("avoid"), w, h),
                goal=str(d.get("goal") or "")[:200], notes=str(d.get("notes") or "")[:200])


class Planner:
    def __init__(self, backend) -> None:
        self.backend = backend
        self._lock = threading.Lock()
        self._frame: tuple[float, Image.Image, dict] | None = None
        self._plan: Plan | None = None
        self._game = 0
        self._notes: list[str] = []
        self._wake = threading.Event()
        self.requests = self.failures = 0
        threading.Thread(target=self._loop, daemon=True, name="planner").start()

    # -- called from step(), never blocks ------------------------------------
    def new_game(self) -> None:
        with self._lock:
            self._game += 1
            self._frame, self._plan, self._notes = None, None, []

    def observe(self, t: float, img: Image.Image, info: dict) -> None:
        with self._lock:
            self._frame = (t, img, info)
        self._wake.set()

    def plan(self, now_s: float) -> Plan | None:
        with self._lock:
            p = self._plan
        return p if p is not None and p.fresh(now_s) else None

    # -- background ----------------------------------------------------------
    def _prompt(self, info: dict, t: float) -> str:
        b = brief_mod.parse(info)
        lines = [f"Game brief: {b.description}", f"Controls: {b.controls}",
                 f"Game time: {t:.0f} s, {float(info.get('remaining_s', 0)):.0f} s left."]
        if self._notes:
            lines.append("Your earlier notes: " + " | ".join(self._notes[-4:]))
        lines.append("Plan the next few seconds. JSON only.")
        return "\n".join(lines)

    def _loop(self) -> None:
        while True:
            self._wake.wait()
            self._wake.clear()
            with self._lock:
                frame, game = self._frame, self._game
            if frame is None:
                continue
            t, img, info = frame
            w, h = int(info.get("width", img.width)), int(info.get("height", img.height))
            start = time.monotonic()
            self.requests += 1
            try:
                text = self.backend.generate(SYSTEM, [img], self._prompt(info, t), MAX_TOKENS)
                plan = parse_plan(text, t, w, h)
            except Exception as exc:  # a dead or slow server must never reach step()
                self.failures += 1
                print(f"planner error: {exc!r:.200}", flush=True)
                time.sleep(ERROR_BACKOFF_S)
                self._wake.set()
                continue
            latency = time.monotonic() - start
            with self._lock:
                if game == self._game and plan is not None:
                    plan.latency_s = latency
                    self._plan = plan
                    if plan.notes:
                        self._notes.append(plan.notes)
            if plan is None:
                self.failures += 1
            time.sleep(max(0.0, MIN_GAP_S))
            self._wake.set()
