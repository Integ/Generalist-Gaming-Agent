"""A model-free CPU baseline: open the game, then play the inputs its brief names.

It is the floor every smarter agent must beat, and the fallback when a model is slow
or absent. Phases per game:

  start  cycle through openers (start keys from the brief, Space, Enter, clicks on the
         centre and on salient points) until the screen keeps changing by itself;
  play   one-second `sequence` programs over the brief's movement and action keys,
         with the pointer held or clicked where the brief says it is used;
  stuck  when the screen barely changes for STUCK_S, go back to start (a game-over or
         menu screen usually needs the same openers).
"""
from __future__ import annotations

import random

from gga.protocol import Action

from . import brief as brief_mod
from . import vision

START_GAP_S = 0.6        # wait between openers, so each can take effect
STUCK_S = 4.0            # little change for this long: assume a menu or game-over screen
STILL = 0.0015           # mean thumbnail change below this counts as "nothing moved"
PLAY_PROGRAM_MS = 1000


class HeuristicAgent:
    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self.reset({})

    # -- per-game state ---------------------------------------------------
    def reset(self, info: dict) -> None:
        self.brief = brief_mod.parse(info)
        self.rng = random.Random(self.seed)
        self.phase = "start"
        self.opener_i = 0
        self.last_opener_s = -1e9
        self.prev_thumb = None
        self.last_motion_s = 0.0
        self.program_until_s = 0.0

    def openers(self, w: int, h: int, pts: list[tuple[int, int]]) -> list[Action]:
        b = self.brief
        acts = [Action("press_key", {"key": k}) for k in b.start_keys]
        acts += [Action("press_key", {"key": "Space"}), Action("press_key", {"key": "Enter"}),
                 Action("click", {"x": w // 2, "y": h // 2})]
        acts += [Action("click", {"x": x, "y": y}) for x, y in pts[:4]]
        return acts

    # -- the policy --------------------------------------------------------
    def step(self, obs, tools) -> Action:
        info = obs.info or {}
        w, h = int(info.get("width", 1600)), int(info.get("height", 1000))
        t = float(obs.elapsed_s)
        img = vision.decode(obs.image)
        th = vision.thumb(img)
        moved = vision.change(self.prev_thumb, th) > STILL
        self.prev_thumb = th
        if moved:
            self.last_motion_s = t

        if self.phase == "play" and t - self.last_motion_s > STUCK_S:
            self.phase, self.opener_i = "start", 0

        if self.phase == "start":
            if moved and self.opener_i > 0 and t - self.last_opener_s > START_GAP_S * 2:
                self.phase = "play"           # the screen runs by itself now
            elif t - self.last_opener_s >= START_GAP_S:
                acts = self.openers(w, h, vision.salient_points(img))
                act = acts[self.opener_i % len(acts)]
                self.opener_i += 1
                self.last_opener_s = t
                if self.opener_i >= 2 * len(acts):
                    self.phase = "play"       # nothing worked: try playing anyway
                act.meta = {"phase": "start", "opener": self.opener_i}
                return act
            else:
                return Action("skip", meta={"phase": "start"})

        if t < self.program_until_s - 0.2:
            return Action("skip", meta={"phase": "play"})
        act = self.play_program(img, w, h)
        self.program_until_s = t + PLAY_PROGRAM_MS / 1000
        return act

    def play_program(self, img, w: int, h: int) -> Action:
        b, rng = self.brief, self.rng
        keys: list[str] = []
        if b.move_keys:
            keys.append(rng.choice(b.move_keys))
        extra = [k for k in b.action_keys if k not in b.start_keys and k not in ("Escape", "Tab")]
        if extra and rng.random() < 0.5:
            keys.append(rng.choice(extra))
        if not keys and not b.uses_pointer:
            keys = [rng.choice(brief_mod.ARROWS + ["Space"])]

        step: dict = {"keys": keys, "duration_ms": PLAY_PROGRAM_MS}
        if b.uses_pointer:
            x, y = rng.choice(vision.salient_points(img))
            step.update(x=x, y=y)
            if b.uses_hold_fire:
                step["buttons"] = ["left"]
            elif not keys:
                return Action("click", {"x": x, "y": y}, meta={"phase": "play"})
        return Action("sequence", {"steps": [step]}, meta={"phase": "play", "keys": keys})
