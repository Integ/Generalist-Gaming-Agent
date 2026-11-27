"""A model-free CPU baseline: open the game, then play the inputs its brief names.

It is the floor every smarter agent must beat, and the fallback when a model is slow
or absent. Phases per game:

  start  the screen still looks like the title screen (the first frame of the game):
         try openers in turn (start keys from the brief, Space, Enter, clicks down the
         centre column where Play buttons sit, then salient points);
  play   one-second `sequence` programs over the brief's movement and action keys,
         with the pointer held or clicked where the brief says it is used;
  back   the screen looks like the title again (the menu came back after a game over):
         return to start. A screen that has not changed at all for STUCK_S also gets
         one opener (a game-over screen that differs from the title).

"Looks like the title" compares a thumbnail of the frame with the title's, so an
animated menu background (moving asteroids, scrolling stars) does not count as the game
having started; frame-to-frame motion alone cannot tell those apart.
"""
from __future__ import annotations

import random

from gga.protocol import Action

from . import brief as brief_mod
from . import vision

START_GAP_S = 0.6        # wait between openers, so each can take effect
MENU_SIM = 0.012         # thumbnail change from the title below this: still on the title screen
MENU_BACK_S = 1.0        # on the title this long during play: the menu came back
STUCK_S = 6.0            # no change at all this long during play: try one opener
STILL = 0.0015           # mean thumbnail change below this counts as "nothing moved"
PLAY_PROGRAM_MS = 1000
START_GIVEUP_S = 12.0    # still "on the title" after this long of openers: trust the title test no more
PLAN_CLICK_GAP_S = 1.5
PLAN_CLICK_STILL_S = 1.5  # during play, the planner's game-over click waits for the screen to settle
AIM_FRESH_S = 3.0         # a planner aim older than this points at where the target was, not is   # the planner's menu clicks, at most this often
COLUMN = (0.5, 0.62, 0.72, 0.42, 0.82, 0.35)   # Play-button heights to try, as fractions of h


class HeuristicAgent:
    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self.reset({})

    # -- per-game state ---------------------------------------------------
    def reset(self, info: dict) -> None:
        self.brief = brief_mod.parse(info)
        self.rng = random.Random(self.seed)
        self.phase = "start"
        self.start_since_s = 0.0
        self.title_reliable = True   # False once openers ran START_GIVEUP_S without leaving the title
        self.opener_i = 0
        self.last_opener_s = -1e9
        self.title = None
        self.prev_thumb = None
        self.last_motion_s = 0.0
        self.menu_since_s = None
        self.program_until_s = 0.0
        self.last_plan_click_s = -1e9
        self.plan = None

    def openers(self, w: int, h: int, pts: list[tuple[int, int]]) -> list[Action]:
        b = self.brief
        acts = [Action("press_key", {"key": k}) for k in b.start_keys]
        if b.uses_pointer or not b.start_keys:
            acts += [Action("click", {"x": w // 2, "y": int(f * h)}) for f in COLUMN]
        acts += [Action("press_key", {"key": "Space"}), Action("press_key", {"key": "Enter"})]
        acts += [Action("click", {"x": x, "y": y}) for x, y in pts[:4]]
        return acts

    def opener(self, img, w: int, h: int, t: float, why: str) -> Action:
        acts = self.openers(w, h, vision.salient_points(img))
        act = acts[self.opener_i % len(acts)]
        self.opener_i += 1
        self.last_opener_s = t
        act.meta = {"phase": why, "opener": self.opener_i}
        return act

    # -- the policy --------------------------------------------------------
    def plan_click(self, t: float, why: str) -> Action | None:
        """The planner's click on a menu or game-over screen, at most every PLAN_CLICK_GAP_S."""
        p = self.plan
        if p is None or p.click is None or p.screen not in ("menu", "game_over", "other"):
            return None
        if t - self.last_plan_click_s < PLAN_CLICK_GAP_S:
            return None
        self.last_plan_click_s = self.last_opener_s = t
        return Action("click", {"x": p.click[0], "y": p.click[1]},
                      meta={"phase": why, "plan": p.goal[:80]})

    def step(self, obs, tools, plan=None) -> Action:
        self.plan = plan
        info = obs.info or {}
        w, h = int(info.get("width", 1600)), int(info.get("height", 1000))
        t = float(obs.elapsed_s)
        img = vision.decode(obs.image)
        th = vision.thumb(img)
        if self.title is None:
            self.title = th
        if vision.change(self.prev_thumb, th) > STILL:
            self.last_motion_s = t
        self.prev_thumb = th
        on_title = self.title_reliable and vision.change(self.title, th) < MENU_SIM

        if self.phase == "start":
            if on_title and t - self.start_since_s >= START_GIVEUP_S:
                # A sparse screen (a few sprites on black) can start without the thumbnail
                # moving off the title: play anyway, and leave game-over handling to STUCK_S.
                self.title_reliable = on_title = False
            if not on_title:
                self.phase, self.menu_since_s = "play", None
            elif t - self.last_opener_s >= START_GAP_S:
                return self.plan_click(t, "start-plan") or self.opener(img, w, h, t, "start")
            else:
                return Action("skip", meta={"phase": "start"})

        # play
        if on_title:
            self.menu_since_s = t if self.menu_since_s is None else self.menu_since_s
            if t - self.menu_since_s >= MENU_BACK_S:
                self.phase, self.opener_i, self.start_since_s = "start", 0, t
                return self.opener(img, w, h, t, "start")
        else:
            self.menu_since_s = None
        # A planner that mistakes a busy game screen with a Restart button for game over would
        # restart the game every few seconds: during play, only click once the screen is still.
        if (plan is not None and plan.screen in ("menu", "game_over")
                and t - self.last_motion_s >= PLAN_CLICK_STILL_S):
            act = self.plan_click(t, "play-plan")
            if act is not None:
                return act
        if t - self.last_motion_s > STUCK_S and t - self.last_opener_s > STUCK_S:
            return self.opener(img, w, h, t, "stuck")

        if t < self.program_until_s - 0.2:
            return Action("skip", meta={"phase": "play"})
        act = self.play_program(img, w, h, t)
        self.program_until_s = t + PLAY_PROGRAM_MS / 1000
        return act

    def play_program(self, img, w: int, h: int, t: float = 0.0) -> Action:
        b, rng, p = self.brief, self.rng, self.plan
        aim = p.aim if p and p.aim and t - p.made_at_s <= AIM_FRESH_S and rng.random() < 0.7 else None
        move_keys = (p.move_keys if p and p.move_keys else None) or b.move_keys
        keys: list[str] = []
        if p and p.move_keys:            # the plan's direction: hold its keys together ("up and right")
            keys += p.move_keys[:2]
        elif move_keys:
            keys.append(rng.choice(move_keys))
        extra = (p.action_keys if p and p.action_keys else None) or [
            k for k in b.action_keys if k not in b.start_keys and k not in ("Escape", "Tab")]
        if extra and rng.random() < 0.5:
            keys.append(rng.choice(extra))
        if not keys and not b.uses_pointer:
            keys = [rng.choice(brief_mod.ARROWS + ["Space"])]

        keys += [k for k in b.hold_keys if k not in keys]   # throttle and the like: always down
        if b.hold_release and not keys:   # hold the button for a random time, then let go
            x, y = aim or (w // 2, h // 2)
            hold = rng.choice((250, 400, 600, 800, 1000, 1300))
            return Action("sequence", {"steps": [{"keys": [], "duration_ms": hold, "x": x, "y": y, "buttons": ["left"]},
                                                 {"keys": [], "duration_ms": 900, "x": x, "y": y}]},
                          meta={"phase": "play", "hold_ms": hold})
        if b.taps and keys:   # one grid step per press: a burst of taps, not a long hold
            steps = []
            for _ in range(PLAY_PROGRAM_MS // 250):
                steps += [{"keys": [rng.choice(move_keys or keys)], "duration_ms": 120},
                          {"keys": list(b.hold_keys), "duration_ms": 130}]
            return Action("sequence", {"steps": steps}, meta={"phase": "play", "taps": len(steps) // 2})
        step: dict = {"keys": keys, "duration_ms": PLAY_PROGRAM_MS}
        if b.uses_pointer or aim or (p and p.hold_fire):
            x, y = aim or rng.choice(vision.salient_points(img))
            step.update(x=x, y=y)
            if b.uses_hold_fire or (p and p.hold_fire):
                step["buttons"] = ["left"]
            elif not keys:
                return Action("click", {"x": x, "y": y}, meta={"phase": "play"})
        meta = {"phase": "play", "keys": keys}
        if p is not None:
            meta["plan"] = p.goal[:80]
        return Action("sequence", {"steps": [step]}, meta=meta)
