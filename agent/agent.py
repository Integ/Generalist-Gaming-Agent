"""Submission entry point: gga loads this file and builds Agent() once for all 32 games.

Two tiers: a CPU policy (ggagent.heuristic) answers every step within milliseconds, and,
when a model server is configured (GGA_MODEL_URL, or GGA_MODEL=mock for tests), a
vision-language planner (ggagent.planner) reads the screen in a background thread and
steers that policy. Without a model the agent is the CPU policy alone.

Local:   python eval/run_local.py --game off_line --minutes 1
Kaggle:  notebooks/build_notebook.py inlines this file and agent/ggagent/ into a notebook.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gga.protocol import Action, Observation, ToolSpec  # noqa: E402

from ggagent import vision  # noqa: E402
from ggagent.heuristic import HeuristicAgent  # noqa: E402
from ggagent.model import backend_from_env  # noqa: E402
from ggagent.planner import Planner  # noqa: E402


class Agent:
    """Detects game boundaries, hands each game to a fresh policy, and feeds the planner."""

    def __init__(self) -> None:
        self.policy = HeuristicAgent(seed=0)
        backend = backend_from_env()
        self.planner = Planner(backend) if backend is not None else None
        print(f"agent: planner={'on (' + type(backend).__name__ + ')' if backend else 'off'}", flush=True)
        self.last_elapsed = float("inf")
        self.games = 0

    def new_game(self, obs: Observation) -> bool:
        # A new game starts with elapsed_s back near 0, or (after a game that ended at
        # its first observation) the same elapsed_s with an empty last_action.
        return obs.elapsed_s < self.last_elapsed - 1.0 or (
            obs.elapsed_s == self.last_elapsed and not obs.last_action)

    def step(self, obs: Observation, tools: list[ToolSpec]) -> Action:
        if self.new_game(obs):
            self.games += 1
            self.policy.reset(obs.info or {})
            if self.planner:
                self.planner.new_game()
            print(f"game {self.games}: {(obs.info or {}).get('game_id', '?')}", flush=True)
        self.last_elapsed = obs.elapsed_s
        try:
            plan = None
            if self.planner:
                self.planner.observe(obs.elapsed_s, vision.decode(obs.image), obs.info or {})
                plan = self.planner.plan(obs.elapsed_s)
            return self.policy.step(obs, tools, plan=plan)
        except Exception as exc:  # never lose a turn to a bug
            print(f"policy error: {exc!r}", flush=True)
            return Action("skip", meta={"error": repr(exc)[:200]})
