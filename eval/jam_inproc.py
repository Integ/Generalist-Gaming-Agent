"""Play the jam test set in this process, with any Agent object: for agents that live in a notebook.

eval/jam.py runs each game through `python -m gga.game`, which loads the agent from a file in a
worker process. Some agents (the public Qwen-coach baseline) are defined in notebook cells and
served in-process, so they cannot be loaded that way. run() patches the already-imported gga
package to serve the jam games instead of the competition's (as eval/jam.py's build/jam overlay
does on disk), then plays each game here: observation -> agent.step -> handle_tool, one game at
a time, a fresh agent per game.

    import jam_inproc
    rows = jam_inproc.run(lambda: Agent(), jam_root, out_dir, minutes=2)
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
import types
from pathlib import Path

import jam   # eval/jam.py: reader_specs, normalise, default_games, PAUSE


def patch_gga(jam_root: Path) -> dict:
    """Point the imported gga package at the jam games and their readers; returns the jam games."""
    import gga.browser as browser
    import gga.game as game
    import gga.spec as spec
    from gga.state import Hook, Reader, reader_js

    extra = jam_root / "data" / "extra_games"
    spec.ASSETS = browser.ASSETS = extra
    games_json = extra / "games_extra.json"
    game.load_games = lambda path=games_json: spec.load_games(Path(games_json))

    try:
        import gga.readers.pause  # noqa: F401
    except ImportError:
        pause = types.ModuleType("gga.readers.pause")
        exec(jam.PAUSE, pause.__dict__)
        sys.modules["gga.readers.pause"] = pause

    def sha(p):
        return hashlib.sha256((extra / p).read_bytes()).hexdigest()

    for gid, s in jam.reader_specs(jam_root / "tools" / "extra_games.py").items():
        js = reader_js(s["body"])
        first = Hook(path=s["path"], sha256=sha(s["path"]), replace=tuple(tuple(r) for r in s["replace"]),
                     append=f"<script>{js}</script>" if s["html"] else js)
        more = tuple(Hook(path=p, sha256=sha(p), replace=tuple(tuple(x) for x in r)) for p, r in s["more"])
        game.READERS[gid] = Reader(game=gid, hooks=(first, *more), note=s["note"])
    return json.loads(games_json.read_text())["games"]


def play(game_id: str, agent, minutes: float) -> dict:
    from gga.game import Game
    g = Game(game_id)
    steps, t0 = 0, time.time()
    try:
        g.initialize({"minutes": minutes})
        tools = g.tools()
        while not g.is_terminal():
            try:
                obs = g.get_observation()
            except Exception:
                break
            if obs.info.get("remaining_s") == 0.0:
                break
            try:
                act = agent.step(obs, tools)
            except Exception as exc:   # a broken step is a skip, as the harness plays it
                print(f"{game_id}: step error {exc!r:.200}", flush=True)
                from gga.protocol import Action
                act = Action("skip")
            g.handle_tool(act.tool, act.params)
            steps += 1
    finally:
        g.stop()
    reading = g.reading or {}
    return {"game": game_id, "steps": steps, "wall_s": round(time.time() - t0, 1),
            "best_value": reading.get("best_value", reading.get("value")), "state": reading.get("state")}


def run(make_agent, jam_root: Path, out: Path, minutes: float = 2.0, games: list[str] | None = None,
        official_games_json: Path | None = None) -> dict:
    all_games = patch_gga(Path(jam_root))
    if games is None:   # every non-trivial scored game that is not also a competition game
        official = set(json.loads(official_games_json.read_text())["games"]) if official_games_json else set()
        games = [g for g, s in all_games.items() if g not in official and not s.get("trivial")
                 and jam.normalise(s, (s.get("scoring") or {}).get("reference")) is not None]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for gid in games:
        row = play(gid, make_agent(), minutes)
        row["norm"] = jam.normalise(all_games[gid], None if row["best_value"] is None else float(row["best_value"]))
        rows.append(row)
        (out / f"{gid}.json").write_text(json.dumps(row))
        print(f"{gid:28} raw={row['best_value']} norm={row['norm']} ({row['wall_s']} s)", flush=True)
    got = [r["norm"] for r in rows if r["norm"] is not None]
    summary = {"mean": sum(got) / len(got) if got else float("nan"), "scored": len(got), "games": len(games),
               "rows": rows}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"mean normalised over {len(got)} of {len(games)}: {summary['mean']:.4f}", flush=True)
    return summary
