"""Play games locally with the official harness and build a trace viewer for the run.

    python eval/run_local.py --game off_line --minutes 1
    python eval/run_local.py --games public --agent agent/agent.py
    python eval/run_local.py --game cavebots --no-frames      # lighter run

Wraps `python -m gga.runner` from the dataset (data/gga, see scripts/setup_data.sh):
headless Chromium from the dataset's kit/, frames saved per decision, the run written to
runs/<stamp>_<name>/, then eval/trace_viewer.py renders runs/.../index.html.
Local runs have no score unless a reader exists for the game (see eval/README.md).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("GGA_DATA", ROOT / "data" / "gga"))


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--game", help="a games.json id, e.g. off_line")
    g.add_argument("--games", help="all | public | private | comma list (gga.runner --games)")
    ap.add_argument("--agent", default=str(ROOT / "agent" / "agent.py"))
    ap.add_argument("--minutes", type=float, default=None)
    ap.add_argument("--name", default=None, help="run name (default: the agent file's stem)")
    ap.add_argument("--no-frames", action="store_true")
    args = ap.parse_args()

    if not (DATA / "games.json").is_file():
        sys.exit(f"no dataset at {DATA}: run scripts/setup_data.sh (or set GGA_DATA)")
    agent = str(Path(args.agent).resolve())
    name = args.name or Path(agent).parent.name + "-" + Path(agent).stem
    out = ROOT / "runs" / f"{time.strftime('%Y%m%d-%H%M%S')}_{name}"

    cmd = [sys.executable, "-m", "gga.runner", "--agent", agent, "--out", str(out)]
    cmd += ["--game", args.game] if args.game else ["--games", args.games]
    if args.minutes:
        cmd += ["--minutes", str(args.minutes)]
    if not args.no_frames:
        cmd.append("--frames")

    env = dict(os.environ)
    env.setdefault("GGA_HEADLESS", "1")
    browsers = DATA / "kit" / "browsers"
    if browsers.is_dir():   # the harness pins Playwright 1.60.0, whose Chromium build ships here
        env["PLAYWRIGHT_BROWSERS_PATH"] = str(browsers)
    print("+", " ".join(cmd), flush=True)
    code = subprocess.call(cmd, cwd=DATA, env=env)

    if out.is_dir():
        subprocess.call([sys.executable, str(ROOT / "eval" / "trace_viewer.py"), str(out)])
    return code


if __name__ == "__main__":
    sys.exit(main())
