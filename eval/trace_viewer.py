"""Render a gga.runner run directory as a static HTML trace you can open in a browser.

    python eval/trace_viewer.py runs/<stamp>_<name>

Writes runs/.../index.html (one row per game) and runs/.../<game>/trace.html: each decision
with the frame the agent saw (when the run used --frames), the action, its params, latency
and the agent's notes (Action.meta), plus the score readings when the game has a reader.
"""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

CSS = """
body{font:13px/1.4 system-ui,sans-serif;margin:16px;background:#fafafa;color:#222}
table{border-collapse:collapse}td,th{border:1px solid #ddd;padding:4px 6px;vertical-align:top}
th{background:#eee;position:sticky;top:0}img{max-width:320px;display:block}
code{white-space:pre-wrap;word-break:break-all;font-size:12px}.bad{color:#b00}.sum{margin-bottom:12px}
"""


def _load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def _short(obj, n: int = 300) -> str:
    s = json.dumps(obj, ensure_ascii=False) if not isinstance(obj, str) else obj
    return html.escape(s if len(s) <= n else s[:n] + "…")


def game_page(game_dir: Path) -> dict:
    episode = {}
    if (game_dir / "episode.json").is_file():
        episode = json.loads((game_dir / "episode.json").read_text())
    steps = _load_jsonl(game_dir / "steps.jsonl")
    frames = game_dir / "frames"
    rows = []
    for i, s in enumerate(steps):
        n = s.get("step", i)
        img = ""
        for cand in (frames / f"{n:04d}.jpg", frames / f"{i:04d}.jpg"):
            if cand.is_file():
                img = f'<a href="frames/{cand.name}"><img loading="lazy" src="frames/{cand.name}"></a>'
                break
        ok = "" if s.get("ok", True) else f'<div class="bad">refused: {_short(s.get("error"))}</div>'
        rows.append(
            f"<tr><td>{n}</td><td>{s.get('elapsed_s', '')}</td><td>{img}</td>"
            f"<td><b>{html.escape(str(s.get('tool')))}</b><br><code>{_short(s.get('params'))}</code>{ok}</td>"
            f"<td><code>{_short(s.get('meta') or '')}</code></td>"
            f"<td>{_short(s.get('wall_latency_s', ''))}</td>"
            f"<td><code>{_short(s.get('score') if s.get('score') is not None else '')}</code></td></tr>")
    summary = {k: episode.get(k) for k in ("status", "steps", "score", "value", "error") if k in episode}
    page = (f"<!doctype html><meta charset=utf-8><title>{game_dir.name}</title><style>{CSS}</style>"
            f"<h2>{html.escape(game_dir.name)}</h2><div class=sum><code>{_short(summary, 2000)}</code></div>"
            "<table><tr><th>#</th><th>t (s)</th><th>frame</th><th>action</th><th>meta</th>"
            f"<th>latency</th><th>score</th></tr>{''.join(rows)}</table>")
    (game_dir / "trace.html").write_text(page)
    tools: dict[str, int] = {}
    for s in steps:
        tools[str(s.get("tool"))] = tools.get(str(s.get("tool")), 0) + 1
    return {"game": game_dir.name, "steps": len(steps), "tools": tools, **summary}


def main(run: Path) -> None:
    games = [d for d in sorted(run.iterdir()) if d.is_dir() and (d / "steps.jsonl").exists()]
    rows = []
    for d in games:
        info = game_page(d)
        rows.append(f"<tr><td><a href='{d.name}/trace.html'>{d.name}</a></td><td>{info.get('status', '')}</td>"
                    f"<td>{info.get('score', '')}</td><td>{info['steps']}</td><td><code>{_short(info['tools'])}</code></td></tr>")
    (run / "index.html").write_text(
        f"<!doctype html><meta charset=utf-8><title>{run.name}</title><style>{CSS}</style><h2>{html.escape(run.name)}</h2>"
        "<table><tr><th>game</th><th>status</th><th>score</th><th>steps</th><th>tools</th></tr>"
        f"{''.join(rows)}</table>")
    print(f"trace: {run / 'index.html'}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
