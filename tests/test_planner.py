"""Planner, model client and plan-following policy, without a GPU.

    GGA_DATA=data/gga python -m pytest tests -q
"""
from __future__ import annotations

import io
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "agent"), os.environ.get("GGA_DATA", str(ROOT / "data" / "gga"))]

from PIL import Image  # noqa: E402

from ggagent import planner as planner_mod  # noqa: E402
from ggagent.heuristic import HeuristicAgent  # noqa: E402
from ggagent.model import ChatBackend, MockBackend  # noqa: E402
from ggagent.planner import Plan, Planner, parse_plan  # noqa: E402

INFO = {"description": "Shoot ships.", "controls": "Click Play. Arrow keys move; hold the left mouse button to fire.",
        "width": 1600, "height": 1000, "remaining_s": 600}


def jpeg(color=(30, 30, 30), size=(1600, 1000)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="JPEG")
    return buf.getvalue()


def obs(t: float, color=(30, 30, 30)):
    return SimpleNamespace(image=jpeg(color), elapsed_s=t, info=INFO, last_action={"ok": True})


def test_parse_plan_scales_and_filters():
    text = 'Sure! {"screen":"menu","click":[500,625],"move_keys":["left","Bogus Key"],' \
           '"action_keys":["space"],"hold_fire":true,"aim":[1200,5],"goal":"press play"}'
    p = parse_plan(text, 3.0, 1600, 1000)
    assert p.screen == "menu" and p.click == (800, 625)
    assert p.move_keys == ["ArrowLeft"] and p.action_keys == ["Space"]
    assert p.aim is None                    # off the 0-1000 grid
    assert p.hold_fire and p.goal == "press play"


def test_parse_plan_rejects_garbage():
    assert parse_plan("no json here", 0, 100, 100) is None
    assert parse_plan("{not json}", 0, 100, 100) is None


def test_plan_expires():
    p = Plan(made_at_s=10.0)
    assert p.fresh(15.0) and not p.fresh(10.0 + planner_mod.TTL_S + 0.1)


def test_planner_runs_in_background_without_blocking():
    pl = Planner(MockBackend(delay_s=0.3))
    start = time.monotonic()
    pl.observe(1.0, Image.new("RGB", (160, 100)), INFO)
    assert time.monotonic() - start < 0.05          # observe() never waits for the model
    assert pl.plan(1.0) is None
    for _ in range(50):
        if pl.plan(1.5):
            break
        time.sleep(0.05)
    p = pl.plan(1.5)
    assert p is not None and p.move_keys and p.latency_s >= 0.3
    pl.new_game()
    assert pl.plan(1.5) is None                     # a new game drops the old plan


def test_planner_survives_backend_errors(monkeypatch):
    class Broken:
        def generate(self, *a, **k):
            raise ConnectionError("server down")
    monkeypatch.setattr(planner_mod, "ERROR_BACKOFF_S", 0.05)
    pl = Planner(Broken())
    pl.observe(1.0, Image.new("RGB", (16, 10)), INFO)
    time.sleep(0.3)
    assert pl.failures >= 1 and pl.plan(1.0) is None


def test_policy_clicks_where_the_plan_says_on_a_menu():
    pol = HeuristicAgent()
    pol.reset(INFO)
    plan = Plan(made_at_s=0.0, screen="menu", click=(800, 625), goal="click play")
    act = pol.step(obs(0.2), [], plan=plan)
    assert act.tool == "click" and act.params == {"x": 800, "y": 625}


def test_policy_follows_plan_keys_and_aim_while_playing():
    pol = HeuristicAgent()
    pol.reset(INFO)
    pol.step(obs(0.2, (30, 30, 30)), [])             # the title frame
    plan = Plan(made_at_s=1.0, screen="playing", move_keys=["ArrowUp"], hold_fire=True, aim=(100, 200))
    steps = []
    for i in range(6):   # a different screen: the game runs; one program a second
        act = pol.step(obs(1.0 + i, (200, 50 + 20 * i, 50)), [], plan=plan)
        assert act.tool == "sequence"
        steps.append(act.params["steps"][0])
    assert all(s["buttons"] == ["left"] for s in steps)
    assert sum(s["keys"][0] == "ArrowUp" for s in steps) >= 3     # the plan's keys, most of the time
    aimed = [(s["x"], s["y"]) == (100, 200) for s in steps]
    assert any(aimed[:3])            # a fresh aim is followed (most of the time)
    assert not any(aimed[4:])        # an aim older than AIM_FRESH_S is not


class _FakeOpenAI(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        self.send_response(200 if self.path == "/health" else 404)
        self.end_headers()

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert body["messages"][1]["content"][0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
        reply = {"choices": [{"message": {"content": '<think>hm</think>{"screen":"playing"}'}}]}
        data = json.dumps(reply).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def test_chat_backend_against_a_fake_server():
    server = HTTPServer(("127.0.0.1", 0), _FakeOpenAI)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        be = ChatBackend(f"http://127.0.0.1:{server.server_port}", "m")
        assert be.healthy()
        out = be.generate("sys", [Image.new("RGB", (64, 40))], "hi")
        assert out == '{"screen":"playing"}'
    finally:
        server.shutdown()
