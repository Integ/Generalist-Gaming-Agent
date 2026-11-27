"""Vision-language model backends for the planner.

ChatBackend talks to any OpenAI-compatible server (SGLang, vLLM, llama.cpp) on localhost;
MockBackend answers from pixels alone, so the planner's threading and parsing can be tested
without a GPU. backend_from_env() picks one:

    GGA_MODEL_URL=http://127.0.0.1:8001  GGA_MODEL_NAME=flashnext   -> ChatBackend
    GGA_MODEL=mock                                                  -> MockBackend
    neither                                                          -> None (CPU policy only)
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import time
import urllib.request

from PIL import Image


def jpeg_data_url(img: Image.Image, max_side: int = 1024, quality: int = 85) -> str:
    img = img.copy()
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


class ChatBackend:
    """One chat completion per call: system text, images, user text -> reply text."""

    def __init__(self, url: str, model: str, timeout_s: float = 30.0) -> None:
        self.url, self.model, self.timeout_s = url.rstrip("/"), model, timeout_s

    def healthy(self, timeout_s: float = 1.5) -> bool:
        for path in ("/health", "/v1/models"):
            try:
                with urllib.request.urlopen(self.url + path, timeout=timeout_s) as r:
                    if r.status == 200:
                        return True
            except Exception:
                continue
        return False

    def generate(self, system: str, images: list[Image.Image], text: str, max_tokens: int = 300) -> str:
        content = [{"type": "image_url", "image_url": {"url": jpeg_data_url(im)}} for im in images]
        content.append({"type": "text", "text": text})
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}],
            "temperature": 0.0, "top_p": 1.0, "max_tokens": int(max_tokens),
            "chat_template_kwargs": {"enable_thinking": False},
        }
        req = urllib.request.Request(self.url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
            msg = json.loads(r.read())["choices"][0]["message"]
        out = (msg.get("content") or msg.get("reasoning_content") or "").strip()
        return re.sub(r"(?s)<think>.*?</think>", "", out).strip()


class MockBackend:
    """Stands in for a model in local tests: a fixed delay, then a plan built from the brief."""

    def __init__(self, delay_s: float = 1.5) -> None:
        self.delay_s = delay_s
        self.calls = 0

    def healthy(self, timeout_s: float = 0.0) -> bool:
        return True

    def generate(self, system: str, images: list[Image.Image], text: str, max_tokens: int = 300) -> str:
        self.calls += 1
        time.sleep(self.delay_s)
        keys = [k for k in ("ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown") if k.lower() in text.lower()]
        return json.dumps({"screen": "playing", "click": None, "move_keys": keys or ["ArrowRight"],
                           "action_keys": [], "hold_fire": False, "aim": None,
                           "goal": "mock plan", "notes": f"call {self.calls}"})


def backend_from_env():
    if os.environ.get("GGA_MODEL", "").lower() == "mock":
        return MockBackend(float(os.environ.get("GGA_MOCK_DELAY_S", "1.5")))
    url = os.environ.get("GGA_MODEL_URL")
    if url:
        return ChatBackend(url, os.environ.get("GGA_MODEL_NAME", "default"),
                           float(os.environ.get("GGA_MODEL_TIMEOUT_S", "30")))
    return None
