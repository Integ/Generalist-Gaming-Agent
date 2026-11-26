"""Cheap pixel features for a CPU policy: frame decode, change detection, salient points."""
from __future__ import annotations

import io

import numpy as np
from PIL import Image

SMALL = (64, 40)   # thumbnail used for change detection


def decode(jpeg: bytes) -> Image.Image:
    return Image.open(io.BytesIO(jpeg)).convert("RGB")


def thumb(img: Image.Image) -> np.ndarray:
    return np.asarray(img.resize(SMALL, Image.BILINEAR), dtype=np.float32) / 255.0


def change(a: np.ndarray | None, b: np.ndarray | None) -> float:
    """Mean absolute pixel change between two thumbnails, 0..1."""
    if a is None or b is None or a.shape != b.shape:
        return 1.0
    return float(np.abs(a - b).mean())


def salient_points(img: Image.Image, n: int = 6, grid: tuple[int, int] = (16, 10)) -> list[tuple[int, int]]:
    """Centres of the grid cells that stand out most from the frame's mean colour:
    a crude guess at buttons and objects worth clicking."""
    w, h = img.size
    gw, gh = grid
    small = np.asarray(img.resize((gw, gh), Image.BOX), dtype=np.float32)
    mean = small.reshape(-1, 3).mean(0)
    score = np.abs(small - mean).sum(-1)
    idx = np.argsort(score.ravel())[::-1][:n]
    pts = []
    for i in idx:
        cy, cx = divmod(int(i), gw)
        pts.append((int((cx + 0.5) * w / gw), int((cy + 0.5) * h / gh)))
    return pts
