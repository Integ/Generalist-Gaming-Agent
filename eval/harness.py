"""Assemble a local harness: the dataset's files plus our score readers.

build/harness/ links everything from data/gga except gga/, which is a copy of the dataset's
gga package whose readers/ is our eval/readers. The dataset itself stays untouched.
"""
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
READERS = ROOT / "eval" / "readers"


def prepare(data: Path, out: Path | None = None) -> Path:
    out = out or ROOT / "build" / "harness"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for item in data.iterdir():
        if item.name != "gga":
            (out / item.name).symlink_to(item.resolve())
    shutil.copytree(data / "gga", out / "gga", ignore=shutil.ignore_patterns("readers", "__pycache__"))
    (out / "gga" / "readers").symlink_to(READERS)
    return out
