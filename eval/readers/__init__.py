"""Our local score readers, swapped in for the dataset's empty gga/readers package.

The official readers are withheld; these are our own approximations of each game's metric,
built on the harness's gga.state.Reader (pin a bundle file by sha256, append a JavaScript
reader that returns {state, value}). eval/harness.py links this package in as gga.readers.
Scores read locally are estimates: the evaluation's own readers and game fixes may differ.
"""
from __future__ import annotations

from .asteroid_capitalist import READER as _asteroid_capitalist
from .blarglike import READER as _blarglike
from .off_line import READER as _off_line

READERS: dict = {r.game: r for r in (_asteroid_capitalist, _blarglike, _off_line)}
