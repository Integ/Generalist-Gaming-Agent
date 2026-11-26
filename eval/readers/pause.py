"""Stand-in for the withheld gga/readers/pause.py: no in-game pause policy locally."""
from __future__ import annotations


def without_pause(reader):
    return reader


def pause_input_script(game_id: str):
    return None
