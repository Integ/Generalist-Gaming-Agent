"""brief.parse: the inputs a game's controls text names."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "agent"))
from ggagent import brief  # noqa: E402


def test_hold_key_for_throttle():
    b = brief.parse({"controls": "Press A or the left arrow to steer left, D or the right arrow to steer right, "
                                 "and hold S or the down arrow for throttle."})
    assert b.hold_keys == ["ArrowDown"]


def test_hold_and_release_pointer():
    b = brief.parse({"controls": "Hold the mouse button (or touch) to grow a bridge and release to lay it down."})
    assert b.hold_release and not b.uses_hold_fire


def test_hold_fire_stays_hold_fire():
    b = brief.parse({"controls": "Aim with the mouse and hold the mouse button to fire."})
    assert b.uses_hold_fire and not b.hold_release


def test_dead_arrows_and_cell_taps():
    b = brief.parse({"controls": "Press Space to start. W, A, S, D move one cell. The arrow keys do nothing."})
    assert b.move_keys == ["w", "a", "s", "d"] and b.taps and b.start_keys == ["Space"]
