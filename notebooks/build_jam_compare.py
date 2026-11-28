"""Build a Kaggle notebook that scores the public Qwen-coach baseline and our agent on the jam test set.

The baseline (adeleascloud/gga-baseline-qwen-coach, a fork of a public notebook) defines its agent
in notebook cells, so this keeps its cells as they are, drops its final serve cell, and appends:
eval/jam.py and eval/jam_inproc.py, our agent (agent/), and a cell that, in an ordinary run, plays
the jam games in-process with the baseline's model agent (no sample-game solvers: no jam game is a
sample) and then with ours, both on the baseline's Qwen3.8-Flash-Next server. Its serve cell comes
last again, so the notebook stays a valid submission.

    kaggle kernels pull adeleascloud/gga-baseline-qwen-coach -p <dir> -m
    python notebooks/build_jam_compare.py <dir>/gga-baseline-qwen-coach.ipynb --id <user>/gga-jam-compare --push
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "build" / "jam_compare"
JAM_DATASET = "alexchilton/gga-open-jam-games-test-set"

RUN = """\
# Jam test set (ordinary runs only): the baseline's model agent, then ours, in this process.
if not RERUN:
    if chromium.wait():
        print('warning: Chromium install returned an error')
    MANAGER.wait_until(SERVE_AT_S)
    print(f'model={MANAGER.state} at {time.time() - NOTEBOOK_START_TIME:.0f}s')
    os.environ['GGA_HEADLESS'] = '1'
    sys.path.insert(0, '/kaggle/working/jamtools')
    import importlib.util
    import jam_inproc
    jam_root = next(Path('/kaggle/input').rglob('games_extra.json')).parents[2]
    official = harness / 'games.json'
    baseline_agent = StartStepAgent()
    jam_inproc.run(lambda: baseline_agent, jam_root, WORKING_DIR / 'jam' / 'baseline',
                   minutes={minutes}, official_games_json=official)
    os.environ.update(GGA_MODEL_URL=FLASH_URL, GGA_MODEL_NAME=SERVED_MODEL_NAME)
    spec = importlib.util.spec_from_file_location('ouragent', '/kaggle/working/ours/agent.py')
    ours = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ours)
    our_agent = ours.Agent()
    jam_inproc.run(lambda: our_agent, jam_root, WORKING_DIR / 'jam' / 'ours',
                   minutes={minutes}, official_games_json=official)
"""


HYBRID = """\
# Submission agent: the sample games keep the baseline's scripted solvers (they only lift the public
# board); every other game is played by our agent (agent/), steered by the same model server.
import importlib.util
os.environ.update(GGA_MODEL_URL=FLASH_URL, GGA_MODEL_NAME=SERVED_MODEL_NAME)
_spec = importlib.util.spec_from_file_location('ouragent', '/kaggle/working/ours/agent.py')
ours = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ours)


class Agent(Dispatch):
    \"\"\"Sample games -> scripted solver; every other game -> our agent.\"\"\"
    def __init__(self):
        super().__init__(fallback=ours.Agent())
"""


def cell(src: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": src.splitlines(keepends=True)}


def writefile(path: str, src: str) -> dict:
    return cell(f"%%writefile {path}\n" + src)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline", help="the pulled baseline .ipynb (its kernel-metadata.json next to it)")
    ap.add_argument("--id", required=True)
    ap.add_argument("--minutes", type=float, default=2.0)
    ap.add_argument("--hybrid", action="store_true",
                    help="no jam run: serve the sample solvers + our agent (a submission notebook)")
    ap.add_argument("--push", action="store_true")
    a = ap.parse_args()
    src = Path(a.baseline)
    nb = json.loads(src.read_text())
    meta = json.loads((src.parent / "kernel-metadata.json").read_text())
    cells = nb["cells"]
    serve = cells[-1]
    assert "GGAInferenceServer" in "".join(serve["source"]), "the baseline's last cell should serve"
    added = [cell("from pathlib import Path\nfor d in ('jamtools', 'ours/ggagent'):\n"
                  "    Path('/kaggle/working', d).mkdir(parents=True, exist_ok=True)")]
    for name in ("jam.py", "jam_inproc.py"):
        added.append(writefile(f"/kaggle/working/jamtools/{name}", (ROOT / "eval" / name).read_text()))
    for f in sorted((ROOT / "agent" / "ggagent").glob("*.py")):
        added.append(writefile(f"/kaggle/working/ours/ggagent/{f.name}", f.read_text()))
    added.append(writefile("/kaggle/working/ours/agent.py", (ROOT / "agent" / "agent.py").read_text()))
    added.append(cell(HYBRID) if a.hybrid else cell(RUN.replace("{minutes}", f"{a.minutes:g}")))
    nb["cells"] = cells[:-1] + added + [serve]
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.ipynb"):
        old.unlink()
    slug = a.id.split("/")[1]
    (OUT / f"{slug}.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    meta.pop("id_no", None)
    meta.update(id=a.id, title=slug.replace("-", " ").title(), code_file=f"{slug}.ipynb", is_private=True)
    meta["dataset_sources"] = list(dict.fromkeys(meta.get("dataset_sources", []) + [JAM_DATASET]))
    (OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    print(f"notebook: {OUT}")
    if a.push:
        return subprocess.call(["kaggle", "kernels", "push", "-p", str(OUT)])
    return 0


if __name__ == "__main__":
    sys.exit(main())
