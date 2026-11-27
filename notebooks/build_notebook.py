"""Build a Kaggle submission notebook from agent/ and optionally push it.

    python notebooks/build_notebook.py --id <user>/gga-ours                    # CPU agent only
    python notebooks/build_notebook.py --id <user>/gga-ours --serve flashnext  # + Qwen3.8-Flash-Next planner (GPU)
    python notebooks/build_notebook.py --id ... --serve flashnext --validate-minutes 2 --push

The notebook puts the competition harness on sys.path, writes agent/agent.py to
/kaggle/working/agent.py with agent/ggagent/*.py next to it, optionally starts a model server
(notebooks/serving/<name>.py) and points the agent at it (GGA_MODEL_URL), then serves the agent
unconditionally in its last cell, as the competition requires.

--validate-minutes N: in an ordinary run (not the competition rerun) the notebook first plays
the four sample games for N minutes each with our score readers (eval/readers) and prints the
scores, so a GPU run of the notebook doubles as a test. Submitting a version to the competition
is a separate step (kaggle competitions submit, see README).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AGENT = ROOT / "agent"
OUT = ROOT / "build" / "notebook"
COMPETITION = "generalist-gaming-agent"

SERVERS = {
    "flashnext": {
        "file": ROOT / "notebooks" / "serving" / "flashnext.py",
        "url": "http://127.0.0.1:8001", "model": "flashnext",
        "datasets": ["dfranzen/pennyroyal-v253"],
        # the image the public notebooks pin: its Python 3.12 runs the cp312-only Pennyroyal wheels
        "docker_image": "gcr.io/kaggle-private-byod/python@sha256:"
                        "9a6bcfda6d03b2cd9aa347ee8c1bae62043328ed0f4afe9bda014e4f8bbf9435",
        "models": ["dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/Transformers/default/1",
                   "dfranzen/albucino-qwen3-8-flash-next-drafter/Transformers/default/1"],
    },
}

SETUP = """\
# Setup: the competition harness on sys.path, Chromium and a virtual display for local play.
import os, sys, time, subprocess
from pathlib import Path
NOTEBOOK_START_TIME = time.time()
RERUN = os.getenv('KAGGLE_IS_COMPETITION_RERUN', '').strip().lower() in {'1', 'true'}
WORKING_DIR = Path('/kaggle/working')
harness = next(Path('/kaggle/input').rglob('gga/protocol.py')).parents[1]
kit = next(Path('/kaggle/input').rglob('kit/wheels')).parent
sys.path.insert(0, str(harness))
os.environ.setdefault('GGA_ROOT', str(harness))
Path('/kaggle/working/ggagent').mkdir(parents=True, exist_ok=True)
chromium = None
if not RERUN:   # only local play needs a browser; the competition runs the games itself
    chromium = subprocess.Popen(f'cp -r {kit}/browsers /tmp && chmod -R +x /tmp/browsers'
                                f' && dpkg -i {kit}/debs/*.deb > /dev/null 2>&1', shell=True)
    subprocess.run(f'pip install -q --no-index --find-links {kit}/wheels playwright', shell=True, check=True)
    subprocess.Popen(['Xvfb', ':99', '-screen', '0', '1920x1080x24'],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    os.environ.update(DISPLAY=':99', PLAYWRIGHT_BROWSERS_PATH='/tmp/browsers')
print('harness:', harness, '| rerun:', RERUN)
"""

WAIT = """\
# Wait for the model server (the 15-minute startup limit covers everything before the last cell),
# then point the agent at it. Until a plan arrives the agent plays its CPU policy.
import urllib.request
SERVE_AT_S = 12.5 * 60
def _healthy():
    try:
        with urllib.request.urlopen('{url}/health', timeout=2) as r:
            return r.status == 200
    except Exception:
        return False
while not _healthy() and time.time() - NOTEBOOK_START_TIME < SERVE_AT_S:
    time.sleep(3)
os.environ.update(GGA_MODEL_URL='{url}', GGA_MODEL_NAME='{model}')
print(f'model healthy={{_healthy()}} at {{time.time() - NOTEBOOK_START_TIME:.0f}}s')
"""

VALIDATE = """\
# Validation (ordinary runs only): play the sample games with our score readers and print the scores.
import shutil
if not RERUN:
    if chromium is not None and chromium.wait():
        print('warning: Chromium install returned an error')
    overlay = Path('/tmp/gga-local')
    shutil.rmtree(overlay, ignore_errors=True)
    overlay.mkdir()
    for item in harness.iterdir():
        if item.name != 'gga':
            (overlay / item.name).symlink_to(item)
    shutil.copytree(harness / 'gga', overlay / 'gga', ignore=shutil.ignore_patterns('readers', '__pycache__'))
    shutil.copytree(WORKING_DIR / 'eval_readers', overlay / 'gga' / 'readers')
    env = dict(os.environ, GGA_HEADLESS='1')
    env.pop('PYTHONPATH', None)
    subprocess.run([sys.executable, '-m', 'gga.runner', '--games', 'public', '--minutes', '{minutes}',
                    '--agent', str(WORKING_DIR / 'agent.py'), '--out', str(WORKING_DIR / 'validation')],
                   cwd=overlay, env=env)
    for game in sorted((WORKING_DIR / 'validation').glob('*/episode.json')):
        print(game.parent.name, json.loads(game.read_text()).get('score'))
"""

SERVE = """\
# Serve the agent: in the competition rerun this plays all 32 games; in a normal run it writes
# the placeholder submission.parquet that makes the version submittable.
from kaggle_evaluation.gga.gga_inference_server import GGAInferenceServer
GGAInferenceServer().run()
"""


def cell(src: str, kind: str = "code") -> dict:
    c = {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True)}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    return c


def writefile(path: str, src: str) -> dict:
    return cell(f"%%writefile {path}\n" + src)


def build(kernel_id: str, title: str, gpu: bool, models: list[str], datasets: list[str],
          serve: str | None, validate_minutes: float | None) -> Path:
    cells = [cell(f"# {title}\n\nGenerated by `notebooks/build_notebook.py` from the repository's `agent/`.",
                  "markdown"), cell(SETUP)]
    server = SERVERS[serve] if serve else None
    if server:
        cells.append(cell(server["file"].read_text()))
        models = models + [m for m in server["models"] if m not in models]
        datasets = datasets + [d for d in server["datasets"] if d not in datasets]
        gpu = True
    for f in sorted((AGENT / "ggagent").glob("*.py")):
        cells.append(writefile(f"/kaggle/working/ggagent/{f.name}", f.read_text()))
    cells.append(writefile("/kaggle/working/agent.py", (AGENT / "agent.py").read_text()))
    if validate_minutes:
        cells.append(cell("from pathlib import Path\nPath('/kaggle/working/eval_readers').mkdir(exist_ok=True)"))
        for f in sorted((ROOT / "eval" / "readers").glob("*.py")):
            cells.append(writefile(f"/kaggle/working/eval_readers/{f.name}", f.read_text()))
    if server:
        cells.append(cell(WAIT.format(url=server["url"], model=server["model"])))
    if validate_minutes:
        cells.append(cell("import json\n" + VALIDATE.replace("{minutes}", f"{validate_minutes:g}")))
    cells.append(cell(SERVE))
    nb = {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3",
                                                      "language": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.ipynb"):
        old.unlink()
    code_file = kernel_id.split("/")[1] + ".ipynb"
    (OUT / code_file).write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    meta = {"id": kernel_id, "title": title, "code_file": code_file, "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": gpu, "enable_tpu": False,
            "enable_internet": False, "dataset_sources": datasets, "kernel_sources": [],
            "competition_sources": [COMPETITION], "model_sources": models}
    if gpu:
        meta["machine_shape"] = "NvidiaRtxPro6000"
    if server and server.get("docker_image"):
        meta["docker_image"] = server["docker_image"]
    (OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    return OUT


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True, help="<kaggle-user>/<kernel-slug>")
    ap.add_argument("--title", default=None)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--serve", choices=sorted(SERVERS), default=None, help="start a model server for the planner")
    ap.add_argument("--validate-minutes", type=float, default=None,
                    help="in ordinary runs, play the sample games this long each and print local scores")
    ap.add_argument("--model", action="append", default=[], help="Kaggle model source, repeatable")
    ap.add_argument("--dataset", action="append", default=[], help="Kaggle dataset source, repeatable")
    ap.add_argument("--push", action="store_true")
    a = ap.parse_args()
    title = a.title or a.id.split("/")[1].replace("-", " ").title()
    out = build(a.id, title, a.gpu, a.model, a.dataset, a.serve, a.validate_minutes)
    print(f"notebook: {out}")
    if a.push:
        return subprocess.call(["kaggle", "kernels", "push", "-p", str(out)])
    return 0


if __name__ == "__main__":
    sys.exit(main())
