"""Qwen3.8-Flash-Next on SGLang (Pennyroyal v2.5.3 fork), served on localhost for the planner.

Taken from the public notebook "GGA: Qwen3.8 Flash Next agent" by nihilisticneuralnet
(https://www.kaggle.com/code/nihilisticneuralnet/gga-qwen3-8-flash-next-agent), which reuses
dfranzen's serving stack from the ARC-AGI-3 milestone-2 solution (built on Tufa Labs' Duck harness).
Its input discovery and launch code are kept unchanged; notebooks/build_notebook.py --serve flashnext
inlines this file as one cell. Needs: WORKING_DIR and NOTEBOOK_START_TIME defined earlier, inputs
dfranzen/pennyroyal-v253 (dataset), dfranzen/intel-qwen3.8-flash-next-w4a16-autoround and
dfranzen/albucino-qwen3-8-flash-next-drafter (models), and an RTX Pro 6000.
"""
import os, sys, time
from pathlib import Path
root = Path('/kaggle/input')

# ---- inputs: the ARC-AGI-3 milestone-2 notebook's paths first, then a search by name --------------------
root = Path('/kaggle/input')


def _find_dir(preferred, *patterns):
    if preferred and Path(preferred).is_dir():
        return Path(preferred)
    for pat in patterns:
        hits = sorted({p.parent for p in root.rglob(pat)})
        if hits:
            return hits[0]
    return None


WHEELHOUSE_DIR = _find_dir('/kaggle/input/datasets/dfranzen/pennyroyal-v253')
if WHEELHOUSE_DIR is None:   # any attached dataset whose path mentions pennyroyal and that holds a requirements.lock
    WHEELHOUSE_DIR = next((p for p in sorted(root.glob('**/*pennyroyal*'))
                           if p.is_dir() and any(p.rglob('requirements.lock'))), None)
FLASH_MODEL_DIR = _find_dir('/kaggle/input/models/dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/transformers/default/1',
                            '*flash-next-w4a16*/**/model.safetensors.index.json')
FLASH_DRAFT_DIR = _find_dir('/kaggle/input/models/dfranzen/albucino-qwen3-8-flash-next-drafter/transformers/default/1',
                            '*flash-next-drafter*/**/model.safetensors.index.json')
SERVED_MODEL_NAME = 'flashnext'
SERVED_MODEL_PORT = 8001
FLASH_URL = f'http://127.0.0.1:{SERVED_MODEL_PORT}'

harness = next(root.rglob('gga/protocol.py')).parents[1]
kit = next(root.rglob('kit/wheels')).parent
print(f'rerun={RERUN}\nwheelhouse={WHEELHOUSE_DIR}\nflash={FLASH_MODEL_DIR}\ndraft={FLASH_DRAFT_DIR}')
if not (WHEELHOUSE_DIR and FLASH_MODEL_DIR):
    raise RuntimeError('Attach dfranzen/pennyroyal-v253 and dfranzen/intel-qwen3.8-flash-next-w4a16-autoround')
if FLASH_DRAFT_DIR is None:
    print('warning: MTP drafter not attached -> no speculative decoding (slower replies)')


# From GGA: QwenFN agent by nihilisticneuralnet (Version 4): https://www.kaggle.com/code/nihilisticneuralnet/gga-qwenfn-agent?scriptVersionId=373376666
# 2. Launch Qwen3.8-Flash-Next on SGLang (Pennyroyal v2.5.3 fork) in the background.
# The Pennyroyal wheelhouse is built for CPython 3.12 (cp312) only, while the current Kaggle image runs 3.13, so the
# server gets its own Python 3.12 (see find_python312); the notebook itself stays on 3.13 and only talks HTTP.
# Install / CUDA / draft-view / argument code is taken verbatim from the ARC-AGI-3 milestone-2 solution
# (dfranzen, built on Tufa Labs' Duck harness); only the sizing changed: this agent sends ONE request at a time
# with a ~3k-token prompt, so a small context and batch free memory and shorten CUDA-graph capture.
import os, sys, glob, time, shutil, subprocess, urllib.request, urllib.error, threading, signal
import contextlib, copy, hashlib, json, re, shlex, socket, zipfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

PREFIX = "/tmp/sgl-intel"
LOG = str(Path(WORKING_DIR) / "serve.log")
CFG = dict(
    CTX=32 * 1024,           # prompts here are ~1-4k tokens
    MEMFRAC=0.94,
    MAXREQ=2,
    CUDAGRAPH_MAXBS=2,
    MAMBA_CACHE=16,          # 8 slots per request (the ARC config used 6: 60 for 10)
    CHUNK=8192,
    MAX_PREFILL=16384,
    SPEC=FLASH_DRAFT_DIR is not None,   # MTP drafter: ~2-3x decode speed with greedy decoding
    FRSPEC=True,
    AUTOTUNE=True,
    PREFETCH_CHECKPOINTS=True,
    DROP_CACHE_AFTER_LOAD=True,
    LINEAR_BACKEND="flashinfer",
    KVDTYPE="fp8_e4m3",
    SSM_DTYPE="bfloat16",
    MAMBA_RADIX="extra_buffer",
    GDN_MTP_CACHE_MODE="none",
    SERVED_NAME=SERVED_MODEL_NAME,
    SPEC_ACCEPT_SINGLE=1.0,
    SPEC_ACCEPT_ACC=1.0,
    SPEC_STEPS=3,
)
VENV = f"{PREFIX}/venv"
PYTHON = f"{VENV}/bin/python"
SGLANG = f"{VENV}/bin/sglang"
TOKEN_MAP_SHA = "becfa41d394b86c26c632bea8f3c6ea64bbb76d7b238d8673c06afae21269f25"
TOKENIZER_SHA = "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523"
EXPERT_TARGET = r"re:^mtp\.layers\.0\.mlp\.experts\.[0-9]+\.(gate_proj|up_proj|down_proj)$"
DENSE_IGNORE = r"re:^(?!mtp\.layers\.0\.mlp\.experts(?:\.|$)).*"


def run(cmd, *, check=True, env=None):
    cmd = list(map(str, cmd))
    print("$", shlex.join(cmd))
    result = subprocess.run(cmd, text=True, capture_output=True, env=env)
    if result.stdout.strip(): print(result.stdout.strip()[-6000:])
    if result.stderr.strip() and result.returncode: print(result.stderr.strip()[-6000:])
    if check and result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}): {shlex.join(cmd)}")
    return result


def find_unique(root, name, *, directory=False, required=True):
    root = Path(root)
    direct = root if root.name == name else root / name
    valid = lambda p: p.is_dir() if directory else p.is_file()
    if valid(direct): return direct
    hits = sorted(p for p in root.rglob(name) if valid(p))
    if len(hits) > 1:
        raise RuntimeError(f"Multiple {name} matches under {root}; use a more specific directory: {hits}")
    if hits: return hits[0]
    if required: raise FileNotFoundError(f"{name} not found under {root}")
    return None


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8*1024*1024), b""): h.update(block)
    return h.hexdigest()


def indexed_shards(root):
    root = Path(root)
    index = json.loads((root / "model.safetensors.index.json").read_text())
    names = sorted(set(index["weight_map"].values()))
    if not names: raise RuntimeError(f"Empty weight index: {root}")
    for name in names:
        rel = Path(name)
        if rel.is_absolute() or ".." in rel.parts:
            raise RuntimeError(f"Invalid shard path: {name}")
        p = root / rel
        if not p.is_file() or p.stat().st_size < 8:
            raise RuntimeError(f"Missing/empty shard: {p}")
    return index, names


def prepare_draft_view(target, source_root, work_root):
    """Adapt metadata only; link original dataset weights without copying them."""
    target, source_root, work_root = map(Path, (target, source_root, work_root))
    configs = [source_root / "config.json"] if (source_root / "config.json").is_file() else []
    configs += sorted(p for p in source_root.rglob("config.json") if p not in configs)
    candidates = []
    for path in configs:
        cfg = json.loads(path.read_text())
        q = cfg.get("quantization_config", {})
        if q.get("quant_method") == "compressed-tensors" and "mtp_routed_experts" in q.get("config_groups", {}):
            candidates.append((path.parent, cfg))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one INT4 g32 MTP checkpoint under DRAFT_MODEL_DIR={source_root}; found {len(candidates)}")
    source, cfg = candidates[0]
    index, shards = indexed_shards(source)
    if not any(k.startswith("mtp.layers.0.mlp.experts.") for k in index["weight_map"]):
        raise RuntimeError("Draft index does not contain the expected MTP experts")
    cfg = copy.deepcopy(cfg)
    q = cfg["quantization_config"]
    group = q["config_groups"]["mtp_routed_experts"]
    w = group["weights"]
    if (w["num_bits"], w["group_size"], w["symmetric"]) != (4, 32, True):
        raise RuntimeError("Expected symmetric INT4 group32 draft experts")
    if group["targets"] not in (["RoutedExperts"], [EXPERT_TARGET]):
        raise RuntimeError(f"Unexpected draft target rules: {group['targets']}")
    if q.get("ignore", []) not in ([], [DENSE_IGNORE]):
        raise RuntimeError("Unexpected draft ignore rules")
    group["targets"] = [EXPERT_TARGET]
    q["ignore"] = [DENSE_IGNORE]
    # Content/path-specific directory makes notebook reruns idempotent without
    # overwriting an unrelated view. We never modify the read-only datasets.
    identity = json.dumps({"source": str(source.resolve()), "target": str(target.resolve()), "config": cfg}, sort_keys=True)
    suffix = hashlib.sha256(identity.encode()).hexdigest()[:16]
    view = work_root / f"draft-view-{suffix}"
    view.mkdir(parents=True, exist_ok=True)
    config_path = view / "config.json"
    config_text = json.dumps(cfg, indent=2) + "\n"
    if config_path.exists() or config_path.is_symlink():
        if config_path.is_symlink() or config_path.read_text() != config_text:
            raise RuntimeError(f"Unexpected existing draft-view config: {config_path}")
    else:
        config_path.write_text(config_text)
    links = {name: source / name for name in shards}
    links["model.safetensors.index.json"] = source / "model.safetensors.index.json"
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "preprocessor_config.json", "generation_config.json"):
        if (target / name).is_file(): links[name] = target / name
    for name, src in links.items():
        dest = view / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() or dest.is_symlink():
            if not dest.is_symlink() or dest.resolve() != src.resolve():
                raise RuntimeError(f"Unexpected existing draft-view file: {dest}")
        else:
            dest.symlink_to(src.resolve())
    print(f"Draft source: {source}\nSGLang draft view: {view} ({len(shards)} linked weight shards)")
    return view




def precache(*paths, times=1, delay=0, threads=16) -> None:
    time.sleep(delay)
    print(f'Precaching: Started...')
    start_time = time.time()

    CHUNK_SIZE = 32 * 1024 * 1024  # 128 MB

    chunks = []  # list of (path, offset, length)
    total_size = 0
    file_count = 0

    def _add_file(i: int, path: str) -> None:
        nonlocal total_size, file_count
        size = os.path.getsize(path)
        total_size += size
        file_count += 1
        for offset in range(0, max(size, 1), CHUNK_SIZE):
            length = min(CHUNK_SIZE, size - offset)
            chunks.append((i, path, offset, length))

    for i, path in enumerate(paths):
        try:
            if os.path.isfile(path):
                _add_file(i, path)
                continue
            for root, _, files in os.walk(path):
                for file_name in files:
                    file_path = os.path.join(root, file_name)
                    if os.path.isfile(file_path):
                        _add_file(i, file_path)
        except: pass

    chunks.sort()

    _local = threading.local()

    def _read_chunk(chunk: tuple) -> None:
        try:
            buf = _local.buf
        except AttributeError:
            _local.buf = buf = bytearray(CHUNK_SIZE)
        i, path, offset, _ = chunk
        try:
            with open(path, 'rb') as f:
                f.seek(offset)
                remaining = CHUNK_SIZE
                for i in range(1024):
                    n = f.readinto(buf)
                    if not n:
                        break
                    remaining -= n
                    if remaining <= 0: break
        except: pass

    with ThreadPoolExecutor(max_workers=threads) as pool:
        for _ in range(times):
            list(pool.map(_read_chunk, chunks))

    elapsed = time.time() - start_time
    print(f'Precaching: loaded {file_count} files ({total_size / 1e9:.2f} GB) in {elapsed:.2f} seconds.\n')



def _py_version(exe):
    try:
        out = subprocess.run([exe, "-c", "import sys, sysconfig, os; print(sys.version_info[0], sys.version_info[1], "
                              "int(os.path.isfile(os.path.join(sysconfig.get_paths()['include'], 'Python.h'))))"],
                             text=True, capture_output=True, timeout=30).stdout.split()
        return int(out[0]), int(out[1]), bool(int(out[2]))
    except Exception:
        return None


def _usable312(exe):
    v = _py_version(exe) if exe and os.path.isfile(exe) else None
    return v is not None and v[:2] == (3, 12) and v[2]   # headers are needed for Triton / JIT kernel builds


def find_python312():
    """A CPython 3.12 with headers: on the image, or a python-build-standalone build attached as an input."""
    cands = [shutil.which("python3.12"), "/usr/bin/python3.12", "/usr/local/bin/python3.12", "/opt/conda/bin/python3.12",
             *sorted(glob.glob("/opt/conda/envs/*/bin/python3.12")),
             *sorted(glob.glob("/root/.local/share/uv/python/cpython-3.12*/bin/python3.12")),
             "/tmp/py312/python/bin/python3.12"]
    for exe in cands:
        if _usable312(exe):
            return exe
    # python-build-standalone ("cpython-3.12.*-x86_64-unknown-linux-gnu-install_only.tar.gz"), as a tarball or as the
    # folder Kaggle unpacks it into. /kaggle/input drops exec bits, so it is copied to /tmp first.
    dest = Path("/tmp/py312")
    tars = sorted(p for p in Path("/kaggle/input").rglob("cpython-3.12*x86_64*linux-gnu*.tar.gz"))
    dirs = sorted(p.parents[1] for p in Path("/kaggle/input").rglob("python3.12")
                  if p.parent.name == "bin" and (p.parents[1] / "include").is_dir())
    if tars:
        import tarfile
        shutil.rmtree(dest, ignore_errors=True)
        dest.mkdir(parents=True)
        with tarfile.open(tars[0]) as tf:
            tf.extractall(dest, filter="tar") if sys.version_info >= (3, 12) else tf.extractall(dest)
    elif dirs:
        shutil.rmtree(dest, ignore_errors=True)
        shutil.copytree(dirs[0], dest / "python", symlinks=True)
    else:
        raise RuntimeError(
            "No Python 3.12 found. The Pennyroyal wheelhouse only has cp312 wheels. Attach a dataset with "
            "python-build-standalone 'cpython-3.12.<x>+<date>-x86_64-unknown-linux-gnu-install_only.tar.gz' "
            "(github.com/astral-sh/python-build-standalone/releases), or pin the notebook to a Python 3.12 Kaggle image.")
    exe = dest / "python/bin/python3.12"
    for p in (dest / "python/bin").iterdir():
        with contextlib.suppress(Exception):
            p.chmod(0o755)
    if not _usable312(str(exe)):
        raise RuntimeError(f"Attached Python 3.12 at {exe} does not run or lacks headers")
    return str(exe)


PY312 = find_python312()
print("server interpreter:", PY312, _py_version(PY312))


def _prepare_flashnext(MODEL_DIR, DRAFT_MODEL_DIR, WHEELHOUSE_DIR):
    # ---- validate notebook inputs / identify the offline bundle ----
    MODEL_DIR = str(Path(MODEL_DIR).expanduser().resolve())
    assert Path(MODEL_DIR).is_dir(), f"model dir not found: {MODEL_DIR}"
    assert CFG["CUDAGRAPH_MAXBS"] >= CFG["MAXREQ"], "CUDAGRAPH_MAXBS must cover MAXREQ"
    assert CFG["CHUNK"] > 0 and CFG["CHUNK"] % 64 == 0
    # Refuse to start a second server on an occupied port; do not kill other jobs.
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", int(SERVED_MODEL_PORT))) == 0:
            raise RuntimeError(f"Port {SERVED_MODEL_PORT} is occupied; stop the existing server before rerunning")
    Path(PREFIX).mkdir(parents=True, exist_ok=True)
    Path(LOG).parent.mkdir(parents=True, exist_ok=True)
    WHEELS = find_unique(WHEELHOUSE_DIR, "wheels", directory=True)
    LOCK = find_unique(WHEELHOUSE_DIR, "requirements.lock")
    sglang_wheels = sorted(WHEELS.glob("sglang-*.whl"))
    assert len(sglang_wheels) == 1, f"Expected one SGLang wheel, found {sglang_wheels}"
    wheel = sglang_wheels[0]
    install_marker = Path(PREFIX) / "installed-bundle.txt"

    # ---- one-time offline install, isolated from the old mratsim environment ----
    install_env = dict(os.environ, UV_OFFLINE="1", UV_PYTHON_DOWNLOADS="never", PYTHONNOUSERSITE="1")
    install_env.pop("PYTHONPATH", None)
    if not Path(SGLANG).exists() or not install_marker.is_file() or (_py_version(PYTHON) or (0, 0))[:2] != (3, 12):
        print("=== installing Pennyroyal (offline) ===")
        # Bootstrap uv directly from its wheel; no global pip changes/ensurepip.
        uv_wheels = sorted(WHEELS.glob("uv-*.whl"))
        assert len(uv_wheels) == 1, "Expected one bundled uv wheel"
        uv_binary = Path(PREFIX) / "bin/uv"
        uv_binary.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(uv_wheels[0]) as archive:
            members = [n for n in archive.namelist() if Path(n).name == "uv" and not n.endswith("/")]
            assert len(members) == 1, "Cannot identify the Linux uv executable in its wheel"
            binary = archive.read(members[0])
            assert binary.startswith(b"\x7fELF"), "Bundled uv executable is not a Linux ELF binary"
            uv_binary.write_bytes(binary)
        uv_binary.chmod(0o755)
        uv = [str(uv_binary)]
        if Path(PYTHON).exists() and (_py_version(PYTHON) or (0, 0))[:2] != (3, 12):
            shutil.rmtree(VENV, ignore_errors=True)   # left over from an attempt with the wrong interpreter
        if not Path(PYTHON).exists():
            run(uv + ["venv", "--python", PY312, VENV], env=install_env)
        install = uv + ["pip", "install", "--python", PYTHON, "--no-index", "--find-links", str(WHEELS)]
        run(install + ["-r", str(LOCK)], env=install_env)
        run(install + ["--reinstall", "--no-deps", str(wheel)], env=install_env)
        install_marker.write_text('install')
    else:
        print(f"Matching offline bundle already installed at {VENV}")

    # ---- bundled CUDA toolkit and linker symlinks ----
    cu13 = sorted(Path(VENV).glob("lib/python*/site-packages/nvidia/cu13"))
    if not cu13 or not (cu13[0] / "bin/nvcc").is_file():
        raise RuntimeError("Bundled CUDA 13 nvcc missing; check that the complete Pennyroyal wheelhouse was uploaded")
    CUDA_HOME = str(cu13[0])
    libdir = Path(CUDA_HOME) / "lib"
    lib64 = Path(CUDA_HOME) / "lib64"
    if not lib64.exists() and not lib64.is_symlink(): lib64.symlink_to("lib")
    for so in sorted(libdir.glob("*.so.*")):
        link = libdir / re.sub(r"\.so\..*$", ".so", so.name)
        if not link.exists() and not link.is_symlink(): link.symlink_to(so.name)
    assert (libdir / "libcudart.so").exists(), "libcudart.so missing: JIT linker cannot find CUDA runtime"
    if not (libdir / "libcuda.so").exists():
        candidates = []
        for directory in ("/usr/local/nvidia/lib64", "/usr/local/nvidia/lib", "/usr/lib/x86_64-linux-gnu", "/usr/lib64"):
            candidates += sorted(p for p in Path(directory).glob("libcuda.so*") if p.is_file())
        if not candidates:
            code = 'import torch; torch.cuda.init(); print(open("/proc/self/maps").read())'
            maps = run([PYTHON, "-c", code], env=install_env).stdout
            candidates = [Path(p) for p in re.findall(r"(/[^\s]*libcuda\.so[^\s]*)", maps)]
        if not candidates: raise RuntimeError("GPU driver libcuda.so not found")
        link = libdir / "libcuda.so"
        if link.is_symlink(): link.unlink()
        link.symlink_to(candidates[0].resolve())
    CXX = next((c for c in ("g++-15", "g++-14", "g++-13", "g++-12", "g++-11", "g++") if shutil.which(c)), None)
    assert CXX, "Host C++ compiler not found"
    CC = CXX.replace("g++", "gcc")
    assert shutil.which(CC), f"Matching C compiler missing: {CC}"

    # ---- runtime environment ----
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    cache = f"{PREFIX}/cache"
    for name in ("huggingface", "torch", "torchinductor", "triton", "flashinfer", "cuda", "sglang/jit"):
        Path(cache, name).mkdir(parents=True, exist_ok=True)
    env.update({
        "PYTHONNOUSERSITE": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1", "UV_OFFLINE": "1", "UV_PYTHON_DOWNLOADS": "never",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "CUDA_HOME": CUDA_HOME, "CUDACXX": f"{CUDA_HOME}/bin/nvcc",
        "PATH": f"{CUDA_HOME}/bin:{VENV}/bin:" + env.get("PATH", ""),
        "LD_LIBRARY_PATH": f"{CUDA_HOME}/lib64:{CUDA_HOME}/lib:" + env.get("LD_LIBRARY_PATH", ""),
        "CC": CC, "CXX": CXX, "CUDAHOSTCXX": CXX,
        "TORCH_CUDA_ARCH_LIST": "12.0",
        "HF_HOME": f"{cache}/huggingface", "XDG_CACHE_HOME": cache,
        "TORCH_HOME": f"{cache}/torch", "CUDA_CACHE_PATH": f"{cache}/cuda",
        "TORCHINDUCTOR_CACHE_DIR": f"{cache}/torchinductor", "TRITON_CACHE_DIR": f"{cache}/triton",
        "FLASHINFER_WORKSPACE_BASE": f"{cache}/flashinfer",
        "SGLANG_CACHE_DIR": f"{cache}/sglang", "SGLANG_JIT_CACHE_DIR": f"{cache}/sglang/jit",
        "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1",
        "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8", "TOKENIZERS_PARALLELISM": "false",
        "MAX_JOBS": "8", "CMAKE_BUILD_PARALLEL_LEVEL": "8",
        "FLASHINFER_NINJA_JOBS": "8", "FLASHINFER_NVCC_THREADS": "2", "TORCHINDUCTOR_COMPILE_THREADS": "8",
        "SGLANG_ENABLE_SM120_LOWM_BF16_GEMM": "1", "SGLANG_SM120_ONLINE_MXFP8": "0",
        "SGLANG_SM120_LOWM_FP8_WEIGHT": "0", "SGLANG_SM120_LM_HEAD_FP8": "0",
        "SGLANG_MAMBA_CONV_DTYPE": "bfloat16", "SGLANG_NUMA_BIND_V2": "false",
        "SGLANG_MM_PREPROCESS_DEVICE": "cpu", "NUMPY_MADVISE_HUGEPAGE": "0",
    })
    run([str(Path(CUDA_HOME) / "bin/nvcc"), "--version"], env=env)
    print("Host compiler:", CXX)
    # Report the optional later caching patch. This cell does not modify wheel code.
    run([PYTHON, "-c", "import sglang, torch; "
         "print('sglang', sglang.__version__, '| torch', torch.__version__); "
         "assert torch.cuda.is_available(), 'CUDA unavailable'; "
         "print('GPU:', torch.cuda.get_device_name(0), 'capability:', torch.cuda.get_device_capability(0))"],
        env=env)

    # ---- validate target and create the read-only draft's SGLang view ----
    target_cfg = json.loads(Path(MODEL_DIR, "config.json").read_text())
    q = target_cfg.get("quantization_config", {})
    assert q.get("quant_method") == "auto-round" and q.get("bits") == 4, "MODEL_DIR must be the Intel/albucino AutoRound INT4 checkpoint"
    _, target_shards = indexed_shards(MODEL_DIR)
    print("Target:", MODEL_DIR, "| shards:", len(target_shards), "| PLE dtype:", target_cfg.get("text_config", {}).get("ple_embedding_dtype", "checkpoint default (typically BF16)"))
    DRAFT_VIEW = None
    if CFG["SPEC"]:
        assert Path(DRAFT_MODEL_DIR).is_dir(), f"draft dir not found: {DRAFT_MODEL_DIR}"
        DRAFT_VIEW = prepare_draft_view(MODEL_DIR, DRAFT_MODEL_DIR, PREFIX)

    # ---- assemble Pennyroyal / AutoRound launch arguments ----
    graph_bs = sorted({1, 2, 4, 7, 8, 9, 10, CFG["MAXREQ"], CFG["CUDAGRAPH_MAXBS"]})
    graph_bs = [n for n in graph_bs if n <= CFG["CUDAGRAPH_MAXBS"]]
    args = [SGLANG, "serve", "--model-path", MODEL_DIR, "--load-format", "safetensors",
        "--model-loader-extra-config", '{"enable_multithread_load":false}',
        "--served-model-name", str(CFG["SERVED_NAME"]), "--host", "127.0.0.1",
        "--port", str(SERVED_MODEL_PORT), "--tensor-parallel-size", "1",
        "--dtype", "bfloat16", "--quantization", "auto-round",
        "--kv-cache-dtype", CFG["KVDTYPE"], "--mem-fraction-static", str(CFG["MEMFRAC"]),
        "--context-length", str(CFG["CTX"]), "--page-size", "64",
        "--max-running-requests", str(CFG["MAXREQ"]), "--chunked-prefill-size", str(CFG["CHUNK"]),
        "--max-prefill-tokens", str(CFG["MAX_PREFILL"]),
        "--cuda-graph-max-bs-decode", str(CFG["CUDAGRAPH_MAXBS"]),
        "--cuda-graph-bs-decode", *map(str, graph_bs),
        "--mamba-ssm-dtype", CFG["SSM_DTYPE"], "--max-mamba-cache-size", str(CFG["MAMBA_CACHE"]),
        "--mamba-radix-cache-strategy", CFG["MAMBA_RADIX"], "--mamba-track-interval", "64",
        "--mamba-backend", CFG["LINEAR_BACKEND"],
        "--linear-attn-decode-backend", CFG["LINEAR_BACKEND"],
        "--linear-attn-prefill-backend", CFG["LINEAR_BACKEND"],
        "--moe-runner-backend", "auto", "--ple-offload-embedding", "--trust-remote-code",
        "--mm-feature-transport", "cpu", "--image-processor-backend", "pil",
        "--reasoning-parser", "qwen3", "--tool-call-parser", "qwen3_coder",
        "--default-chat-template-kwargs", '{"preserve_thinking":true}',
        "--watchdog-timeout", "1800", "--schedule-policy", "lpm", "--warmups", "structured_output",
        "--enable-cache-report", "--enable-metrics", "--enable-request-time-stats-logging"]
    if CFG["PREFETCH_CHECKPOINTS"]: args += ["--weight-loader-prefetch-checkpoints"]
    if CFG["DROP_CACHE_AFTER_LOAD"]: args += ["--weight-loader-drop-cache-after-load"]
    tmpl = Path(MODEL_DIR, "chat_template.jinja")
    if tmpl.is_file(): args += ["--chat-template", str(tmpl)]
    if not CFG["AUTOTUNE"]: args += ["--disable-flashinfer-autotune"]
    if CFG["GDN_MTP_CACHE_MODE"]: args += ["--gdn-mtp-cache-mode", CFG["GDN_MTP_CACHE_MODE"]]
    if CFG["SPEC"]:
        args += ["--speculative-algorithm", "NEXTN", "--speculative-num-steps", str(CFG["SPEC_STEPS"]),
            "--speculative-eagle-topk", "1", "--speculative-num-draft-tokens", str(CFG["SPEC_STEPS"]+1),
            "--speculative-draft-model-path", str(DRAFT_VIEW),
            "--speculative-draft-model-quantization", "compressed-tensors", "--speculative-moe-runner-backend", "auto",
            "--speculative-draft-kv-cache-dtype", CFG["KVDTYPE"],
            "--speculative-accept-threshold-single", str(CFG["SPEC_ACCEPT_SINGLE"]),
            "--speculative-accept-threshold-acc", str(CFG["SPEC_ACCEPT_ACC"])]
        if CFG["FRSPEC"]:
            tok = find_unique(WHEELHOUSE_DIR, "hot_tokens_64k.pt", required=False)
            if tok is None: tok = find_unique(WHEELHOUSE_DIR, "flash-next-64k.pt")
            assert sha256(tok) == TOKEN_MAP_SHA, "Wrong FR-Spec map; use the map from the pinned Pennyroyal bundle"
            assert sha256(Path(MODEL_DIR, "tokenizer.json")) == TOKENIZER_SHA, "Tokenizer differs from the FR-Spec map; set CFG['FRSPEC']=False"
            args += ["--speculative-token-map", str(tok)]


    return args, env


def _flash_healthy(timeout=2.0):
    try:
        with urllib.request.urlopen(f"{FLASH_URL}/health", timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


FLASH_PROC = None
FLASH_ARGS = FLASH_ENV = None


def launch_flash():
    """Start (or restart) the SGLang server process in the background; returns the Popen."""
    global FLASH_PROC
    with open(LOG, "ab", buffering=0) as logf:
        FLASH_PROC = subprocess.Popen(FLASH_ARGS, env=FLASH_ENV, stdout=logf, stderr=subprocess.STDOUT,
                                      start_new_session=True)
    print(f"SGLang pid {FLASH_PROC.pid} -> {LOG} (launched at {time.time() - NOTEBOOK_START_TIME:.0f}s)")
    return FLASH_PROC


def show_log_tail(n=6000):
    with contextlib.suppress(Exception):
        with open(LOG, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - n))
            print(f.read().decode(errors="replace"))


if _flash_healthy():
    print("Flash-Next server already running; reusing it")
else:
    precache(str(WHEELHOUSE_DIR))   # wheels first (the install reads them), weights stream in meanwhile
    FLASH_ARGS, FLASH_ENV = _prepare_flashnext(str(FLASH_MODEL_DIR), str(FLASH_DRAFT_DIR or ""), str(WHEELHOUSE_DIR))
    threading.Thread(target=precache, args=(str(FLASH_MODEL_DIR), str(FLASH_DRAFT_DIR or FLASH_MODEL_DIR)),
                     kwargs={"threads": 3}, daemon=True).start()
    print("\n" + shlex.join(FLASH_ARGS) + "\n")
    launch_flash()
