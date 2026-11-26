#!/usr/bin/env bash
# Download the competition dataset into data/gga and install the local-play dependencies.
# Needs Kaggle credentials (KAGGLE_API_TOKEN, or KAGGLE_USERNAME + KAGGLE_KEY) and Python >= 3.11.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="$ROOT/data/gga"
mkdir -p "$DATA"

command -v kaggle >/dev/null || pip install -q kaggle
if [ ! -f "$DATA/games.json" ]; then
  kaggle competitions download generalist-gaming-agent -p "$DATA"
  unzip -q -o "$DATA/generalist-gaming-agent.zip" -d "$DATA"
  rm -f "$DATA/generalist-gaming-agent.zip"
fi
# The Kaggle zip loses exec bits: restore them on the bundled browsers.
find "$DATA/kit/browsers" -type f \( -name "chrome*" -o -name "*.so*" -o -name "ffmpeg*" -o -name "*crashpad*" \) -exec chmod +x {} +

# Playwright 1.60.0 and its Chromium ship in the dataset's kit/ (the harness pins that version).
pip install -q "$DATA"/kit/wheels/playwright-*.whl "$DATA"/kit/wheels/pyee-*.whl \
  "$DATA"/kit/wheels/greenlet-*.whl "$DATA"/kit/wheels/typing_extensions-*.whl \
  pillow numpy fastapi uvicorn 2>/dev/null || pip install -q playwright==1.60.0 pillow numpy fastapi uvicorn

echo "Data ready in $DATA"
echo "Run: python eval/run_local.py --game off_line --minutes 1"
