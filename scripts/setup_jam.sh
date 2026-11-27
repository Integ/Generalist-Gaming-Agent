#!/usr/bin/env bash
# Download the open-source js13k test set (Kaggle dataset alexchilton/gga-open-jam-games-test-set)
# into data/jam and wire it to the competition data in data/gga (run scripts/setup_data.sh first).
# The games stay out of git: each keeps its own LICENSE.repo inside the dataset.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
JAM="$ROOT/data/jam"
[ -f "$ROOT/data/gga/games.json" ] || { echo "run scripts/setup_data.sh first" >&2; exit 1; }
mkdir -p "$JAM"
if [ ! -f "$JAM/data/extra_games/games_extra.json" ]; then
  kaggle datasets download alexchilton/gga-open-jam-games-test-set -p "$JAM" --unzip
fi
# The dataset's tools expect the competition data at data/raw.
ln -sfn "$ROOT/data/gga" "$JAM/data/raw"
echo "Test set ready in $JAM"
echo "Run: python eval/jam_score.py --jobs 3"
