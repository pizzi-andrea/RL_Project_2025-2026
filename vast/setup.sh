#!/bin/bash
set -euo pipefail

# Use sudo only if not root
SUDO=""
if [ "$(id -u)" -ne 0 ]; then
  SUDO="sudo"
fi

echo "[1/5] System packages (base + ViZDoom deps)..."
$SUDO apt-get update
$SUDO apt-get install -y --no-install-recommends \
  unzip dos2unix \
  build-essential cmake git \
  libsdl2-2.0-0 libsdl2-dev \
  libboost-all-dev \
  libopenal1 libopenal-dev \
  ca-certificates

echo "[2/5] Fixing .sh line endings + permissions..."
find .. -name "*.sh" -exec dos2unix {} \; || true
find .. -name "*.sh" -exec chmod +x {} \; || true

echo "[3/5] Python packages..."
python -m pip install --upgrade pip setuptools wheel

echo "[4/5] Install project requirements..."
pip install -r requirements.txt

echo "[5/5] Quick sanity check (optional)..."
python - << 'PY'
try:
    import vizdoom
    print("ViZDoom OK:", vizdoom.__version__)
except Exception as e:
    print("ViZDoom import failed:", e)
PY

echo "[DONE]"