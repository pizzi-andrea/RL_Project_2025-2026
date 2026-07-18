#!/usr/bin/env bash
set -e

# 1) System deps for OpenCV/VizDoom
apt-get update
apt-get install -y libgl1 libglib2.0-0

# 2) Miniconda (only if missing)
if [ ! -d "/root/miniconda3" ]; then
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash Miniconda3-latest-Linux-x86_64.sh -b -p /root/miniconda3
fi
source /root/miniconda3/etc/profile.d/conda.sh

# 3) Create env (idempotent-ish)
if ! conda env list | awk '{print $1}' | grep -qx "RL_PRJ"; then
  conda create -n RL_PRJ python=3.10 -y
fi
conda activate RL_PRJ

# 4) Torch + torchvision CUDA 12.1 (official wheels)
pip install -U pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 5) Project deps
pip install numpy vizdoom opencv-python pandas matplotlib seaborn torchmetrics lightning tqdm timm rich

# 6) Smoke tests
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -c "import vizdoom; print('vizdoom OK')"
python -c "from torchvision.transforms.v2 import Compose, Resize, ToImage, ToDtype; print('tv v2 OK')"

echo "✅ Setup complete"