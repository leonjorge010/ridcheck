#!/bin/bash
# One GPU on the CBU compute lab. The queue allows one GPU per job.
# The 5090s are Blackwell, so an older CUDA 12.4 wheel is not enough.
#SBATCH --job-name=ridcheck
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/lab/home/jduran/ridcheck/logs/%x-%j.out
#SBATCH --error=/lab/home/jduran/ridcheck/logs/%x-%j.err

set -euo pipefail

cd /lab/home/jduran/ridcheck
mkdir -p logs

echo "host $(hostname)  arch $(uname -m)  python $(python3 --version)  date $(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cuda_ok() {
  .venv/bin/python - << 'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit(1)
value = (torch.zeros(4, device="cuda") + 1).sum().item()
print(torch.cuda.get_device_name(0), torch.__version__, value)
PY
}

install_torch() {
  local index="$1"
  # The CUDA wheel index has torch and the NVIDIA libraries. It does not have
  # pytest, and --index-url would hide PyPI, so the install would die before
  # torch is tested. Training does not need pytest.
  .venv/bin/pip install --upgrade "torch>=2.2" --index-url "https://download.pytorch.org/whl/${index}"
  .venv/bin/pip install "numpy>=1.26"
}

if [[ ! -x .venv/bin/python ]] || ! cuda_ok; then
  rm -rf .venv
  if ! python3 -m venv .venv; then
    python3 -m venv --without-pip .venv
    curl -fsSL https://bootstrap.pypa.io/get-pip.py | .venv/bin/python
  fi
  .venv/bin/pip install -U pip
  installed=0
  for index in cu128 cu130 cu126; do
    echo "trying torch index ${index}"
    if ! install_torch "$index"; then
      echo "torch from ${index} did not install" >&2
      continue
    fi
    if cuda_ok; then
      installed=1
      break
    fi
    echo "torch from ${index} installed, but CUDA did not run" >&2
  done
  if [[ "$installed" -ne 1 ]]; then
    echo "trying the default PyPI torch wheel" >&2
    if .venv/bin/pip install "torch>=2.2" "numpy>=1.26" && cuda_ok; then
      installed=1
    fi
  fi
  if [[ "$installed" -ne 1 ]]; then
    echo "could not install a CUDA torch wheel for this GPU" >&2
    exit 1
  fi
fi

.venv/bin/python -m ridcheck train \
  --device cuda \
  --scenes 8000 \
  --epochs 20 \
  --eval-tracks 120
