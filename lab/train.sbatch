#!/bin/bash
# One GPU on the CBU compute lab. The queue allows one GPU per job.
# The 5090s are Blackwell, so an older CUDA 12.4 wheel is not enough.
#SBATCH --job-name=ridcheck
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/lab/home/jduran/ridcheck/logs/%x-%j.out
#SBATCH --error=/lab/home/jduran/ridcheck/logs/%x-%j.err

set -euo pipefail

cd /lab/home/jduran/ridcheck
mkdir -p logs

echo "host $(hostname)  arch $(uname -m)  date $(date -Is)"
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
  if [[ "$index" == "pypi" ]]; then
    .venv/bin/pip install -r requirements.txt
  else
    .venv/bin/pip install -r requirements.txt --index-url "https://download.pytorch.org/whl/${index}"
  fi
}

if [[ ! -x .venv/bin/python ]] || ! cuda_ok; then
  rm -rf .venv
  if ! python3 -m venv .venv; then
    python3 -m venv --without-pip .venv
    curl -fsSL https://bootstrap.pypa.io/get-pip.py | .venv/bin/python
  fi
  .venv/bin/pip install -U pip
  installed=0
  for index in cu128 cu130 cu126 pypi; do
    echo "trying torch index ${index}"
    if install_torch "$index" && cuda_ok; then
      installed=1
      break
    fi
    echo "torch from ${index} does not run on this GPU" >&2
  done
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
