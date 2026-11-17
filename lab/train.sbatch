#!/bin/bash
# One GPU on the CBU compute lab. The queue allows one GPU per job.
#SBATCH --job-name=spoofdet
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/lab/home/jduran/spoofdet/logs/%x-%j.out
#SBATCH --error=/lab/home/jduran/spoofdet/logs/%x-%j.err

set -euo pipefail

cd /lab/home/jduran/spoofdet
mkdir -p logs

echo "host $(hostname)  date $(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if [[ ! -x .venv/bin/python ]] || ! .venv/bin/python -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"; then
  python3 -m venv .venv
  .venv/bin/pip install -U pip
  installed=0
  for index in cu124 cu128 cu126; do
    if .venv/bin/pip install -r requirements.txt --index-url "https://download.pytorch.org/whl/${index}"; then
      installed=1
      break
    fi
  done
  if [[ "$installed" -ne 1 ]]; then
    echo "could not install a CUDA torch wheel" >&2
    exit 1
  fi
fi

.venv/bin/python -c "import torch; print(torch.cuda.get_device_name(0), torch.__version__)"
.venv/bin/python -m spoofdet train \
  --device cuda \
  --scenes 8000 \
  --epochs 20 \
  --eval-tracks 120
