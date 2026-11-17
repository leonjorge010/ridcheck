#!/bin/bash
# Copy this repo to the CBU compute lab and submit lab/train.sbatch.
# Run it on the laptop, with the lab WireGuard tunnel active.
# The tunnel's private key stays on that laptop; this script never asks for it.
#
#   bash lab/submit.sh          # submit and print the job id
#   bash lab/submit.sh --wait   # submit, wait, and copy the weights back

set -euo pipefail

LAB_USER=${LAB_USER:-jduran}
LAB_HOST=${LAB_HOST:-10.77.0.1}
LAB_PORT=${LAB_PORT:-2222}
LAB_IDENTITY=${LAB_IDENTITY:-$HOME/.ssh/cbu_lab}
REMOTE=${REMOTE:-/lab/home/jduran/ridcheck}
EXPECT_FP=${EXPECT_FP:-SHA256:w3zDp2wg5VOp6Y8Z/yCv6LflPLi+cnkMTJqbJfvh5A8}

ROOT=$(cd "$(dirname "$0")/.." && pwd)
WAIT=0
if [[ "${1:-}" == "--wait" ]]; then
  WAIT=1
elif [[ -n "${1:-}" ]]; then
  echo "usage: bash lab/submit.sh [--wait]" >&2
  exit 2
fi

if [[ ! -f "$LAB_IDENTITY" ]]; then
  echo "missing ${LAB_IDENTITY}. The portal key is ~/.ssh/cbu_lab, not id_ed25519." >&2
  exit 1
fi

target="${LAB_USER}@${LAB_HOST}"
identity_opts=(-i "$LAB_IDENTITY" -o IdentitiesOnly=yes)

echo "checking ${target} port ${LAB_PORT}"
scan=$(ssh-keyscan -p "$LAB_PORT" -T 10 "$LAB_HOST" 2>/dev/null || true)
if [[ -z "$scan" ]]; then
  echo "login node did not answer. Activate the lab WireGuard tunnel and try again." >&2
  exit 1
fi
fps=$(printf '%s\n' "$scan" | ssh-keygen -lf - 2>/dev/null | awk '{print $2}')
ok=0
while read -r fp; do
  if [[ "$fp" == "$EXPECT_FP" ]]; then
    ok=1
  fi
done <<< "$fps"
if [[ "$ok" -ne 1 ]]; then
  echo "host key does not match the lab fingerprint ${EXPECT_FP}" >&2
  echo "saw:" >&2
  printf '%s\n' "$fps" >&2
  echo "do not continue; tell the lab admins." >&2
  exit 1
fi

known=$(mktemp)
trap 'rm -f "$known"' EXIT
printf '%s\n' "$scan" > "$known"
ssh_base=(ssh -p "$LAB_PORT" "${identity_opts[@]}" -o ConnectTimeout=12 \
  -o "UserKnownHostsFile=${known}" -o StrictHostKeyChecking=yes)
rsync_ssh="ssh -p ${LAB_PORT} -i ${LAB_IDENTITY} -o IdentitiesOnly=yes -o UserKnownHostsFile=${known} -o StrictHostKeyChecking=yes"

echo "copying the project to ${REMOTE}"
"${ssh_base[@]}" "$target" "mkdir -p ${REMOTE}/logs"
rsync -az -e "$rsync_ssh" \
  --exclude .venv --exclude .git --exclude __pycache__ --exclude '*.pyc' --exclude runs \
  "${ROOT}/" "${target}:${REMOTE}/"

job=$("${ssh_base[@]}" "$target" "cd ${REMOTE} && sbatch --parsable lab/train.sbatch")
echo "submitted job ${job}"
echo "watch it with: ssh -i ${LAB_IDENTITY} -o IdentitiesOnly=yes -p ${LAB_PORT} ${target} squeue --me"

if [[ "$WAIT" -eq 1 ]]; then
  echo "waiting for job ${job}"
  "${ssh_base[@]}" "$target" "while squeue -h -j ${job} | grep -q .; do sleep 15; done"
  rsync -az -e "$rsync_ssh" \
    "${target}:${REMOTE}/results/" "${ROOT}/results/"
  rsync -az -e "$rsync_ssh" \
    "${target}:${REMOTE}/firmware/src/model_weights.h" "${ROOT}/firmware/src/model_weights.h"
  echo "copied results/ and firmware/src/model_weights.h back"
  "${ssh_base[@]}" "$target" "tail -n 40 ${REMOTE}/logs/ridcheck-${job}.out"
fi
