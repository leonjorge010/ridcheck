#!/bin/bash
# WSL only. Windows PowerShell 5.1 does not expand ~ for git, and the lab
# tunnel is on Windows. From PowerShell use lab/register-leonpc.ps1.
#
# Register this repo with a Cursor worker on LeonPC, the same machine that
# serves cbu-autonomous-robot. Run it in the Linux shell where ~/ros2_ws
# lives. Leave the robot worker running.
#
#   bash lab/register-leonpc.sh
#   bash lab/register-leonpc.sh --start

set -euo pipefail

ROOT="${RIDCHECK_DIR:-$HOME/ridcheck}"
ROBOT="${ROBOT_DIR:-$HOME/ros2_ws}"
NAME="${WORKER_NAME:-LeonPC-ridcheck}"
WIN_KEY="/mnt/c/Users/jorge/.ssh/cbu_lab"
START=0
if [[ "${1:-}" == "--start" ]]; then
  START=1
elif [[ -n "${1:-}" ]]; then
  echo "usage: bash lab/register-leonpc.sh [--start]" >&2
  exit 2
fi

if [[ ! -d "$ROOT/.git" ]]; then
  git clone https://github.com/leonjorge010/ridcheck.git "$ROOT"
else
  git -C "$ROOT" pull --ff-only
fi

mkdir -p "$HOME/.ssh"
chmod 700 "$HOME/.ssh"
if [[ ! -f "$HOME/.ssh/cbu_lab" && -f "$WIN_KEY" ]]; then
  cp "$WIN_KEY" "$HOME/.ssh/cbu_lab"
  cp "${WIN_KEY}.pub" "$HOME/.ssh/cbu_lab.pub"
  chmod 600 "$HOME/.ssh/cbu_lab"
  echo "copied the lab key from the Windows profile"
fi

if [[ ! -f "$HOME/.ssh/cbu_lab" ]]; then
  echo "missing ~/.ssh/cbu_lab. The lab portal key has to be on this machine." >&2
  exit 1
fi

echo "checking the lab login node"
if timeout 5 bash -c 'echo >/dev/tcp/10.77.0.1/2222' 2>/dev/null; then
  echo "lab login node is reachable from this shell"
else
  echo "this shell cannot reach 10.77.0.1:2222." >&2
  echo "The WireGuard tunnel is up on Windows. WSL needs mirrored networking" >&2
  echo "so this worker uses that tunnel. In /etc/wsl.conf:" >&2
  echo "  [wsl2]" >&2
  echo "  networkingMode=mirrored" >&2
  echo "Then run: wsl --shutdown" >&2
  echo "and start this script again." >&2
  exit 1
fi

if ! command -v agent >/dev/null 2>&1; then
  echo "install the Cursor CLI, then run this again:" >&2
  echo "  curl https://cursor.com/install -fsS | bash" >&2
  echo "  agent login" >&2
  exit 1
fi

cat <<EOF
Checkout: $ROOT
Robot checkout: $ROBOT
Start a second worker so the robot worker can stay up:

  agent worker start --name "$NAME" --worker-dir "$ROOT"

In Cursor, start the next agent on $NAME under My Machines.
That agent's terminal runs on LeonPC and can submit lab jobs with:

  bash lab/submit.sh --wait
EOF

if [[ "$START" -eq 1 ]]; then
  exec agent worker start --name "$NAME" --worker-dir "$ROOT"
fi
