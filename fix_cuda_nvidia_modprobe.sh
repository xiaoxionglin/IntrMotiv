#!/usr/bin/env bash
set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  echo "Run this with sudo: sudo bash $0"
  exit 1
fi

echo "Before repair:"
stat -c '%U %G %a %n' /usr/bin/nvidia-modprobe || true
ls -la /dev/nvidia* 2>/dev/null || true

echo
echo "Reinstalling nvidia-modprobe to match the loaded 580 driver when available..."
apt-get install --reinstall -y nvidia-modprobe=580.105.08-0ubuntu1 || apt-get install --reinstall -y nvidia-modprobe

echo
echo "Restoring helper ownership and setuid bit..."
chown root:root /usr/bin/nvidia-modprobe
chmod 4755 /usr/bin/nvidia-modprobe
stat -c '%U %G %a %n' /usr/bin/nvidia-modprobe

echo
echo "Creating NVIDIA device nodes through the NVIDIA helper..."
nvidia-modprobe -u -c=0
ls -la /dev/nvidia*

echo
echo "Checking driver and SF_git PyTorch CUDA visibility..."
nvidia-smi
/home/xiaoxiong/miniforge3/envs/SF_git/bin/python - <<'PY'
import torch

print("torch", torch.__version__)
print("cuda build", torch.version.cuda)
print("torch.cuda.is_available()", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device 0", torch.cuda.get_device_name(0))
PY
