#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="${ROOT_DIR}/build-tools/zimage-train/venv"
REQ_BASE="${ROOT_DIR}/scripts/faceid/requirements_zimage_train.txt"
REQ_WITH_TORCH="${ROOT_DIR}/scripts/faceid/requirements_zimage_train_with_torch.txt"

OS_NAME="$(uname -s)"
INSTALL_TORCH="${INSTALL_TORCH:-}"
if [[ -z "${INSTALL_TORCH}" ]]; then
  if [[ "${OS_NAME}" == "Darwin" ]]; then
    INSTALL_TORCH="1"
  else
    # Vast's PyTorch templates ship with CUDA torch already installed system-wide.
    INSTALL_TORCH="0"
  fi
fi

USE_SYSTEM_SITE_PACKAGES="${USE_SYSTEM_SITE_PACKAGES:-}"
if [[ -z "${USE_SYSTEM_SITE_PACKAGES}" ]]; then
  if [[ "${OS_NAME}" == "Darwin" ]]; then
    USE_SYSTEM_SITE_PACKAGES="0"
  else
    USE_SYSTEM_SITE_PACKAGES="1"
  fi
fi

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  if [[ "${INSTALL_TORCH}" == "0" && "${USE_SYSTEM_SITE_PACKAGES}" == "1" ]]; then
    for candidate in python3 python; do
      if command -v "${candidate}" >/dev/null 2>&1; then
        if "${candidate}" - <<'PY' >/dev/null 2>&1
import torch
PY
        then
          PYTHON_BIN="${candidate}"
          break
        fi
      fi
    done
  else
    if command -v python3 >/dev/null 2>&1; then
      PYTHON_BIN="python3"
    else
      PYTHON_BIN="python"
    fi
  fi
fi

echo "[zimage-train] Root: ${ROOT_DIR}"
echo "[zimage-train] Venv: ${VENV_DIR}"
echo "[zimage-train] Python: ${PYTHON_BIN}"

REQ_FILE="${REQ_BASE}"
if [[ "${INSTALL_TORCH}" == "1" ]]; then
  REQ_FILE="${REQ_WITH_TORCH}"
fi

if [[ ! -d "${VENV_DIR}" ]]; then
  if [[ "${USE_SYSTEM_SITE_PACKAGES}" == "1" ]]; then
    "${PYTHON_BIN}" -m venv --system-site-packages "${VENV_DIR}"
  else
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
  fi
fi

"${VENV_DIR}/bin/python" -m pip install --upgrade pip setuptools wheel
"${VENV_DIR}/bin/python" -m pip install -r "${REQ_FILE}"

echo "[zimage-train] Verifying torch + device..."
"${VENV_DIR}/bin/python" - <<'PY'
import torch

print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda", torch.version.cuda)
    print("device", torch.cuda.get_device_name(0))
print("mps available", hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
PY

echo "[zimage-train] Done."
echo "[zimage-train] Activate with: source \"${VENV_DIR}/bin/activate\""
