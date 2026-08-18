#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${RDSYNTH_VENV_DIR:-venv}"
PYTHON_EXE="${REPO_ROOT}/${VENV_DIR}/bin/python"

if [[ ! -x "${PYTHON_EXE}" ]]; then
  echo "Virtual environment not found at '${VENV_DIR}'. Create it first, then rerun." >&2
  exit 1
fi

cd "${REPO_ROOT}"

if [[ $# -eq 0 ]]; then
  exec "${PYTHON_EXE}" --version
fi

exec "${PYTHON_EXE}" "$@"
