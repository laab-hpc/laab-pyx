#!/bin/bash

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${REPO_DIR}/jube/cpu-mt-v2026"

cd "${RESULTS_DIR}"
pyx-collect-profiles .
pyx-prepare-md-report . --out-dir "${REPO_DIR}/docs"

cd "${REPO_DIR}"
mkdocs build --strict -f mkdocs.yaml
