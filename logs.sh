#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
source ./scripts/common.sh
ensure_platform_file
ensure_env_file
if [[ $# -gt 0 ]]; then
  compose logs -f --tail=200 "$@"
else
  compose logs -f --tail=200
fi
