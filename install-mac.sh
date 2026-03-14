#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

log() {
  echo
  echo "[$(date '+%F %T')] $*"
}

fail() {
  echo
  echo "ERROR: $*" >&2
  exit 1
}

require_macos() {
  [[ "$(uname -s)" == "Darwin" ]] || fail "This installer is for macOS only."
}

ensure_homebrew() {
  if command -v brew >/dev/null 2>&1; then
    log "Homebrew already installed"
    return
  fi
  log "Installing Homebrew"
  NONINTERACTIVE=1 /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  if [[ -x /opt/homebrew/bin/brew ]]; then
    eval "$(/opt/homebrew/bin/brew shellenv)"
  elif [[ -x /usr/local/bin/brew ]]; then
    eval "$(/usr/local/bin/brew shellenv)"
  fi
}

ensure_brew_on_path() {
  if [[ -x /opt/homebrew/bin/brew ]]; then
    eval "$(/opt/homebrew/bin/brew shellenv)"
  elif [[ -x /usr/local/bin/brew ]]; then
    eval "$(/usr/local/bin/brew shellenv)"
  fi
  command -v brew >/dev/null 2>&1 || fail "brew is not on PATH after installation."
}

install_tools() {
  log "Installing Colima, Docker CLI, Compose, jq, and helper tools"
  brew update
  brew install colima docker docker-compose jq openssl curl coreutils
}

start_colima() {
  if colima status >/dev/null 2>&1; then
    if colima status | grep -q '^status: Running'; then
      log "Colima is already running"
      return
    fi
  fi
  log "Starting Colima"
  colima start --cpu 8 --memory 16 --disk 100
}

verify_docker() {
  docker info >/dev/null 2>&1 || fail "Docker engine is not reachable."
}

prepare_env() {
  log "Preparing .env for macOS CPU profile (preserving existing cloud/local values)"
  python3 scripts/write_env.py --platform mac --root "$ROOT_DIR"
  echo "mac" > .stack-platform
}

wait_http() {
  local name="$1"
  local url="$2"
  local attempts="${3:-180}"
  local delay="${4:-2}"
  log "Waiting for ${name} at ${url}"
  for ((i=1; i<=attempts; i++)); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      log "${name} is ready"
      return 0
    fi
    sleep "$delay"
  done
  fail "${name} did not become ready in time (${url})"
}

pull_model() {
  local model
  model="$(grep '^OLLAMA_MODEL=' .env | cut -d= -f2-)"
  log "Pulling Ollama model: ${model}"
  docker exec bpo-ollama ollama pull "$model"
}

main() {
  require_macos
  ensure_homebrew
  ensure_brew_on_path
  install_tools
  start_colima
  verify_docker
  prepare_env
  bash ./start.sh
  wait_http "Qdrant" "http://localhost:6333/readyz"
  wait_http "Ollama" "http://localhost:11434/api/tags"
  wait_http "Speech service" "http://localhost:8001/health"
  wait_http "App" "http://localhost:8080/api/health"
  pull_model
  bash ./status.sh
  cat <<EOF

============================================================
Stack is up in macOS CPU test mode.

Admin / upload UI:
  http://localhost:8080/admin

Voice test UI:
  http://localhost:8080/test

Note:
- This macOS profile is for functional testing, not throughput testing.
- The first model pull can take time.
============================================================
EOF
}

main "$@"
