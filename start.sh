#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

detect_compose() {
  if docker compose version >/dev/null 2>&1; then
    COMPOSE_CMD=(docker compose)
  elif command -v docker-compose >/dev/null 2>&1; then
    COMPOSE_CMD=(docker-compose)
  else
    die "neither 'docker compose' nor 'docker-compose' is available"
  fi
}

detect_platform() {
  PLATFORM="mac"
  if [[ -f .stack-platform ]]; then
    PLATFORM="$(tr -d '[:space:]' < .stack-platform)"
  fi
}

build_compose_files() {
  COMPOSE_FILES=(-f docker-compose.base.yml)

  case "$PLATFORM" in
    mac)
      [[ -f docker-compose.mac.yml ]] && COMPOSE_FILES+=(-f docker-compose.mac.yml)
      ;;
    gpu|ubuntu)
      [[ -f docker-compose.gpu.yml ]] && COMPOSE_FILES+=(-f docker-compose.gpu.yml)
      ;;
    *)
      log "Unknown platform '$PLATFORM'; using base compose only."
      ;;
  esac
}

compose() {
  "${COMPOSE_CMD[@]}" "${COMPOSE_FILES[@]}" "$@"
}

load_env() {
  [[ -f .env ]] || die ".env file not found in $ROOT_DIR"

  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ -z "$line" ]] && continue
    [[ "$line" =~ ^[[:space:]]*# ]] && continue

    key="${line%%=*}"
    value="${line#*=}"

    key="$(printf '%s' "$key" | xargs)"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"

    if [[ "$value" =~ ^\".*\"$ ]]; then
      value="${value:1:${#value}-2}"
    elif [[ "$value" =~ ^\'.*\'$ ]]; then
      value="${value:1:${#value}-2}"
    fi

    export "$key=$value"
  done < .env
}

render_livekit_config() {
  local template="livekit/livekit.yaml.template"
  local output="livekit/livekit.yaml"

  [[ -f "$template" ]] || return 0

  mkdir -p "$(dirname "$output")"

  python3 - "$template" "$output" <<'PY'
import os
import re
import sys

template_path, output_path = sys.argv[1], sys.argv[2]
text = open(template_path, "r", encoding="utf-8").read()

def repl(match):
    key = match.group(1)
    return os.environ.get(key, match.group(0))

rendered = re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", repl, text)
with open(output_path, "w", encoding="utf-8") as f:
    f.write(rendered)
PY

  log "Rendered $ROOT_DIR/$output"
}

wait_http() {
  local name="$1"
  local url="$2"
  local attempts="${3:-60}"

  for ((i=1; i<=attempts; i++)); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      log "$name is ready"
      return 0
    fi
    sleep 2
  done

  die "$name did not become ready: $url"
}

auto_pull_ollama_model() {
  [[ -n "${OLLAMA_MODEL:-}" ]] || return 0

  log "Ensuring Ollama model is present: $OLLAMA_MODEL"
  docker exec bpo-ollama ollama pull "$OLLAMA_MODEL"
}

show_urls() {
  echo
  echo "============================================================"
  echo "Stack is up."
  echo
  echo "Admin / upload UI:"
  echo "  ${APP_PUBLIC_URL%/}/admin"
  echo
  echo "Voice test UI:"
  echo "  ${APP_PUBLIC_URL%/}/test"
  echo "============================================================"
}

main() {
  detect_compose
  detect_platform
  build_compose_files
  load_env
  render_livekit_config

  log "Pre-cleaning previous stack state"
  "$ROOT_DIR/stop.sh" >/dev/null 2>&1 || true

  log "Starting stack (platform=$PLATFORM)"
  compose up -d --build

  # Host-side readiness checks must use localhost, not Docker service names.
  wait_http "Qdrant" "http://localhost:6333/readyz" 60
  wait_http "Ollama" "http://localhost:11434/api/tags" 90
  wait_http "Speech service" "http://localhost:8001/health" 90
  wait_http "App" "http://localhost:8080/api/health" 90

  auto_pull_ollama_model

  log "Current containers:"
  docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}' | grep -E '^NAMES|^bpo-' || true

  show_urls
}

main "$@"