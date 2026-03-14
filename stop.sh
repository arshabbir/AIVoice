#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

detect_compose() {
  if docker compose version >/dev/null 2>&1; then
    COMPOSE_CMD=(docker compose)
  elif command -v docker-compose >/dev/null 2>&1; then
    COMPOSE_CMD=(docker-compose)
  else
    echo "ERROR: neither 'docker compose' nor 'docker-compose' is available." >&2
    exit 1
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

cleanup_named_containers() {
  local names=(
    bpo-speech
    bpo-agent
    bpo-app
    bpo-ollama
    bpo-qdrant
    bpo-redis
    bpo-livekit
  )

  for name in "${names[@]}"; do
    docker rm -f "$name" >/dev/null 2>&1 || true
  done
}

cleanup_orphan_networks() {
  docker network ls --format '{{.Name}}' \
    | grep -E '^bpo_voice_agent_stack(_v[0-9_]+)?_default$|^bpo_voice_agent_stack_v[0-9_]+_default$' \
    | while read -r net; do
        docker network rm "$net" >/dev/null 2>&1 || true
      done
}

main() {
  detect_compose
  detect_platform
  build_compose_files

  log "Stopping stack (platform=$PLATFORM)"
  compose down --remove-orphans >/dev/null 2>&1 || true

  log "Cleaning fixed-name containers"
  cleanup_named_containers

  log "Cleaning orphan networks"
  cleanup_orphan_networks

  if [[ "${1:-}" == "--volumes" ]]; then
    log "Removing project volumes"
    docker volume ls --format '{{.Name}}' \
      | grep -E '^bpo_voice_agent_stack(_v[0-9_]+)?_(qdrant_storage|ollama_storage|app_data)$|^bpo_voice_agent_stack_v[0-9_]+_(qdrant_storage|ollama_storage|app_data)$' \
      | while read -r vol; do
          docker volume rm "$vol" >/dev/null 2>&1 || true
        done
  fi

  log "Stop completed"
}

main "$@"