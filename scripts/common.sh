#!/usr/bin/env bash

ensure_platform_file() {
  [[ -f .stack-platform ]] || { echo "ERROR: .stack-platform not found. Run install-ubuntu.sh or install-mac.sh first." >&2; exit 1; }
}

ensure_env_file() {
  [[ -f .env ]] || { echo "ERROR: .env not found. Run install-ubuntu.sh or install-mac.sh first." >&2; exit 1; }
}

compose() {
  local platform
  local -a files
  platform="$(cat .stack-platform)"
  case "$platform" in
    ubuntu-gpu)
      files=(-f docker-compose.base.yml -f docker-compose.gpu.yml)
      ;;
    mac)
      files=(-f docker-compose.base.yml -f docker-compose.mac.yml)
      ;;
    *)
      echo "ERROR: unknown platform '$platform'" >&2
      exit 1
      ;;
  esac

  if command -v docker-compose >/dev/null 2>&1; then
    docker-compose "${files[@]}" "$@"
  elif docker compose version >/dev/null 2>&1; then
    docker compose "${files[@]}" "$@"
  else
    echo "ERROR: neither 'docker-compose' nor 'docker compose' is available" >&2
    exit 1
  fi
}

render_livekit_config() {
  python3 scripts/render_livekit_config.py --root .
}

wait_http() {
  local name="$1"
  local url="$2"
  local attempts="${3:-120}"
  local delay="${4:-2}"
  for ((i=1; i<=attempts; i++)); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep "$delay"
  done
  echo "ERROR: ${name} did not become ready in time (${url})" >&2
  return 1
}

ensure_ollama_model() {
  local pull_on_start model
  pull_on_start="$(grep '^OLLAMA_PULL_ON_START=' .env | cut -d= -f2- || true)"
  [[ -n "$pull_on_start" ]] || pull_on_start="true"
  [[ "$pull_on_start" == "true" ]] || return 0

  model="$(grep '^OLLAMA_MODEL=' .env | cut -d= -f2-)"
  [[ -n "$model" ]] || return 0

  wait_http "Ollama" "http://localhost:11434/api/tags" 180 2

  if compose exec -T ollama sh -lc "ollama list | awk 'NR>1 {print \$1}' | grep -Fxq '$model'"; then
    echo "Ollama model already present: $model"
    return 0
  fi

  echo "Pulling Ollama model: $model"
  compose exec -T ollama ollama pull "$model"
}
