#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ ${EUID:-$(id -u)} -ne 0 ]]; then
  exec sudo -E bash "$0" "$@"
fi

log() {
  echo
  echo "[$(date '+%F %T')] $*"
}

fail() {
  echo
  echo "ERROR: $*" >&2
  exit 1
}

require_ubuntu() {
  [[ -f /etc/os-release ]] || fail "/etc/os-release not found"
  . /etc/os-release
  [[ "${ID:-}" == "ubuntu" ]] || fail "This installer is targeted to Ubuntu 22.04/24.04. Detected: ${PRETTY_NAME:-unknown}."
  [[ "${VERSION_ID:-}" == "22.04" || "${VERSION_ID:-}" == "24.04" ]] || fail "This installer is targeted to Ubuntu 22.04/24.04. Detected: ${PRETTY_NAME:-unknown}."
}

require_nvidia() {
  command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi not found. Install NVIDIA drivers on the host first."
  log "Detected NVIDIA driver"
  nvidia-smi >/dev/null || fail "nvidia-smi failed."
}

install_prereqs() {
  log "Installing OS prerequisites"
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y         ca-certificates         curl         gnupg         lsb-release         software-properties-common         jq         openssl         unzip         python3         python3-venv
}

install_docker() {
  if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
    log "Docker and Docker Compose plugin already installed"
    return
  fi

  log "Installing Docker CE and Docker Compose plugin"
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  chmod a+r /etc/apt/keyrings/docker.gpg
  . /etc/os-release
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu ${VERSION_CODENAME} stable" > /etc/apt/sources.list.d/docker.list
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
  systemctl enable docker
  systemctl restart docker
}

install_nvidia_container_toolkit() {
  if ! command -v nvidia-ctk >/dev/null 2>&1; then
    log "Installing NVIDIA Container Toolkit"
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list |           sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g'           > /etc/apt/sources.list.d/nvidia-container-toolkit.list
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y nvidia-container-toolkit
  else
    log "NVIDIA Container Toolkit already installed"
  fi

  log "Configuring Docker runtime for NVIDIA"
  nvidia-ctk runtime configure --runtime=docker
  systemctl restart docker
}

prepare_env() {
  log "Preparing .env for Ubuntu GPU profile (preserving existing cloud/local values)"
  python3 scripts/write_env.py --platform ubuntu-gpu --root "$ROOT_DIR"
  echo "ubuntu-gpu" > .stack-platform
}

wait_http() {
  local name="$1"
  local url="$2"
  local attempts="${3:-120}"
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
  require_ubuntu
  require_nvidia
  install_prereqs
  install_docker
  install_nvidia_container_toolkit
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
Stack is up.

Admin / upload UI:
  http://localhost:8080/admin

Voice test UI:
  http://localhost:8080/test
============================================================
EOF
}

main "$@"
