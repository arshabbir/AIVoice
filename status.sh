#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
source ./scripts/common.sh
ensure_platform_file
ensure_env_file
compose ps
echo
echo "App health:"
curl -fsS http://localhost:8080/api/health || true
echo
echo "Speech health:"
curl -fsS http://localhost:8001/health || true
echo
echo "Qdrant health:"
curl -fsS http://localhost:6333/readyz || true
echo
echo "Current mode summary:"
grep -E '^(ANSWER_MODE|LIVEKIT_PUBLIC_URL|LIVEKIT_AGENT_URL|WHISPER_MODEL|WHISPER_DEVICE|MAX_AGENT_CONCURRENCY|OLLAMA_NUM_PARALLEL|OLLAMA_MODEL)=' .env || true
echo
echo "Ollama models:"
compose exec -T ollama ollama list || true
echo
