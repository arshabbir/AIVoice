# Configuration notes

- `ANSWER_MODE=hybrid` allows general answers when no uploaded documents are available, and grounded answers when documents are available and relevant.
- `OLLAMA_MODEL=qwen2.5:7b` is the default for v4 because the previously used `qwen2.5:7b-instruct` was not available in the tested Ollama environment.
- `LIVEKIT_PUBLIC_URL` is what the browser uses.
- `LIVEKIT_AGENT_URL` is what the agent container uses.
- `PRELOAD_WHISPER=true` and `PRELOAD_KOKORO=true` reduce first-request latency in CPU test mode.
- `OLLAMA_PULL_ON_START=true` ensures the requested model is pulled automatically if missing.
