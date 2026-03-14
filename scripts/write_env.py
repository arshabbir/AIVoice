#!/usr/bin/env python3
import argparse
import secrets
from pathlib import Path

DEFAULTS = {
    "APP_PUBLIC_URL": "http://localhost:8080",
    "LIVEKIT_PUBLIC_URL": "ws://localhost:7880",
    "LIVEKIT_AGENT_URL": "ws://livekit:7880",
    "LIVEKIT_API_KEY": "devkey",
    "LIVEKIT_API_SECRET": "__GENERATE_ME__",
    "LIVEKIT_USE_EXTERNAL_IP": "false",
    "LIVEKIT_RTC_UDP_START": "50000",
    "LIVEKIT_RTC_UDP_END": "50100",
    "APP_HOST": "0.0.0.0",
    "APP_PORT": "8080",
    "APP_BASE_URL": "http://app:8080",
    "UPLOAD_DIR": "/data/uploads",
    "APP_DB_PATH": "/data/app.db",
    "QDRANT_URL": "http://qdrant:6333",
    "QDRANT_COLLECTION": "knowledge_base",
    "EMBED_MODEL": "sentence-transformers/all-MiniLM-L6-v2",
    "SEARCH_TOP_K": "4",
    "RAG_CONFIDENCE_THRESHOLD": "0.42",
    "ANSWER_MODE": "hybrid",
    "AGENT_LOG_LEVEL": "INFO",
    "APP_LOG_LEVEL": "INFO",
    "SPEECH_LOG_LEVEL": "INFO",
    "OLLAMA_BASE_URL": "http://ollama:11434/v1",
    "OLLAMA_MODEL": "qwen2.5:7b",
    "OLLAMA_TEMPERATURE": "0.2",
    "SPEECH_BASE_URL": "http://speech:8001/v1",
    "SPEECH_API_KEY": "local-speech-key",
    "STT_MODEL": "whisper-1",
    "TTS_MODEL": "kokoro",
    "KOKORO_VOICE": "af_heart",
    "GREETING_TEXT": "Hello. Ask a question.",
    "MAX_AGENT_CONCURRENCY": "20",
    "OLLAMA_NUM_PARALLEL": "4",
    "OLLAMA_MAX_LOADED_MODELS": "1",
    "OLLAMA_MAX_QUEUE": "128",
    "OLLAMA_CONTEXT_LENGTH": "4096",
    "WHISPER_MODEL": "distil-large-v3",
    "WHISPER_DEVICE": "cuda",
    "WHISPER_COMPUTE_TYPE": "float16",
    "KOKORO_LANG_CODE": "a",
    "KOKORO_DEFAULT_VOICE": "af_heart",
    "PRELOAD_WHISPER": "true",
    "PRELOAD_KOKORO": "true",
    "OLLAMA_PULL_ON_START": "true",
    "OLLAMA_GPU": "0",
    "SPEECH_GPU": "1",
}

PLATFORM_OVERRIDES = {
    "ubuntu-gpu": {
        "WHISPER_DEVICE": "cuda",
        "WHISPER_COMPUTE_TYPE": "float16",
        "MAX_AGENT_CONCURRENCY": "20",
        "OLLAMA_NUM_PARALLEL": "4",
        "WHISPER_MODEL": "distil-large-v3",
        "OLLAMA_GPU": "0",
        "SPEECH_GPU": "1",
    },
    "mac": {
        "WHISPER_DEVICE": "cpu",
        "WHISPER_COMPUTE_TYPE": "int8",
        "WHISPER_MODEL": "tiny.en",
        "MAX_AGENT_CONCURRENCY": "1",
        "OLLAMA_NUM_PARALLEL": "1",
        "OLLAMA_MAX_LOADED_MODELS": "1",
        "OLLAMA_MAX_QUEUE": "8",
        "OLLAMA_CONTEXT_LENGTH": "2048",
        "OLLAMA_GPU": "",
        "SPEECH_GPU": "",
    },
}

ORDERED_KEYS = list(DEFAULTS.keys())


def parse_env(path: Path):
    data = {}
    if not path.exists():
        return data
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key] = value
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", required=True, choices=sorted(PLATFORM_OVERRIDES))
    parser.add_argument("--root", default=".")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    env_path = root / ".env"
    existing = parse_env(env_path)

    if args.force:
        data = {**DEFAULTS, **PLATFORM_OVERRIDES[args.platform]}
    else:
        data = {**DEFAULTS, **PLATFORM_OVERRIDES[args.platform], **existing}

    if not data.get("LIVEKIT_API_SECRET") or data["LIVEKIT_API_SECRET"] == "__GENERATE_ME__":
        data["LIVEKIT_API_SECRET"] = secrets.token_hex(24)

    lines = [f"{key}={data.get(key, '')}" for key in ORDERED_KEYS]
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {env_path}")


if __name__ == "__main__":
    main()
