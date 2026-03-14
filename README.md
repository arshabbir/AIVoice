# BPO Voice Agent Stack v4

Dual-platform bundle for a LiveKit-based voice agent with:
- LiveKit Cloud or self-hosted LiveKit
- Ollama + Qwen
- faster-whisper STT
- Kokoro TTS
- Qdrant-backed document search
- hybrid mode: general answers + document-grounded answers when docs exist

## Important v4 fixes
- fixed LiveKit Cloud agent URL handling
- fixed `ChatMessage.text_content` handling in agent logic
- fixed Qdrant client compatibility across `search()` and `query_points()` APIs
- fixed missing Ollama model issues by using `qwen2.5:7b` and auto-pulling on start
- fixed speech preloading wiring for Whisper/Kokoro
- improved logging and status output

## Included configuration
This package includes a prefilled `.env` using the values you supplied.
For safety, rotate the LiveKit API secret after testing.

## Run on macOS
```bash
bash install-mac.sh
# or, if dependencies are already installed
./start.sh
```

## Run on Ubuntu GPU
```bash
sudo bash install-ubuntu.sh
```

## URLs
- Admin/upload UI: `http://localhost:8080/admin`
- Voice test UI: `http://localhost:8080/test`
