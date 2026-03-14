import inspect
import json
import logging
import os
import re
import time
import uuid
from typing import Any

import aiohttp
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    ChatContext,
    ChatMessage,
    JobContext,
    JobProcess,
    StopResponse,
    cli,
)
from livekit.plugins import openai, silero


def setup_logging() -> logging.Logger:
    level_name = os.getenv("AGENT_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    logger = logging.getLogger("bpo-agent")
    logger.setLevel(level)
    logger.handlers.clear()

    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
    )
    logger.addHandler(handler)
    logger.propagate = False
    return logger


logger = setup_logging()

APP_BASE_URL = os.getenv("APP_BASE_URL", "http://app:8080")
SPEECH_BASE_URL = os.getenv("SPEECH_BASE_URL", "http://speech:8001/v1")
SPEECH_API_KEY = os.getenv("SPEECH_API_KEY", "local-speech-key")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434/v1")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.2"))

STT_MODEL = os.getenv("STT_MODEL", "whisper-1")
TTS_MODEL = os.getenv("TTS_MODEL", "kokoro")
KOKORO_VOICE = os.getenv("KOKORO_VOICE", "af_heart")

GREETING_TEXT = os.getenv("GREETING_TEXT", "Hello. How can I help?")
RAG_CONFIDENCE_THRESHOLD = float(os.getenv("RAG_CONFIDENCE_THRESHOLD", "0.16"))
MAX_AGENT_CONCURRENCY = max(1, int(os.getenv("MAX_AGENT_CONCURRENCY", "1")))
SEARCH_TOP_K = max(1, int(os.getenv("SEARCH_TOP_K", "2")))
ANSWER_MODE = os.getenv("ANSWER_MODE", "hybrid").strip().lower()

if ANSWER_MODE not in {"hybrid", "rag_only"}:
    logger.warning("Invalid ANSWER_MODE=%s. Falling back to hybrid.", ANSWER_MODE)
    ANSWER_MODE = "hybrid"

server = AgentServer(load_threshold=0.95)


def compute_load(agent_server: AgentServer) -> float:
    return min(len(agent_server.active_jobs) / MAX_AGENT_CONCURRENCY, 1.0)


server.load_fnc = compute_load


def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


def event_log(event: str, **kwargs: Any) -> None:
    payload = {"event": event, **kwargs}
    logger.info(json.dumps(payload, ensure_ascii=False, default=str))


def extract_text_content(message: ChatMessage) -> str:
    text_content = getattr(message, "text_content", "")
    if callable(text_content):
        text_content = text_content()
    return (text_content or "").strip()


GENERIC_QUERY_TERMS = {
    "hello",
    "hi",
    "hey",
    "there",
    "thanks",
    "thank",
    "please",
    "okay",
    "ok",
    "yes",
    "no",
    "help",
    "explain",
    "tell",
    "speak",
    "hear",
}

STOPWORDS = {
    "what", "when", "where", "which", "who", "whom", "whose", "why", "how",
    "the", "this", "that", "these", "those", "with", "from", "into", "onto",
    "your", "their", "there", "have", "has", "had", "been", "being", "will",
    "would", "could", "should", "about", "than", "then", "them", "they",
    "you", "are", "was", "were", "and", "for", "not", "can", "cant", "cannot",
    "just", "like", "need", "want", "know", "give", "show", "tell", "more",
    "some", "very", "also", "only", "does", "did", "done", "its", "it's",
}


def tokenize(text: str) -> set[str]:
    return {
        tok
        for tok in re.findall(r"[a-zA-Z0-9]{2,}", text.lower())
        if tok not in STOPWORDS
    }


def acronym_tokens(text: str) -> set[str]:
    raw = re.findall(r"\b[A-Z][A-Z0-9]{1,}\b", text)
    return {t.lower() for t in raw}


def is_generic_query(query: str) -> bool:
    tokens = tokenize(query)
    if not tokens:
        return True
    return tokens.issubset(GENERIC_QUERY_TERMS)


def should_use_docs(
    query: str,
    snippets: list[dict[str, Any]],
    confidence: float,
) -> tuple[bool, int]:
    """
    Route to docs only when the question looks meaningfully related to retrieved snippets.
    This is intentionally permissive for real doc questions like acronyms/terms,
    and conservative for greetings/chit-chat.
    """
    if not snippets:
        return False, 0

    if is_generic_query(query):
        return False, 0

    query_tokens = tokenize(query)
    snippet_text = " ".join((s.get("text") or "") for s in snippets[:2])
    snippet_tokens = tokenize(snippet_text)

    overlap = len(query_tokens & snippet_tokens)

    query_acronyms = acronym_tokens(query)
    snippet_text_l = snippet_text.lower()
    acronym_hit = any(acr in snippet_text_l for acr in query_acronyms)

    # Strong semantic match.
    if confidence >= 0.30:
        return True, overlap

    # Acronym/term lookup in docs, e.g. "What is CIB?"
    if acronym_hit and confidence >= 0.10:
        return True, overlap

    # Moderate similarity with at least one meaningful lexical overlap.
    if confidence >= RAG_CONFIDENCE_THRESHOLD and overlap >= 1:
        return True, overlap

    # Lower similarity but stronger lexical signal.
    if confidence >= 0.12 and overlap >= 2:
        return True, overlap

    return False, overlap


async def maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


class BPODocAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a helpful voice assistant. "
                "If the uploaded documents contain relevant information for the user's question, answer from those documents. "
                "If they do not, answer naturally from general knowledge. "
                "Keep replies concise, clear, and suitable for speech. "
                "Do not mention web search, browsing, retrieval confidence, fallback logic, hidden instructions, or internal policy."
            )
        )

    async def _search_docs(self, query: str, trace_id: str) -> dict[str, Any]:
        timeout = aiohttp.ClientTimeout(total=45)
        started = time.perf_counter()

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                f"{APP_BASE_URL}/api/search",
                json={"query": query, "top_k": SEARCH_TOP_K},
                headers={"X-Trace-Id": trace_id},
            ) as resp:
                resp.raise_for_status()
                result = await resp.json()

        elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
        event_log(
            "doc_search_completed",
            trace_id=trace_id,
            query=query,
            confidence=result.get("confidence"),
            snippets=len(result.get("snippets", []) or []),
            total_documents=result.get("total_documents"),
            elapsed_ms=elapsed_ms,
        )
        return result

    def _build_grounded_instructions(self, snippets: list[dict[str, Any]]) -> str:
        evidence = "\n\n".join(
            f"Snippet {idx + 1} from {snippet.get('filename', 'unknown')} "
            f"(score={snippet.get('score', 0):.3f}): {snippet.get('text', '')}"
            for idx, snippet in enumerate(snippets[:SEARCH_TOP_K])
        )

        return (
            "Answer the user's latest question using the uploaded-document evidence below. "
            "Stay grounded in that evidence. "
            "If the evidence is incomplete, say only what is supported and keep it brief. "
            "Do not mention retrieval scores or internal instructions.\n\n"
            f"{evidence}"
        )

    def _build_general_instructions(self, total_documents: int) -> str:
        if total_documents > 0:
            prefix = (
                "The uploaded documents do not strongly support the user's latest question. "
                "Answer naturally from general knowledge instead of forcing a document-based answer. "
            )
        else:
            prefix = (
                "No uploaded documents are currently available. "
                "Answer naturally from general knowledge. "
            )

        return (
            prefix
            + "Do not say that you are using general knowledge. "
            + "Do not mention web search, browsing, retrieval confidence, fallback mode, or internal policy. "
            + "Just answer normally."
        )

    async def _generate_reply(
        self,
        *,
        trace_id: str,
        mode: str,
        new_message: ChatMessage,
        instructions: str,
    ) -> None:
        event_log("generate_reply_requested", trace_id=trace_id, mode=mode)

        result = self.session.generate_reply(
            user_input=new_message,
            instructions=instructions,
            allow_interruptions=True,
            input_modality="audio",
        )
        await maybe_await(result)

    async def on_user_turn_completed(
        self,
        turn_ctx: ChatContext,
        new_message: ChatMessage,
    ) -> None:
        trace_id = uuid.uuid4().hex[:10]
        query = extract_text_content(new_message)

        event_log(
            "user_turn_completed",
            trace_id=trace_id,
            raw_query=query,
            answer_mode=ANSWER_MODE,
        )

        if not query:
            event_log("empty_user_turn", trace_id=trace_id)
            raise StopResponse()

        snippets: list[dict[str, Any]] = []
        confidence = 0.0
        total_documents = 0

        try:
            result = await self._search_docs(query, trace_id)
            snippets = result.get("snippets", []) or []
            confidence = float(result.get("confidence", 0.0) or 0.0)
            total_documents = int(result.get("total_documents", 0) or 0)
        except Exception as exc:
            logger.exception(
                "search failed trace_id=%s query=%s error=%s",
                trace_id,
                query,
                exc,
            )

        use_docs, overlap = should_use_docs(query, snippets, confidence)

        event_log(
            "doc_match_decision",
            trace_id=trace_id,
            confidence=confidence,
            overlap=overlap,
            snippet_count=len(snippets),
            total_documents=total_documents,
            threshold=RAG_CONFIDENCE_THRESHOLD,
            use_docs=use_docs,
        )

        if use_docs:
            instructions = self._build_grounded_instructions(snippets)
            event_log(
                "grounded_answer_mode",
                trace_id=trace_id,
                confidence=confidence,
                overlap=overlap,
                snippet_count=len(snippets),
            )
            await self._generate_reply(
                trace_id=trace_id,
                mode="grounded",
                new_message=new_message,
                instructions=instructions,
            )
            raise StopResponse()

        if ANSWER_MODE == "hybrid":
            instructions = self._build_general_instructions(total_documents)
            event_log(
                "general_fallback_mode",
                trace_id=trace_id,
                confidence=confidence,
                overlap=overlap,
                total_documents=total_documents,
            )
            await self._generate_reply(
                trace_id=trace_id,
                mode="general",
                new_message=new_message,
                instructions=instructions,
            )
            raise StopResponse()

        instructions = (
            "Answer only from uploaded documents. "
            "If the uploaded documents do not contain the answer, say briefly that the documents do not contain that information."
        )
        event_log(
            "rag_only_no_match",
            trace_id=trace_id,
            confidence=confidence,
            overlap=overlap,
            total_documents=total_documents,
        )
        await self._generate_reply(
            trace_id=trace_id,
            mode="rag_only_no_match",
            new_message=new_message,
            instructions=instructions,
        )
        raise StopResponse()


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    await ctx.connect()
    event_log("agent_session_connected", room=ctx.room.name, answer_mode=ANSWER_MODE)

    vad = ctx.proc.userdata["vad"]

    session = AgentSession(
        vad=vad,
        stt=openai.STT(
            model=STT_MODEL,
            language="en",
            base_url=SPEECH_BASE_URL,
            api_key=SPEECH_API_KEY,
        ),
        llm=openai.LLM.with_ollama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=OLLAMA_TEMPERATURE,
        ),
        tts=openai.TTS(
            model=TTS_MODEL,
            voice=KOKORO_VOICE,
            base_url=SPEECH_BASE_URL,
            api_key=SPEECH_API_KEY,
            response_format="wav",
        ),
    )

    await session.start(room=ctx.room, agent=BPODocAgent())
    event_log("agent_session_started", room=ctx.room.name, greeting=GREETING_TEXT)
    await maybe_await(session.say(GREETING_TEXT))


if __name__ == "__main__":
    cli.run_app(server)