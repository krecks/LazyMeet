"""Summarize a speaker-labelled transcript with an in-process GGUF LLM.

Uses ``llama-cpp-python`` so there is no external Ollama service. Long meetings
are chunked to fit the model's context window (map-reduce: summarize each chunk
to notes, then summarize the notes), so transcripts of any length work. The
final answer is strict JSON ``{summary, takeaways[], action_items[]}`` parsed
defensively (LLMs occasionally wrap JSON in prose or code fences).
"""

from __future__ import annotations

import json
import re

from .. import config

_llm = None

_SYSTEM = (
    "You are a meeting assistant. Given a transcript with speaker labels, produce "
    "a concise summary, the key takeaways, and concrete action items. "
    "Respond with ONLY a JSON object of the form "
    '{"summary": str, "takeaways": [str], "action_items": [str]}.'
)

_NOTES_SYSTEM = (
    "You are a meeting assistant. Summarize this portion of a meeting transcript "
    "into concise bullet notes capturing decisions, topics, and any action items. "
    "Return plain text bullets only."
)

# Tokens reserved for the model's response plus prompt scaffolding.
_RESERVE = 2560


def _load_llm():
    global _llm
    if _llm is not None:
        return _llm

    from llama_cpp import Llama

    _llm = Llama(
        model_path=str(config.LLM_MODEL),
        n_ctx=config.LLM_N_CTX,
        n_threads=config.NUM_THREADS,
        verbose=False,
    )
    return _llm


def _transcript_lines(segments: list[dict], speakers: dict[str, str]) -> list[str]:
    lines = []
    for seg in segments:
        name = speakers.get(seg["speaker"], seg["speaker"])
        lines.append(f"{name}: {seg['text']}")
    return lines


def _count_tokens(llm, text: str) -> int:
    return len(llm.tokenize(text.encode("utf-8"), add_bos=False, special=False))


def _chunk_lines(llm, lines: list[str], budget: int) -> list[str]:
    chunks: list[str] = []
    cur: list[str] = []
    cur_tok = 0
    for line in lines:
        t = _count_tokens(llm, line) + 1
        if cur and cur_tok + t > budget:
            chunks.append("\n".join(cur))
            cur, cur_tok = [], 0
        cur.append(line)
        cur_tok += t
    if cur:
        chunks.append("\n".join(cur))
    return chunks


def _truncate_to_tokens(llm, text: str, budget: int) -> str:
    toks = llm.tokenize(text.encode("utf-8"), add_bos=False, special=False)
    if len(toks) <= budget:
        return text
    toks = toks[:budget]
    return llm.detokenize(toks).decode("utf-8", errors="ignore")


def summarize(segments: list[dict], speakers: dict[str, str]) -> dict:
    if config.FAKE_MODE:
        return _fake_summary()

    llm = _load_llm()
    lines = _transcript_lines(segments, speakers)
    budget = max(1024, config.LLM_N_CTX - _RESERVE)
    full = "\n".join(lines)

    if _count_tokens(llm, full) <= budget:
        return _final_summary(llm, full)

    # Map-reduce for long meetings.
    notes = []
    for chunk in _chunk_lines(llm, lines, budget):
        notes.append(_chunk_notes(llm, chunk))
    combined = _truncate_to_tokens(llm, "\n".join(notes), budget)
    return _final_summary(llm, combined, from_notes=True)


def _chunk_notes(llm, chunk: str) -> str:
    resp = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": _NOTES_SYSTEM},
            {"role": "user", "content": chunk},
        ],
        temperature=0.2,
        max_tokens=512,
    )
    return resp["choices"][0]["message"]["content"].strip()


def _final_summary(llm, body: str, from_notes: bool = False) -> dict:
    label = "Meeting notes" if from_notes else "Transcript"
    resp = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": f"{label}:\n{body}"},
        ],
        temperature=0.2,
        max_tokens=1024,
    )
    return _parse(resp["choices"][0]["message"]["content"])


def _parse(content: str) -> dict:
    data = None
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                data = None
    if not isinstance(data, dict):
        return {"summary": content.strip(), "takeaways": [], "action_items": []}
    return {
        "summary": str(data.get("summary", "")).strip(),
        "takeaways": [str(x) for x in data.get("takeaways", []) if str(x).strip()],
        "action_items": [str(x) for x in data.get("action_items", []) if str(x).strip()],
    }


def _fake_summary() -> dict:
    return {
        "summary": (
            "Weekly sync covering the transcription pipeline. The diarization step is "
            "complete and the speaker library is the remaining item before Friday."
        ),
        "takeaways": [
            "Diarization step is implemented.",
            "Speaker library is the next deliverable.",
        ],
        "action_items": [
            "Add the speaker library before Friday.",
            "Send the demo recording to the team.",
        ],
    }
