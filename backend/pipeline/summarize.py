"""Summarize a speaker-labelled transcript with an in-process GGUF LLM.

Uses ``llama-cpp-python`` so there is no external Ollama service. The model is
asked for strict JSON ``{summary, takeaways[], action_items[]}`` and parsed
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


def _load_llm():
    global _llm
    if _llm is not None:
        return _llm

    from llama_cpp import Llama

    _llm = Llama(
        model_path=str(config.LLM_MODEL),
        n_ctx=8192,
        n_threads=config.NUM_THREADS,
        verbose=False,
    )
    return _llm


def _transcript_text(segments: list[dict], speakers: dict[str, str]) -> str:
    lines = []
    for seg in segments:
        name = speakers.get(seg["speaker"], seg["speaker"])
        lines.append(f"{name}: {seg['text']}")
    return "\n".join(lines)


def summarize(segments: list[dict], speakers: dict[str, str]) -> dict:
    if config.FAKE_MODE:
        return _fake_summary()

    transcript = _transcript_text(segments, speakers)
    llm = _load_llm()
    resp = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": f"Transcript:\n{transcript}"},
        ],
        temperature=0.2,
        max_tokens=1024,
    )
    content = resp["choices"][0]["message"]["content"]
    return _parse(content)


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
