"""HLE LLM-judge grader (the official HLE judge prompt without its confidence field, gpt-5.4-nano).

The judge needs the question too, so HLE is graded in a separate pass rather than through
``Benchmark.verify``. ``build_hle_judge_request`` and ``parse_hle_judge_response`` are shared by
the synchronous and Batch paths; ``grade_hle_response`` makes one synchronous call.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

# --------------------------------------------------------------------------------------
# Judge configuration (as used for the paper)
# --------------------------------------------------------------------------------------
HLE_JUDGE_MODEL = "gpt-5.4-nano"
HLE_JUDGE_REASONING_EFFORT = "low"
# The judge emits one short JSON object, so this cap should never bind; it is
# kept at the paper's value so truncation behaviour -- and therefore the rate
# of unparseable verdicts -- matches the paper's grading exactly.
HLE_JUDGE_MAX_COMPLETION_TOKENS = 4096


def hle_grader_id(
    judge_model: str = HLE_JUDGE_MODEL,
    reasoning_effort: str = HLE_JUDGE_REASONING_EFFORT,
) -> str:
    """Grader id: ``<model>__hle_official_style_llm_judge__<effort>``."""
    raw = f"{judge_model}__hle_official_style_llm_judge__{reasoning_effort}"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", raw)


# --------------------------------------------------------------------------------------
# Response truncation
#
# The judge only needs the model's final answer, which sits at the END of the response,
# so we keep only the last `max_chars` chars. The cut is snapped FORWARD to the next
# sentence/line boundary so the kept text does not start mid-sentence; because the
# answer is at the very end, snapping forward never risks dropping it.
# --------------------------------------------------------------------------------------
DEFAULT_JUDGE_MAX_CHARS = 1500
OMITTED_MARKER = "…[earlier reasoning omitted]…"

# Terminal punctuation (+ optional close quote/bracket) then whitespace, OR a newline.
# `match.end()` lands at the start of the next sentence/line.
_BOUNDARY = re.compile(r"""[.!?]["')\]]?\s+|\n""")


def split_for_preview(
    text: Optional[str],
    *,
    max_chars: int = DEFAULT_JUDGE_MAX_CHARS,
    snap_to_boundary: bool = True,
) -> tuple[str, str]:
    """Return (removed_prefix, kept_suffix): roughly the last `max_chars` chars kept."""
    if not text:
        return "", text or ""
    if len(text) <= max_chars:
        return "", text
    cut_index = len(text) - max_chars
    if snap_to_boundary:
        match = _BOUNDARY.search(text, cut_index)
        # Only snap forward if it doesn't discard the entire kept window.
        if match is not None and match.end() < len(text):
            cut_index = match.end()
    return text[:cut_index], text[cut_index:]


def truncate_response_for_judge(
    text: Optional[str],
    *,
    max_chars: int = DEFAULT_JUDGE_MAX_CHARS,
    snap_to_boundary: bool = True,
    marker: str = OMITTED_MARKER,
) -> str:
    """The response text to send to the judge: last `max_chars` chars, marker-prefixed."""
    removed, kept = split_for_preview(
        text, max_chars=max_chars, snap_to_boundary=snap_to_boundary
    )
    if not removed:
        return kept
    return f"{marker}\n{kept}"


# --------------------------------------------------------------------------------------
# Judge prompt + structured-output schema
# --------------------------------------------------------------------------------------
HLE_JUDGE_PROMPT = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.
"""

EXTRACTED_ANSWER_RESPONSE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "ExtractedAnswer",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "extracted_final_answer": {"type": "string"},
                "reasoning": {"type": "string"},
                "correct": {"type": "string", "enum": ["yes", "no"]},
                "strict": {"type": "boolean", "enum": [True]},
            },
            "required": [
                "extracted_final_answer",
                "reasoning",
                "correct",
                "strict",
            ],
        },
    },
}


def build_hle_judge_prompt(
    *,
    question: str,
    correct_answer: str,
    response: str,
    max_response_chars: Optional[int] = DEFAULT_JUDGE_MAX_CHARS,
) -> str:
    """Fill the judge prompt; response truncated to its last `max_response_chars` chars."""
    if max_response_chars is not None:
        response = truncate_response_for_judge(response, max_chars=max_response_chars)
    return HLE_JUDGE_PROMPT.format(
        question=question,
        correct_answer=correct_answer,
        response=response,
    )


def build_hle_judge_request(
    *,
    question: str,
    response: str,
    correct_answer: str,
    judge_model: str = HLE_JUDGE_MODEL,
    reasoning_effort: str = HLE_JUDGE_REASONING_EFFORT,
    max_completion_tokens: int = HLE_JUDGE_MAX_COMPLETION_TOKENS,
    max_response_chars: Optional[int] = DEFAULT_JUDGE_MAX_CHARS,
) -> dict[str, Any]:
    """Build the chat-completions request body (identical for sync and Batch API).

    A Batch API line is
    ``{"custom_id": ..., "method": "POST", "url": "/v1/chat/completions",
    "body": build_hle_judge_request(...)}``.
    """
    prompt = build_hle_judge_prompt(
        question=question,
        correct_answer=correct_answer,
        response=response,
        max_response_chars=max_response_chars,
    )
    return {
        "model": judge_model,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": EXTRACTED_ANSWER_RESPONSE_FORMAT,
        "reasoning_effort": reasoning_effort,
        "max_completion_tokens": max_completion_tokens,
    }


# --------------------------------------------------------------------------------------
# Grade type + parsing
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class HLEGrade:
    """One judge verdict.

    ``is_correct``/``extracted_answer`` are ``None`` when the judge errored or its
    output could not be parsed (``error`` says why); callers decide the policy for
    ungraded rollouts (the released tables count them as incorrect).
    """

    is_correct: Optional[bool]
    extracted_answer: Optional[str]
    reasoning: Optional[str] = None
    grader_id: str = field(default_factory=hle_grader_id)
    error: Optional[str] = None
    raw: Optional[dict[str, Any]] = None

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "grader_type": "hle_official_style_llm_judge",
            "grader_id": self.grader_id,
            "reasoning": self.reasoning,
            "error": self.error,
        }


def _parse_judge_content(content: Any) -> Optional[dict[str, Any]]:
    """Parse the judge message content into a dict."""
    if isinstance(content, dict):
        return content
    if not isinstance(content, str):
        return None
    text = content.strip()
    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```").strip()
        if text.endswith("```"):
            text = text[:-3].strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def parse_hle_judge_response(
    content: Any,
    *,
    grader_id: Optional[str] = None,
) -> HLEGrade:
    """Turn a judge message's content (str or dict) into an ``HLEGrade``."""
    gid = grader_id or hle_grader_id()
    parsed = _parse_judge_content(content)
    if parsed is None:
        return HLEGrade(
            is_correct=None,
            extracted_answer=None,
            grader_id=gid,
            error="unparseable judge response",
        )
    return HLEGrade(
        is_correct=parsed.get("correct") == "yes",
        extracted_answer=parsed.get("extracted_final_answer"),
        reasoning=parsed.get("reasoning"),
        grader_id=gid,
        raw=parsed,
    )


# --------------------------------------------------------------------------------------
# One-shot synchronous grade (the concurrent pass fans this out over a thread pool)
# --------------------------------------------------------------------------------------
def grade_hle_response(
    question: str,
    response_text: str,
    gold_answer: str,
    *,
    client: Any = None,
    judge_model: str = HLE_JUDGE_MODEL,
    reasoning_effort: str = HLE_JUDGE_REASONING_EFFORT,
    max_completion_tokens: int = HLE_JUDGE_MAX_COMPLETION_TOKENS,
    max_response_chars: Optional[int] = DEFAULT_JUDGE_MAX_CHARS,
) -> HLEGrade:
    """Grade a single response with one judge call.

    ``client`` is an ``openai.OpenAI`` instance (thread-safe — share one across the
    concurrent pass); a fresh one is created from the environment if omitted.
    Transient/API errors are returned as an ungraded ``HLEGrade`` (``error`` set),
    never raised, so a single bad row can't sink a whole grading pass.
    """
    if client is None:
        from openai import OpenAI

        client = OpenAI()

    body = build_hle_judge_request(
        question=question,
        response=response_text,
        correct_answer=gold_answer,
        judge_model=judge_model,
        reasoning_effort=reasoning_effort,
        max_completion_tokens=max_completion_tokens,
        max_response_chars=max_response_chars,
    )
    gid = hle_grader_id(judge_model, reasoning_effort)
    try:
        resp = client.chat.completions.create(**body)
    except Exception as exc:  # noqa: BLE001 - surface as ungraded, don't crash the pass
        return HLEGrade(
            is_correct=None,
            extracted_answer=None,
            grader_id=gid,
            error=f"{type(exc).__name__}: {exc}",
        )

    content = resp.choices[0].message.content
    return parse_hle_judge_response(content, grader_id=gid)
