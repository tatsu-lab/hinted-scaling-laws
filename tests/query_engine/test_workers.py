from __future__ import annotations

from hinted.models import MODELS
from hinted.query_engine.types import InferenceConfig
from hinted.query_engine.workers import VLLMInferenceWorker

# The budget helper lives on the worker but touches no engine, so it is
# testable anywhere. Per request:
#     min(requested_max_tokens, context_limit - prompt_tokens - margin)
# with margin 256. Budgets below config.min_generation_tokens become error
# results in the caller instead of generating, so there is no floor here.


def _worker() -> VLLMInferenceWorker:
    return VLLMInferenceWorker("http://localhost:1")


def test_requested_budget_binds_for_short_prompts() -> None:
    config = InferenceConfig(model=MODELS["Qwen3-0.6B"])  # 32000 / 32768
    assert _worker()._request_max_tokens(200, config) == 32000


def test_context_binds_for_long_prompts() -> None:
    config = InferenceConfig(model=MODELS["Qwen3-0.6B"])
    # 32768 - 4000 - 256 = 28512 < 32000
    assert _worker()._request_max_tokens(4000, config) == 28512


def test_overfull_prompt_gets_no_budget() -> None:
    config = InferenceConfig(model=MODELS["Llama-2-7b-chat-hf"])  # 4096 / 4096
    assert _worker()._request_max_tokens(4095, config) < 1


def test_llama2_hint_sized_prompt() -> None:
    config = InferenceConfig(model=MODELS["Llama-2-7b-chat-hf"])
    # 4096 - 2500 - 256 = 1340 tokens of reasoning room
    assert _worker()._request_max_tokens(2500, config) == 1340
