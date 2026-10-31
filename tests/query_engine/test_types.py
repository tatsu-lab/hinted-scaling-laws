from __future__ import annotations

import pytest

from hinted.models import MODELS
from hinted.query_engine.types import InferenceConfig, InferenceTask, QueryResult


def test_task_json_round_trip() -> None:
    task = InferenceTask(
        task_key="p1|s0|h0.4",
        prompt="student prompt",
        teacher_prompt="teacher prompt",
        hint_level=0.4,
    )
    assert InferenceTask.from_json(task.to_json()) == task


def test_task_defaults_to_hint_free() -> None:
    task = InferenceTask(task_key="p1|s0|h0", prompt="just a question")
    restored = InferenceTask.from_json(task.to_json())
    assert restored.teacher_prompt is None
    assert restored.hint_level == 0.0


def test_task_rejects_negative_hint_level() -> None:
    with pytest.raises(ValueError, match="hint_level"):
        InferenceTask(task_key="bad", prompt="q", hint_level=-0.1)


def test_query_result_round_trip() -> None:
    result = QueryResult(
        task_key="p1|s0|h0.4",
        response_text="\\boxed{70}",
        finish_reason="stop",
        is_error=False,
        input_token_count=100,
        output_token_count=50,
    )
    assert QueryResult.from_dict(result.to_dict()) == result


def test_inference_config_validates_gpu_count() -> None:
    model = MODELS["Qwen3-0.6B"]
    with pytest.raises(ValueError, match="joint_gpus_per_worker"):
        InferenceConfig(model=model, joint_gpus_per_worker=3)


def test_inference_config_validates_top_k_payload() -> None:
    model = MODELS["gemma-3-1b-it"]  # sampling top_k = 64
    with pytest.raises(ValueError, match="top_k_payload"):
        InferenceConfig(model=model, teacher_model=model, joint_top_k_payload=32)


def test_inference_config_validates_teacher_scale() -> None:
    model = MODELS["Qwen3-0.6B"]
    with pytest.raises(ValueError, match="joint_teacher_scale"):
        InferenceConfig(model=model, joint_teacher_scale="gap")
