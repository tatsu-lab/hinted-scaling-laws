"""Task, result, and configuration types for the query engine."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Optional

from hinted.models import ModelConfig


@dataclass(frozen=True)
class InferenceTask:
    """One unit of work: generate a completion for one prompt.

    For plain inference only `prompt` is used. For joint decoding, `prompt`
    is the student prompt (side A, whose output text we keep), and
    `teacher_prompt` is the full-hint prompt (side B); `hint_level` is the
    logit-mixing weight on the teacher (advisor_weight in joint_decode).
    """

    task_key: str  # caller-chosen identifier, e.g. "aime2025_2026_0001|s3|h0.4"
    prompt: str
    teacher_prompt: Optional[str] = None
    hint_level: float = 0.0

    def __post_init__(self) -> None:
        # Not capped at 1: under the unnormalized-ADD selector this is an
        # advisor bias strength, not a mixing weight, and the grids run well
        # past 1 (e.g. 1.6 ... 10).
        if self.hint_level < 0.0:
            raise ValueError(f"hint_level must be >= 0, got {self.hint_level}")

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @staticmethod
    def from_json(payload: str) -> "InferenceTask":
        return InferenceTask(**json.loads(payload))


@dataclass
class QueryResult:
    """Result of one inference task."""

    task_key: str
    response_text: str
    finish_reason: str
    is_error: bool
    error: Optional[str] = None
    input_token_count: Optional[int] = None
    output_token_count: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> "QueryResult":
        return QueryResult(**payload)


@dataclass(frozen=True)
class InferenceConfig:
    """Everything a worker needs beyond the model registry entry.

    The registry ModelConfig carries the per-model sampling recommendations;
    this adds engine- and run-level settings.
    """

    model: ModelConfig
    dtype: str = "bfloat16"
    # Generation length is derived from the model registry entry: engines run
    # at model.context_limit, and each batch's max_tokens is
    # min(model.requested_max_tokens,
    #     model.context_limit - longest_prompt_in_batch - token_safety_margin).
    # The margin absorbs tokenizer count discrepancies (BOS/template fuzz).
    token_safety_margin: int = 256
    # A request whose budget (context_limit - prompt - margin) is below this
    # is not generated: the worker returns an is_error result with
    # finish_reason "prompt_too_long" (the runner records it, analysis drops
    # it). Guards against few-shot + hint prompts that fill a small context.
    min_generation_tokens: int = 32
    # Apply the model's chat template to prompts (single user turn). Both
    # worker kinds apply it identically so h=0 and h>0 runs are comparable.
    # False for raw-completion (base-model, few-shot) benchmarks.
    apply_chat_template: bool = True
    # Stop strings for the plain vLLM worker: generation halts when any of
    # these appears (the strings are excluded from the output). Used by
    # raw-completion benchmarks to cut the model off when it starts a new
    # few-shot block, e.g. ("Question:", "</s>", "<|im_end|>") for GSM8K.
    # Joint decode carries its own stop config and ignores this.
    stop: tuple[str, ...] = ()
    # How many tasks a worker pulls (and generates) at a time.
    gen_batch_size: int = 64
    # Tensor parallelism for the plain vLLM worker: shards the engine over
    # this many GPUs (the worker's slurm job requests the same number).
    # For models whose weights don't fit one card (>=32B on 80G, 70B tier).
    # Not supported for joint decoding.
    tensor_parallel_size: int = 1
    # --- Joint decode only ---
    # Model for side B (the teacher). None = the student's own model advises
    # itself. The paper's logit mixing sets it to a FIXED teacher: side B
    # loads this model and its context limit instead. The two vocabularies
    # must agree, since the selector mixes logits by token id and feeds one
    # token back to both sides (Delphi and Llama-3.x share the 128,256-token
    # Llama-3 vocab; each post-trained family shares one within itself).
    teacher_model: Optional[ModelConfig] = None
    # The two sides' logits are mixed as logit_A + h * logit_B (decoder at
    # full weight, h the teacher weight, unbounded). "std" first rescales
    # side B to side A's spread at each step:
    #   logit_A + h * (sigma_A / sigma_B) * logit_B
    joint_teacher_scale: str = "none"
    # 1: both engines share one card (~0.45 memory each); 2: one card each.
    # WARNING: 1-GPU mode is currently unreliable at fleet scale — the two
    # engines boot concurrently and vLLM's memory profiler counts the
    # sibling's mid-boot allocations against its own budget (random
    # "insufficient KV cache" boot failures). Use 2 unless the vendored
    # decoder learns to boot engines sequentially.
    joint_gpus_per_worker: int = 2
    # Size of the top-k logit payload each side reports to the coordinator.
    # Mixing happens over the union of the two payloads, so this bounds the
    # candidate set; keep comfortably above the model's sampling top_k.
    joint_top_k_payload: int = 128
    joint_barrier_timeout_s: float = 600.0
    joint_enable_prefix_caching: bool = True
    # Cap on tokens per engine step. The vendored default derives ~8x
    # max_model_len (~262k at 32768), which makes vLLM's memory profiler
    # reserve huge activation buffers and starve the KV cache, especially
    # in the 0.45-memory 1-GPU mode. Bounds prefill chunking only; decode
    # throughput is unaffected.
    joint_max_num_batched_tokens: int = 8192

    def __post_init__(self) -> None:
        if self.joint_gpus_per_worker not in (1, 2):
            raise ValueError("joint_gpus_per_worker must be 1 or 2")
        if not 1 <= self.tensor_parallel_size <= 8:
            raise ValueError("tensor_parallel_size must be in [1, 8]")
        if self.joint_teacher_scale not in ("none", "std"):
            raise ValueError("joint_teacher_scale must be 'none' or 'std', got "
                             f"{self.joint_teacher_scale!r}")
        if self.teacher_model is not None and self.joint_top_k_payload < self.model.top_k:
            raise ValueError(
                "joint_top_k_payload should be >= the model's sampling top_k "
                f"({self.joint_top_k_payload} < {self.model.top_k})"
            )


@dataclass
class ResourcesConfig:
    """Slurm resources for a fleet of workers (one worker = one slurm job)."""

    log_dir: str
    # Site-specific settings default to None, which leaves them to slurm's
    # own defaults; set them for your cluster.
    # Comma-separated list submits to multiple partitions at once.
    partition: Optional[str] = None
    # Slurm node features (--constraint), e.g. an OR of GPU types to keep
    # workers on cards large enough for the model. vLLM needs Ampere or newer.
    constraints: Optional[str] = None
    account: Optional[str] = None
    mem: str = "64G"
    time: str = "12:00:00"
    cpus_per_task: int = 8
    num_jobs: int = 4
    exclude: str = ""
    node_list: Optional[str] = None
    exclusive: bool = False
