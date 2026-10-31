"""Inference workers: a pull/process/push loop around one vLLM engine (VLLMInferenceWorker) or
the joint-decode pair for logit mixing (JointDecodeInferenceWorker).

Engine imports are deferred to setup(), so this module imports without a GPU or vLLM.
"""

from __future__ import annotations

import os
import random
import sys
import time
import traceback
from abc import abstractmethod
from typing import Any, List, Optional

from hinted.query_engine.server import (
    SubmititWorker,
    SubmitResult,
    SubmitResults,
    TaskItem,
)
from hinted.query_engine.types import InferenceConfig, InferenceTask, QueryResult


class InferenceWorker(SubmititWorker):
    """Generic worker loop; subclasses provide engine setup and batch inference."""

    @abstractmethod
    def setup(self, config: InferenceConfig) -> None:
        """Boot the inference engine(s). Called once, on the GPU node."""

    @abstractmethod
    def process_batch(
        self, tasks: List[InferenceTask], config: InferenceConfig
    ) -> List[QueryResult]:
        """Generate completions for a batch; must return one result per task."""

    def teardown(self) -> None:
        """Release engine resources. Called once on exit (best-effort)."""

    def _apply_chat_template(self, prompt: str) -> str:
        return self._tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def _load_tokenizer(self, config: InferenceConfig) -> None:
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(config.model.hf_path)

    def _count_tokens(self, prompt: str) -> int:
        return len(self._tokenizer(prompt, add_special_tokens=False)["input_ids"])

    def _request_max_tokens(self, prompt_tokens: int, config: InferenceConfig) -> int:
        """Generation budget for ONE request:
        min(requested_max_tokens, context_limit - prompt_tokens - margin).

        Per request rather than per batch: a single long prompt used to set the
        batch's shared max_tokens and silently truncate every other completion
        in it (and drive it to 0 -- a vLLM error that killed the worker -- when
        the prompt filled the context). The caller turns budgets below
        config.min_generation_tokens into error results instead of generating.
        """
        margin = getattr(config, "token_safety_margin", 256)
        return min(
            config.model.requested_max_tokens,
            config.model.context_limit - prompt_tokens - margin,
        )

    def __call__(self, config: InferenceConfig) -> None:
        # Slurm workers inherit the submission env, which invokes
        # .venv/bin/python directly without the venv's bin dir on PATH — but
        # vLLM shells out to venv-installed binaries (e.g. ninja for
        # torch.compile), and joint-decode worker subprocesses inherit this
        # env too. Put our interpreter's bin dir first.
        os.environ["PATH"] = (
            os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", "")
        )
        # Workers never talk to the HuggingFace Hub: the driver pre-downloads
        # weights to the shared cache, and a large fleet booting at once
        # would 429 the Hub API and death-spiral (each 429 kills a worker,
        # its replacement 429s again).
        os.environ["HF_HUB_OFFLINE"] = "1"

        exit_reason = "unknown"
        status_check_interval = 2.0
        last_status_check = 0.0

        try:
            self.setup(config)

            while True:
                # Jitter so a fleet of workers doesn't hit the server in lockstep.
                time.sleep(random.uniform(0.2, 0.8))

                now = time.time()
                if now - last_status_check > status_check_interval:
                    last_status_check = now
                    if self.get_done():
                        break

                tasks: List[TaskItem] = self.get_tasks(
                    num_tasks=config.gen_batch_size
                ).tasks
                if not tasks:
                    time.sleep(1)
                    continue

                parsed = [InferenceTask.from_json(t.task) for t in tasks]
                results = self.process_batch(parsed, config)
                assert len(results) == len(tasks), "one result required per task"

                self.submit_results(
                    SubmitResults(
                        results=[
                            SubmitResult(
                                task_id=task.task_id,
                                worker_id=self.worker_id,
                                result=result.to_dict(),
                            )
                            for task, result in zip(tasks, results)
                        ]
                    )
                )

            exit_reason = "finished"

        except Exception as e:
            print(f"Error in worker {self.worker_id}: {e}")
            print(traceback.format_exc())
            exit_reason = f"error: {e}"
            raise

        finally:
            try:
                self.teardown()
            except Exception as e:
                print(f"Error in worker teardown: {e}")
            self.report_dead(reason=exit_reason)
            if exit_reason != "finished":
                sys.exit(1)


class VLLMInferenceWorker(InferenceWorker):
    """Single-engine vLLM worker for plain (hint-free) generation."""

    def setup(self, config: InferenceConfig) -> None:
        from vllm import LLM

        self._load_tokenizer(config)
        self._llm = LLM(
            model=config.model.hf_path,
            dtype=config.dtype,
            max_model_len=config.model.context_limit,
            tensor_parallel_size=config.tensor_parallel_size,
            seed=random.randint(0, 2**31 - 1),
        )

    def process_batch(
        self, tasks: List[InferenceTask], config: InferenceConfig
    ) -> List[QueryResult]:
        from vllm import SamplingParams

        prompts = [
            self._apply_chat_template(t.prompt) if config.apply_chat_template else t.prompt
            for t in tasks
        ]
        counts = [self._count_tokens(p) for p in prompts]
        budgets = [self._request_max_tokens(n, config) for n in counts]

        # Prompts that leave too little room to answer become error rows, so a
        # 2,000-token few-shot prompt neither kills the worker nor truncates
        # its batch-mates. Everything else gets its own max_tokens.
        # getattr: configs pickled by drivers started before these fields
        # existed are unpickled into the new class without them
        min_gen = getattr(config, "min_generation_tokens", 32)
        margin = getattr(config, "token_safety_margin", 256)
        results: List[Optional[QueryResult]] = [None] * len(tasks)
        keep: List[int] = []
        for i, (task, n, budget) in enumerate(zip(tasks, counts, budgets)):
            if budget < min_gen:
                results[i] = QueryResult(
                    task_key=task.task_key,
                    response_text="",
                    finish_reason="prompt_too_long",
                    is_error=True,
                    error=(f"prompt is {n} tokens; context {config.model.context_limit} "
                           f"minus margin {margin} leaves {budget} "
                           f"< min_generation_tokens {min_gen}"),
                    input_token_count=n,
                    output_token_count=0,
                )
            else:
                keep.append(i)

        if keep:
            sampling_params = [
                SamplingParams(
                    temperature=config.model.temperature,
                    top_p=config.model.top_p,
                    top_k=config.model.top_k,
                    repetition_penalty=config.model.repetition_penalty,
                    max_tokens=budgets[i],
                    stop=list(config.stop) or None,
                )
                for i in keep
            ]
            outputs = self._llm.generate(
                [prompts[i] for i in keep], sampling_params=sampling_params
            )
            for i, output in zip(keep, outputs):
                completion = output.outputs[0]
                results[i] = QueryResult(
                    task_key=tasks[i].task_key,
                    response_text=completion.text,
                    finish_reason=str(completion.finish_reason),
                    is_error=False,
                    input_token_count=len(output.prompt_token_ids),
                    output_token_count=len(completion.token_ids),
                )
        return [r for r in results if r is not None]


class JointDecodeInferenceWorker(InferenceWorker):
    """Two engines — the student and a teacher — mixed per task hint_level.

    Both sides load the same model unless config.teacher_model names a
    different one (a fixed external teacher).

    Side A is the student (question-only prompt; its text is the output),
    side B is the teacher (full-hint prompt). The selector closure reads the
    per-request hint level from _batch_hint_levels, so one batch may mix
    different hint levels without engine reboots.
    """

    def setup(self, config: InferenceConfig) -> None:
        if config.tensor_parallel_size != 1:
            raise NotImplementedError(
                "tensor parallelism is not plumbed through joint decoding; "
                "its engines are spawned by the vendored worker CLI"
            )
        from hinted.joint_decode.config import JointDecodeSamplingConfig
        from hinted.joint_decode.gpu.config import (
            JointDecodeConfig,
            JointDecodeModelConfig,
        )
        from hinted.joint_decode.gpu.decoder import joint_decoder
        from hinted.joint_decode.selection import select_add_logits

        self._load_tokenizer(config)

        # joint_decode pins each engine via CUDA_VISIBLE_DEVICES=<gpu_index>,
        # which is a PHYSICAL device id. Slurm may have allocated us any
        # GPU(s) on a shared node (exported in CUDA_VISIBLE_DEVICES), so map
        # through that allocation — hardcoding 0/1 would stack every worker
        # on the node onto physical GPU 0 and exhaust its memory.
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        devices = [int(d) for d in visible.split(",") if d.strip()] or [0, 1]
        if config.joint_gpus_per_worker == 1 or len(devices) == 1:
            gpu_indices = (devices[0], devices[0])
            memory_utilization: tuple[Optional[float], Optional[float]] = (0.45, 0.45)
        else:
            gpu_indices = (devices[0], devices[1])
            memory_utilization = (None, None)

        # side 0 = student (its text is the output), side 1 = teacher. With
        # config.teacher_model set, side 1 is a different model entirely; its
        # own context limit applies, since a teacher prompt carrying the full
        # hint is longer than the student's and the teacher may have more room.
        teacher = getattr(config, "teacher_model", None) or config.model

        def model_config(side: int) -> JointDecodeModelConfig:
            m = config.model if side == 0 else teacher
            return JointDecodeModelConfig(
                model_path=m.hf_path,
                gpu_index=gpu_indices[side],
                max_model_len=m.context_limit,
                gpu_memory_utilization=memory_utilization[side],
                enable_prefix_caching=config.joint_enable_prefix_caching,
                enforce_eager=False,
            )

        # max_tokens is fixed at engine spawn, so it can't be clamped per
        # batch here. requested_max_tokens is safe regardless: vLLM's
        # scheduler length-caps every request at prompt + output =
        # max_model_len (FINISHED_LENGTH_CAPPED), i.e. an exact per-request
        # clamp. A prompt too long for the context makes vLLM raise, which
        # crashes the worker — deliberate; the crash is visible in the
        # server log (death reason) and the submitit job log.
        # record both sides: worker logs are deleted on success, and "which
        # model actually advised" is the one fact this arm turns on
        print(f"[joint] side A student={config.model.name} "
              f"({config.model.hf_path}) gpu={gpu_indices[0]} | "
              f"side B teacher={teacher.name} ({teacher.hf_path}) "
              f"gpu={gpu_indices[1]}", flush=True)
        joint_config = JointDecodeConfig(
            model_a=model_config(0),
            model_b=model_config(1),
            sampling=JointDecodeSamplingConfig(
                max_tokens_a=config.model.requested_max_tokens,
                max_tokens_b=config.model.requested_max_tokens,
                top_k_a=config.joint_top_k_payload,
                top_k_b=config.joint_top_k_payload,
                barrier_timeout_s=config.joint_barrier_timeout_s,
                seed=random.randint(0, 2**31 - 1),
                stop=tuple(config.stop),
                # vLLM requires >= max_model_len when chunked prefill is
                # disabled (as it is in the vendored engine setup), on both
                # sides: an external teacher may have the longer context.
                max_num_batched_tokens=max(
                    config.joint_max_num_batched_tokens, config.model.context_limit,
                    teacher.context_limit,
                ),
            ),
        )

        self._batch_hint_levels: List[float] = []
        temperature = config.model.temperature
        teacher_scale = config.joint_teacher_scale
        print(f"[joint] logit mixing, teacher_scale={teacher_scale} "
              f"temperature={temperature}", flush=True)

        def select_token(
            a_topk: List[dict[str, Any]],
            b_topk: List[dict[str, Any]],
            *,
            rng: random.Random,
            request_index: int,
        ) -> tuple[List[int], List[int]]:
            return select_add_logits(
                a_topk,
                b_topk,
                advisor_weight=self._batch_hint_levels[request_index],
                temperature=temperature,
                rng=rng,
                request_index=request_index,
                teacher_scale=teacher_scale,
            )

        self._decoder_cm = joint_decoder(joint_config, select_token=select_token)
        self._decoder = self._decoder_cm.__enter__()

    def teardown(self) -> None:
        decoder_cm = getattr(self, "_decoder_cm", None)
        if decoder_cm is not None:
            decoder_cm.__exit__(None, None, None)
            self._decoder_cm = None

    def process_batch(
        self, tasks: List[InferenceTask], config: InferenceConfig
    ) -> List[QueryResult]:
        for task in tasks:
            if task.teacher_prompt is None:
                raise ValueError(
                    f"Task {task.task_key} has no teacher_prompt; joint decoding "
                    "requires one (use VLLMInferenceWorker for hint-free tasks)"
                )

        if config.apply_chat_template:
            prompts_a = [self._apply_chat_template(t.prompt) for t in tasks]
            prompts_b = [self._apply_chat_template(t.teacher_prompt) for t in tasks]
        else:
            prompts_a = [t.prompt for t in tasks]
            prompts_b = [t.teacher_prompt for t in tasks]

        self._batch_hint_levels = [t.hint_level for t in tasks]
        outputs = self._decoder.generate(prompts_a, prompts_b)

        # GenerateOutput carries no token counts; count with the tokenizer so
        # rollout rows and server throughput reporting have them (input count
        # is the student prompt, whose stream the output text comes from).
        return [
            QueryResult(
                task_key=task.task_key,
                response_text=output.text,
                finish_reason=output.finish_reason,
                is_error=False,
                input_token_count=self._count_tokens(prompt_a),
                output_token_count=self._count_tokens(output.text),
            )
            for task, output, prompt_a in zip(tasks, outputs, prompts_a)
        ]
