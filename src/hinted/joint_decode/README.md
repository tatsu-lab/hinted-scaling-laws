# joint-decode

Joint decoding for two vLLM models on GPU, adapted from
[RohithKuditipudi/joint-decode](https://github.com/RohithKuditipudi/joint-decode). In this repo
it backs logit mixing (`JointDecodeInferenceWorker` in `hinted.query_engine.workers`).

At each decode step, worker A and worker B each send top-k logits for the same
active request ids to a parent HTTP coordinator. The coordinator chooses one
side-local token list per request. Each worker then masks its logits so vLLM emits
the chosen local token. Output text is taken from worker A. The protocol allows models with
different tokenizers, but the logit-mixing rule matches tokens by id, so for logit mixing the
two models must share a vocabulary.

The protocol (coordinator, worker run loop, per-step decision logic) lives
in `hinted.joint_decode`; `hinted.joint_decode.gpu` provides engine
construction and the interception point, a vLLM custom logits processor.

This package needs a pinned fork of vLLM, declared in the top-level `pyproject.toml`:

```
vllm @ git+https://github.com/RohithKuditipudi/vllm.git@e0ce7b17fcce3c6426162a9763cf12f394fe1db8
```

The fork differs from upstream vLLM by one scheduler change of four lines: a
`held_request_ids` set, and a check in `schedule()` that skips any running
request in it. The worker loop holds every request that is waiting on the
coordinator's token decision, which stock vLLM has no way to express. The
fork is based on the marin-community vLLM fork, not on upstream vLLM. Tests
live in `tests/joint_decode/`.

## CLI

A CLI serves as a simple end-to-end demonstration.

```bash
joint-decode-gpu --prompts prompts.jsonl --output completions.jsonl
```

Important flags:

- `--model-a`, `--model-b`: model paths or HF ids (default `Qwen/Qwen3-0.6B` for both).
- `--gpu-a`, `--gpu-b`: physical GPU ids. The parent sets
  `CUDA_VISIBLE_DEVICES` separately for each worker.
- `--top-k-a`, `--top-k-b`: top-k payload size each worker sends to the
  coordinator.
- `--advisor-weight`: the teacher weight H in the logit-mixing rule
  `logit_A + H * logit_B` (model A keeps full weight; H >= 0).
- `--teacher-scale std`: rescale model B's logits to model A's spread at each
  step first, `logit_A + H * (sigma_A / sigma_B) * logit_B`.
- `--temperature`: joint sampling temperature used by the coordinator.
- `--max-microbatch-size`: cap on live requests in the synchronized sliding
  window. The effective window is the minimum of this cap and each worker's KV
  cache capacity, and it ramps up over the first decision rounds within the
  per-step token budget (`--max-num-batched-tokens`).
- `--max-tokens`: shared side-local vLLM generation cap.
- `--max-tokens-a`, `--max-tokens-b`: optional side-specific generation caps.
- `--prompts`: path to a JSONL file.
  Each row must contain `id`, `prompt_a`, and `prompt_b` —
  `prompt_a` is fed to model A, `prompt_b` to model B. They can differ (e.g.
  different chat templates or system prompts):

  ```json
  {"id": "p0", "prompt_a": "Write a short proof that...", "prompt_b": "Write a short proof that..."}
  {"id": "p1", "prompt_a": "Explain why...", "prompt_b": "Explain why..."}
  ```

The CLI writes JSONL, one row per input row in input order:

```json
{"id": "p0", "prompt_a": "...", "prompt_b": "...", "completion": "...", "finish_reason": "..."}
```

`completion` is the model-A continuation.

## Custom Aggregation

The CLI uses the logit-mixing rule (`selection.select_add_logits`). Experiments can bypass the CLI and call
`run_joint_decode` directly with their own token selector. A selector can be
an arbitrary function that takes two lists of tokens and associated logits
(i.e., the top-k tokens from each model), plus keyword arguments `rng` and
`request_index` (the request's position in the batch), and returns side-local
token lists. Returning a single `int` is supported only as a shorthand
for forcing the same token id on both sides. For example:

```python
import random
from typing import Any

from hinted.joint_decode.gpu.decoder import run_joint_decode

def select_dumb(
    a_topk: list[dict[str, Any]], # [{"token_id": int, "logit": float},...]
    b_topk: list[dict[str, Any]],
    *,
    rng: random.Random,
    request_index: int,
):
  return [a_topk[0]["token_id"]], [b_topk[0]["token_id"]]

outputs = run_joint_decode(
    config,
    prompts_a,
    prompts_b,
    select_token=select_dumb
)
```

The two models may be given different prompts, which is how a hint is shown
to the second model only.

## Tests

`pytest tests/joint_decode` runs the CPU unit tests (coordinator, decision logic, selection,
runtime state, worker loop). The GPU tests are marked `gpu` and skip unless their model is set.

The core correctness test is a single-worker vLLM workload with a deterministic
HTTP decision coordinator. It forces distinct token streams per request id and
asserts vLLM outputs by request id. This tests the GPU-specific row-to-request-id
mapping in `JointDecodeLogitsProcessor`.

Run it from the repo root with a local tiny vLLM-compatible model:

```bash
JOINT_DECODE_GPU_TEST_MODEL="Qwen/Qwen3-0.6B" \
pytest -q tests/joint_decode/gpu/test_joint_decode_logits_processor_gpu.py -m gpu
```

Without `JOINT_DECODE_GPU_TEST_MODEL`, the test skips. The end-to-end tests in
`tests/joint_decode/gpu/test_joint_decode_end_to_end_gpu.py` run two engines and need
`JOINT_DECODE_GPU_TEST_MODEL_A` and `JOINT_DECODE_GPU_TEST_MODEL_B`.
