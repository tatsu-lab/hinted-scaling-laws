"""Registry of supported models and their generation parameters.

Sampling settings are each family's officially recommended ones. Every
field is written out explicitly for every model on purpose: these numbers
are deliberate, and edits should be reviewable rather than hidden behind
defaults.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """Sampling settings for one model. Greedy decoding would be temperature=0;
    every supported model samples.

    requested_max_tokens is the experiment's generation budget; context_limit
    is the model's maximum context (prompt + output). The effective per-batch
    cap is min(requested_max_tokens, context_limit - prompt_tokens - margin).
    The Llama-2 models use requested_max_tokens = context_limit (4096): a
    32000 budget is unreachable inside a 4096-token context.
    """

    name: str
    hf_path: str
    temperature: float
    top_p: float
    top_k: int
    requested_max_tokens: int
    context_limit: int
    repetition_penalty: float


MODELS: dict[str, ModelConfig] = {
    config.name: config
    for config in [
        ModelConfig(
            name="Qwen3-0.6B",
            hf_path="Qwen/Qwen3-0.6B",
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            requested_max_tokens=32000,
            context_limit=32768,
            repetition_penalty=1.0,
        ),
        ModelConfig(
            name="Qwen3-1.7B",
            hf_path="Qwen/Qwen3-1.7B",
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            requested_max_tokens=32000,
            context_limit=32768,
            repetition_penalty=1.0,
        ),
        ModelConfig(
            name="Qwen3-4B",
            hf_path="Qwen/Qwen3-4B",
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            requested_max_tokens=32000,
            context_limit=32768,
            repetition_penalty=1.0,
        ),
        ModelConfig(
            name="Qwen3-8B",
            hf_path="Qwen/Qwen3-8B",
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            requested_max_tokens=32000,
            context_limit=32768,
            repetition_penalty=1.0,
        ),
        ModelConfig(
            name="Qwen3-14B",
            hf_path="Qwen/Qwen3-14B",
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            requested_max_tokens=32000,
            context_limit=32768,
            repetition_penalty=1.0,
        ),
        ModelConfig(
            name="Qwen3-30B-A3B",
            hf_path="Qwen/Qwen3-30B-A3B",
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            requested_max_tokens=32000,
            context_limit=32768,
            repetition_penalty=1.0,
        ),
        ModelConfig(
            name="Qwen3-32B",
            hf_path="Qwen/Qwen3-32B",
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            requested_max_tokens=32000,
            context_limit=32768,
            repetition_penalty=1.0,
        ),
        # --- Qwen3.5 dense ---------------------------------------------
        # Sampling params come from the model's own generation_config.json
        # (temperature 0.6, top_p 0.95, top_k 20), which happens to match the
        # Qwen3 entries above, so the two families stay protocol-comparable.
        # The model card also suggests temperature=1.0 with
        # presence_penalty=1.5 for "thinking mode, general tasks": not used
        # here — ModelConfig has no presence_penalty field, the joint-decode
        # path is temperature-only by design, and matching Qwen3 matters more
        # for the family comparison.
        #
        # context_limit stays 32768 even though Qwen3.5 is natively 262144:
        # context_limit IS vLLM's max_model_len (workers.py), and the vendored
        # max_num_batched_tokens default derives ~8x it, which starves the KV
        # cache (see query_engine/types.py). 32768 is ample for an HLE
        # question plus a full-solution hint.
        #
        # Architecture is a vision-language wrapper over a GDN hybrid text
        # tower; vLLM loads and generates with it as a text model.
        ModelConfig(
            name="Qwen3.5-0.8B",
            hf_path="Qwen/Qwen3.5-0.8B",
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            requested_max_tokens=32000,
            context_limit=32768,
            repetition_penalty=1.0,
        ),
        ModelConfig(
            name="Qwen3.5-2B",
            hf_path="Qwen/Qwen3.5-2B",
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            requested_max_tokens=32000,
            context_limit=32768,
            repetition_penalty=1.0,
        ),
        ModelConfig(
            name="Qwen3.5-4B",
            hf_path="Qwen/Qwen3.5-4B",
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            requested_max_tokens=32000,
            context_limit=32768,
            repetition_penalty=1.0,
        ),
        ModelConfig(
            name="Qwen3.5-9B",
            hf_path="Qwen/Qwen3.5-9B",
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            requested_max_tokens=32000,
            context_limit=32768,
            repetition_penalty=1.0,
        ),
        ModelConfig(
            name="Qwen3.5-27B",
            hf_path="Qwen/Qwen3.5-27B",
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            requested_max_tokens=32000,
            context_limit=32768,
            repetition_penalty=1.0,
        ),
        ModelConfig(
            name="Qwen2.5-0.5B-Instruct",
            hf_path="Qwen/Qwen2.5-0.5B-Instruct",
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            requested_max_tokens=32000,
            context_limit=32768,
            repetition_penalty=1.1,
        ),
        ModelConfig(
            name="Qwen2.5-1.5B-Instruct",
            hf_path="Qwen/Qwen2.5-1.5B-Instruct",
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            requested_max_tokens=32000,
            context_limit=32768,
            repetition_penalty=1.1,
        ),
        ModelConfig(
            name="Qwen2.5-3B-Instruct",
            hf_path="Qwen/Qwen2.5-3B-Instruct",
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            requested_max_tokens=32000,
            context_limit=32768,
            repetition_penalty=1.05,
        ),
        ModelConfig(
            name="Qwen2.5-7B-Instruct",
            hf_path="Qwen/Qwen2.5-7B-Instruct",
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            requested_max_tokens=32000,
            context_limit=32768,
            repetition_penalty=1.05,
        ),
        ModelConfig(
            name="Qwen2.5-14B-Instruct",
            hf_path="Qwen/Qwen2.5-14B-Instruct",
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            requested_max_tokens=32000,
            context_limit=32768,
            repetition_penalty=1.05,
        ),
        ModelConfig(
            name="Qwen2.5-32B-Instruct",
            hf_path="Qwen/Qwen2.5-32B-Instruct",
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            requested_max_tokens=32000,
            context_limit=32768,
            repetition_penalty=1.05,
        ),
        ModelConfig(
            name="Qwen2.5-72B-Instruct",
            hf_path="Qwen/Qwen2.5-72B-Instruct",
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            requested_max_tokens=32000,
            context_limit=32768,
            repetition_penalty=1.05,
        ),
        ModelConfig(
            name="Llama-2-7b-chat-hf",
            hf_path="meta-llama/Llama-2-7b-chat-hf",
            temperature=0.6,
            top_p=0.9,
            top_k=50,
            requested_max_tokens=4096,
            context_limit=4096,
            repetition_penalty=1.0,
        ),
        ModelConfig(
            name="Llama-2-13b-chat-hf",
            hf_path="meta-llama/Llama-2-13b-chat-hf",
            temperature=0.6,
            top_p=0.9,
            top_k=50,
            requested_max_tokens=4096,
            context_limit=4096,
            repetition_penalty=1.0,
        ),
        # Llama-3.1-8B-Instruct is also the logit-mixing teacher for the Delphi
        # ladder: it shares the 128,256-token Llama-3 vocabulary with Delphi,
        # which is what makes token-id logit mixing across the two well
        # defined, and is prompted in its own chat format
        # (GSM8KDelphiBenchmark(teacher=True)).
        ModelConfig(
            name="Llama-3.1-8B-Instruct",
            hf_path="meta-llama/Llama-3.1-8B-Instruct",
            temperature=0.6,
            top_p=0.9,
            top_k=50,
            requested_max_tokens=32000,
            context_limit=32768,
            repetition_penalty=1.0,
        ),
        ModelConfig(
            name="Llama-3.1-70B-Instruct",
            hf_path="meta-llama/Llama-3.1-70B-Instruct",
            temperature=0.6,
            top_p=0.9,
            top_k=50,
            requested_max_tokens=32000,
            context_limit=32768,
            repetition_penalty=1.0,
        ),
        ModelConfig(
            name="Llama-3.3-70B-Instruct",
            hf_path="meta-llama/Llama-3.3-70B-Instruct",
            temperature=0.6,
            top_p=0.9,
            top_k=50,
            requested_max_tokens=32000,
            context_limit=32768,
            repetition_penalty=1.0,
        ),
        ModelConfig(
            name="gemma-3-270m-it",
            hf_path="google/gemma-3-270m-it",
            temperature=1.0,
            top_p=0.95,
            top_k=64,
            requested_max_tokens=32000,
            context_limit=32768,
            repetition_penalty=1.0,
        ),
        ModelConfig(
            name="gemma-3-1b-it",
            hf_path="google/gemma-3-1b-it",
            temperature=1.0,
            top_p=0.95,
            top_k=64,
            requested_max_tokens=32000,
            context_limit=32768,
            repetition_penalty=1.0,
        ),
        ModelConfig(
            name="gemma-3-4b-it",
            hf_path="google/gemma-3-4b-it",
            temperature=1.0,
            top_p=0.95,
            top_k=64,
            requested_max_tokens=32000,
            context_limit=32768,
            repetition_penalty=1.0,
        ),
        ModelConfig(
            name="gemma-3-12b-it",
            hf_path="google/gemma-3-12b-it",
            temperature=1.0,
            top_p=0.95,
            top_k=64,
            requested_max_tokens=32000,
            context_limit=32768,
            repetition_penalty=1.0,
        ),
        ModelConfig(
            name="gemma-3-27b-it",
            hf_path="google/gemma-3-27b-it",
            temperature=1.0,
            top_p=0.95,
            top_k=64,
            requested_max_tokens=32000,
            context_limit=32768,
            repetition_penalty=1.0,
        ),
        ModelConfig(
            name="gemma-4-E2B-it",
            hf_path="google/gemma-4-E2B-it",
            temperature=1.0,
            top_p=0.95,
            top_k=64,
            requested_max_tokens=32000,
            context_limit=32768,
            repetition_penalty=1.0,
        ),
        ModelConfig(
            name="gemma-4-E4B-it",
            hf_path="google/gemma-4-E4B-it",
            temperature=1.0,
            top_p=0.95,
            top_k=64,
            requested_max_tokens=32000,
            context_limit=32768,
            repetition_penalty=1.0,
        ),
        ModelConfig(
            name="gemma-4-26B-A4B-it",
            hf_path="google/gemma-4-26B-A4B-it",
            temperature=1.0,
            top_p=0.95,
            top_k=64,
            requested_max_tokens=32000,
            context_limit=32768,
            repetition_penalty=1.0,
        ),
        ModelConfig(
            name="gemma-4-31B-it",
            hf_path="google/gemma-4-31B-it",
            temperature=1.0,
            top_p=0.95,
            top_k=64,
            requested_max_tokens=32000,
            context_limit=32768,
            repetition_penalty=1.0,
        ),
        ModelConfig(
            name="Llama-3.2-1B-Instruct",
            hf_path="meta-llama/Llama-3.2-1B-Instruct",
            temperature=0.6,
            top_p=0.9,
            top_k=50,
            requested_max_tokens=32000,
            context_limit=131072,
            repetition_penalty=1.0,
        ),
        ModelConfig(
            name="Mistral-7B-Instruct-v0.3",
            hf_path="mistralai/Mistral-7B-Instruct-v0.3",
            temperature=0.7,
            top_p=1.0,
            top_k=0,  # blank in the CSV -> vLLM's disabled value (consider all)
            requested_max_tokens=32000,
            context_limit=32768,
            repetition_penalty=1.0,
        ),
        ModelConfig(
            name="gemma-1.1-2b-it",
            hf_path="google/gemma-1.1-2b-it",
            temperature=1.0,
            top_p=0.95,
            top_k=64,
            requested_max_tokens=8192,
            context_limit=8192,
            repetition_penalty=1.0,
        ),
        ModelConfig(
            name="gemma-1.1-7b-it",
            hf_path="google/gemma-1.1-7b-it",
            temperature=1.0,
            top_p=0.95,
            top_k=64,
            requested_max_tokens=8192,
            context_limit=8192,
            repetition_penalty=1.0,
        ),
        ModelConfig(
            name="gemma-2-2b-it",
            hf_path="google/gemma-2-2b-it",
            temperature=1.0,
            top_p=0.95,
            top_k=64,
            requested_max_tokens=8192,
            context_limit=8192,
            repetition_penalty=1.0,
        ),
        ModelConfig(
            name="gemma-2-9b-it",
            hf_path="google/gemma-2-9b-it",
            temperature=1.0,
            top_p=0.95,
            top_k=64,
            requested_max_tokens=8192,
            context_limit=8192,
            repetition_penalty=1.0,
        ),
        ModelConfig(
            name="Qwen1.5-1.8B-Chat",
            hf_path="Qwen/Qwen1.5-1.8B-Chat",
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            requested_max_tokens=32000,
            context_limit=32768,
            repetition_penalty=1.1,
        ),
        # --- Delphi scaling ladder (marin-community) ----------------------
        # One compute-optimal rung per training-FLOP scale, named by the bare
        # scale. BASE models, no chat template -- run with
        # apply_chat_template=False. Sampling is the paper's: temperature 0.4,
        # top_p 1.0, top_k 1000, max_tokens 512 (the unhinted runs turn top-k
        # off and logit mixing samples from the top 16; see
        # experiments/query_hinted_benchmarks/single_runner.py); stops are
        # per-run (InferenceConfig.stop), not per-model. Context limit 4096
        # from the HF configs (Qwen3-arch with a llama3 tokenizer).
        ModelConfig(
            name="delphi-3e18",
            hf_path="marin-community/delphi-3e18-447Mparams-1.2Btokens",
            temperature=0.4,
            top_p=1.0,
            top_k=1000,
            requested_max_tokens=512,
            context_limit=4096,
            repetition_penalty=1.0,
        ),
        ModelConfig(
            name="delphi-9e18",
            hf_path="marin-community/delphi-9e18-550Mparams-2.9Btokens",
            temperature=0.4,
            top_p=1.0,
            top_k=1000,
            requested_max_tokens=512,
            context_limit=4096,
            repetition_penalty=1.0,
        ),
        ModelConfig(
            name="delphi-2e19",
            hf_path="marin-community/delphi-2e19-837Mparams-3.6Btokens",
            temperature=0.4,
            top_p=1.0,
            top_k=1000,
            requested_max_tokens=512,
            context_limit=4096,
            repetition_penalty=1.0,
        ),
        ModelConfig(
            name="delphi-3e19",
            hf_path="marin-community/delphi-3e19-998Mparams-5Btokens",
            temperature=0.4,
            top_p=1.0,
            top_k=1000,
            requested_max_tokens=512,
            context_limit=4096,
            repetition_penalty=1.0,
        ),
        ModelConfig(
            name="delphi-9e19",
            hf_path="marin-community/delphi-9e19-1.4Bparams-10.6Btokens",
            temperature=0.4,
            top_p=1.0,
            top_k=1000,
            requested_max_tokens=512,
            context_limit=4096,
            repetition_penalty=1.0,
        ),
        ModelConfig(
            name="delphi-2e20",
            hf_path="marin-community/delphi-2e20-1.9Bparams-14.8Btokens",
            temperature=0.4,
            top_p=1.0,
            top_k=1000,
            requested_max_tokens=512,
            context_limit=4096,
            repetition_penalty=1.0,
        ),
        ModelConfig(
            name="delphi-3e20",
            hf_path="marin-community/delphi-3e20-2.5Bparams-18.6Btokens",
            temperature=0.4,
            top_p=1.0,
            top_k=1000,
            requested_max_tokens=512,
            context_limit=4096,
            repetition_penalty=1.0,
        ),
        ModelConfig(
            name="delphi-1e21",
            hf_path="marin-community/delphi-1e21-3.4Bparams-46.3Btokens",
            temperature=0.4,
            top_p=1.0,
            top_k=1000,
            requested_max_tokens=512,
            context_limit=4096,
            repetition_penalty=1.0,
        ),
        ModelConfig(
            name="delphi-1e22",
            hf_path="marin-community/delphi-1e22-9.7Bparams-160Btokens",
            temperature=0.4,
            top_p=1.0,
            top_k=1000,
            requested_max_tokens=512,
            context_limit=4096,
            repetition_penalty=1.0,
        ),
        ModelConfig(
            name="delphi-1e23",
            hf_path="marin-community/delphi-1e23-25Bparams-628Btokens",
            temperature=0.4,
            top_p=1.0,
            top_k=1000,
            requested_max_tokens=512,
            context_limit=4096,
            repetition_penalty=1.0,
        ),
    ]
}


# --- Delphi training seeds ----------------------------------------------------
# The paper runs every Delphi rung at three training seeds (the checkpoints
# above, which the released tables call seed "original", plus 42 and 62746)
# and adds 5e19 and 6e20 rungs (seeds 0, 42, 62746); 1e23 has one seed. Same
# sampling as the rungs above. The seed checkpoints that marin-community does not
# host are under the HF user rohithck, trained with the original Delphi recipe and
# listed in the marin-community Delphi collection; the unsuffixed 5e19 and 6e20
# repos are their seed 0 (5.3e19 and 5.5e20 FLOPs, rounded to the labels).
DELPHI_SEEDS = {
    "delphi-3e18-seed42": "rohithck/delphi-3e18-447Mparams-1.2Btokens-seed42",
    "delphi-3e18-seed62746": "rohithck/delphi-3e18-447Mparams-1.2Btokens-seed62746",
    "delphi-9e18-seed42": "rohithck/delphi-9e18-550Mparams-2.9Btokens-seed42",
    "delphi-9e18-seed62746": "rohithck/delphi-9e18-550Mparams-2.9Btokens-seed62746",
    "delphi-2e19-seed42": "rohithck/delphi-2e19-837Mparams-3.6Btokens-seed42",
    "delphi-2e19-seed62746": "rohithck/delphi-2e19-837Mparams-3.6Btokens-seed62746",
    "delphi-3e19-seed42": "rohithck/delphi-3e19-998Mparams-5Btokens-seed42",
    "delphi-3e19-seed62746": "rohithck/delphi-3e19-998Mparams-5Btokens-seed62746",
    "delphi-5e19-seed0": "rohithck/delphi-5e19-1.2Bparams-7.4Btokens",
    "delphi-5e19-seed42": "rohithck/delphi-5e19-1.2Bparams-7.4Btokens-seed42",
    "delphi-5e19-seed62746": "rohithck/delphi-5e19-1.2Bparams-7.4Btokens-seed62746",
    "delphi-9e19-seed42": "rohithck/delphi-9e19-1.4Bparams-10.6Btokens-seed42",
    "delphi-9e19-seed62746": "rohithck/delphi-9e19-1.4Bparams-10.6Btokens-seed62746",
    "delphi-2e20-seed42": "rohithck/delphi-2e20-1.9Bparams-14.8Btokens-seed42",
    "delphi-2e20-seed62746": "rohithck/delphi-2e20-1.9Bparams-14.8Btokens-seed62746",
    "delphi-3e20-seed42": "rohithck/delphi-3e20-2.5Bparams-18.6Btokens-seed42",
    "delphi-3e20-seed62746": "rohithck/delphi-3e20-2.5Bparams-18.6Btokens-seed62746",
    "delphi-6e20-seed0": "rohithck/delphi-6e20-2.9Bparams-29.9Btokens",
    "delphi-6e20-seed42": "rohithck/delphi-6e20-2.9Bparams-29.9Btokens-seed42",
    "delphi-6e20-seed62746": "rohithck/delphi-6e20-2.9Bparams-29.9Btokens-seed62746",
    "delphi-1e21-seed42": "marin-community/delphi-1e21-3.4Bparams-46.3Btokens-seed42",
    "delphi-1e21-seed62746": "marin-community/delphi-1e21-3.4Bparams-46.3Btokens-seed62746",
    "delphi-1e22-seed42": "marin-community/delphi-1e22-9.7Bparams-160Btokens-seed42",
    "delphi-1e22-seed62746": "marin-community/delphi-1e22-9.7Bparams-160Btokens-seed62746",
}
for _name, _path in DELPHI_SEEDS.items():
    MODELS[_name] = ModelConfig(
        name=_name,
        hf_path=_path,
        temperature=0.4,
        top_p=1.0,
        top_k=1000,
        requested_max_tokens=512,
        context_limit=4096,
        repetition_penalty=1.0,
    )
