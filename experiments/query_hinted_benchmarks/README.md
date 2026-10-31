# Running the hinted benchmarks

Regenerates the rollouts behind the paper's forecasting figures. `launch_all.py` defines the
experiments and emits one command per cell; `single_runner.py` runs one cell; `grade.py` grades
afterwards; `to_rollouts.py` turns graded rollouts into result tables.

## The grid

| Benchmark | Families | Problems | Rollouts/problem |
|---|---|---|---|
| GSM8K | Delphi (11 compute budgets x 3 training seeds, plus 1e23: 34 checkpoints) | 256 | 32 hinted, 64 unhinted |
| AIME | Qwen2.5 (6), Qwen3 (6), Gemma-3 (5) | 60 | 10 |
| HLE | Qwen2.5 (6), Qwen3 (6), Gemma-3 (5), Qwen3.5 (5) | 176 | 10 |

Every pair runs `no_hint`, `mask` and `truncate` (H = 0.1 … 1.0). Logit mixing runs on GSM8K x
Delphi with Llama-3.1-8B-Instruct as the teacher (H = 0.1 … 16), and on AIME x {Qwen2.5, Qwen3,
Gemma-3} with the family's second-largest model as the teacher (H = 0.11 … 16), raw and with std
rescaling. That is 281 cells and about 15.1M rollouts; `python
experiments/query_hinted_benchmarks/launch_all.py` prints the grid.

## The methods

| Method | Hint enters via |
|---|---|
| `no_hint` | nothing: the H = 0 anchor, measured once per pair and shared by every method |
| `truncate` | the first H fraction of the reference solution, in the prompt |
| `mask` | a random H fraction of its words kept, in the prompt |
| `logit_mix` | a teacher (`--teacher`) reads the whole solution; tokens are sampled from `logit_model + H * logit_teacher` over the union of both models' top 16 tokens |
| `logit_mix_std` | as `logit_mix`, with the teacher's logits rescaled by `sigma_model / sigma_teacher` at each step |

Logits are mixed by token id, so the teacher must share the model's vocabulary. Logit mixing runs
two engines, each pinned to one GPU and not tensor-parallel: `--joint-gpus 2` (default) gives
each model a card, `--joint-gpus 1` puts both on one. The other methods accept
`--tensor-parallel`.

## Running

Run from the repository root.

```bash
# one cell
python experiments/query_hinted_benchmarks/single_runner.py \
    --model Qwen3-4B --benchmark hle --method truncate \
    --hint-file data/benchmarks/hle_solution_hints.jsonl --samples 10 \
    --levels 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --out experiments/query_hinted_benchmarks/out/hle__qwen3/truncate/Qwen3-4B.jsonl

# logit mixing on Delphi, two teacher weights
python .../single_runner.py --model delphi-1e21 --benchmark gsm8k --method logit_mix \
    --teacher Llama-3.1-8B-Instruct --hint-file data/benchmarks/gsm8k_solution_hints.jsonl \
    --samples 32 --levels 1 4 --out mix.jsonl

# every cell, as shell commands or as Slurm jobs (adapt SBATCH_HEADER in launch_all.py first)
python .../launch_all.py --print | bash
python .../launch_all.py --sbatch --submit
```

`--limit N` runs the first N problems only. Re-running a cell skips rollouts already in `--out`,
so an interrupted run resumes.

## Prompts without a model

Every prompt of the paper is on HuggingFace as
[suzeva/hinted-benchmarks](https://huggingface.co/datasets/suzeva/hinted-benchmarks).
`make_prompts.py` rebuilds them with the runner's own prompt code, without a GPU:

```bash
python experiments/query_hinted_benchmarks/make_prompts.py \
    --benchmark aime --method mask --levels 0.3 0.7 --out prompts/aime__mask.jsonl
python experiments/query_hinted_benchmarks/make_prompts.py \
    --benchmark hle --method truncate --levels 0.5 --show     # print one prompt
```

Sample *i* uses solution *i* mod the problem's number of solutions (10 on AIME and HLE, 1 on
GSM8K), and masking is seeded by (problem, solution, level), so the prompts are a deterministic
function of `data/benchmarks/`.

## Grading

| Benchmark | Rule |
|---|---|
| GSM8K | the number after the first `####`, `$` and commas dropped, compared as a number |
| AIME | the integer in `\boxed{}`, compared to the official answer |
| HLE | an LLM judge (gpt-5.4-nano); needs `OPENAI_API_KEY` |

```bash
python experiments/query_hinted_benchmarks/grade.py experiments/query_hinted_benchmarks/out/gsm8k__delphi

# HLE through the OpenAI Batch API (half price, results within 24 hours)
python .../grade.py .../out/hle__qwen3 --mode build      # requests and a cost estimate; spends nothing
python .../grade.py .../out/hle__qwen3 --mode submit
python .../grade.py .../out/hle__qwen3 --mode retrieve   # re-run until all batches are in
python .../grade.py some/Qwen3-4B.jsonl --mode sync      # direct calls, for a smoke test
```

## From rollouts to plots

```bash
python experiments/query_hinted_benchmarks/to_rollouts.py --dest my_rollouts/
python experiments/forecasting/plot.py all --pairs hle/qwen3 --rollouts-dir my_rollouts/ --out-dir my_plots/
```

`to_rollouts.py` writes tables in the format of `data/results/rollouts/`. `plot.py` reads only
`--rollouts-dir`, so it needs every table of the pair, including `<benchmark>__no_hint.csv`, and
every model of the family at every hint level. The ranking experiments need deeper unhinted
pools than this runner produces, so they are meant for the released data.
