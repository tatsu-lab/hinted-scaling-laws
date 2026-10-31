# Results behind the paper's figures

The graded outcome of every rollout in the paper, as per-problem pass/fail strings, plus the
other inputs of the forecasting and ranking scripts.

## `rollouts/`

    <benchmark>__no_hint.csv      model, problem_id, passed, pool          H = 0, shared by all methods
    <benchmark>__<method>.csv     model, hint, problem_id, passed          H > 0

Benchmarks are `aime`, `hle` and `gsm8k`; methods are `mask` and `truncate` (H = 0.1 .. 1.0) on
every benchmark, `logit_mix` on GSM8K (Delphi, H = 0.1 .. 16) and AIME (H = 0.11 .. 16), and
`logit_mix_std` on AIME. For logit mixing H is the teacher's weight. Problem ids match
`data/benchmarks/`.

- `passed` has one character per rollout, `1` if graded correct, in a fixed order independent of
  correctness, so a problem's first k rollouts are its first k characters. The fits use the first 10
  on AIME and HLE and all of them on GSM8K.
- `pool` (unhinted AIME and HLE) marks each rollout for the ranking experiments: `x` sampling pool,
  `y` held-out ground truth, `u` unused. Qwen3.5 on HLE has its first 10 rollouts followed by a
  256-deep pool marked `u`, used by the pass@k analysis.
- The GSM8K tables have a `seed` column: each Delphi compute budget at three training seeds
  (`original`, `42`, `62746`; `0`, `42`, `62746` at 5e19 and 6e20), 1e23 at `original` only. A
  budget's score is the mean over its seeds.

`read_rollouts` in `experiments/forecasting/plot.py` or `experiments/ranking/ranking.py` expands a
table to one row per rollout.

## Other files

- `models.csv`: one row per (model, benchmark) with its family, whether the forecasts use it
  (`in_forecasts`), and its scale `capability`. For Delphi this is log10 of the recorded training
  budget, which the names round (`delphi_2e19` is 1.8e19 FLOPs, `delphi_5e19` 5.3e19,
  `delphi_2e20` 1.8e20, `delphi_6e20` 5.5e20).
- `ranking_config.json`: per benchmark, the model universe, the seven scored models, hint levels
  and Monte-Carlo settings.
- `bpb.csv`: bits per byte of each reference solution under each model, for the log-loss ranking
  baseline.
- `bpb_delphi.csv`: bits per byte of the GSM8K reference solutions under each original-seed Delphi
  model, for the two-stage baseline (`experiments/forecasting/two_stage.py`).
- `held_et_al_reference.csv`: [Held (2026)](https://www.openathena.ai/blog/delphi/)'s published
  bits per byte and GSM8K accuracy, for eleven external models (`group = external`, the two-stage
  baseline's reference models) and the Delphi models (used only to check `bpb_delphi.csv`).
- `manifest.csv`: rollout, model and problem counts per table.
