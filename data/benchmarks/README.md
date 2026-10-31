# Benchmark problems and reference solutions

The problem sets and the reference solutions that become hints.

| File | Rows | Contents |
|---|---|---|
| `aime2025_2026.jsonl` | 60 | AIME I and II, 2025 and 2026, from MathArena. Integer answers 0–999. |
| `hle.jsonl` | 176 | Humanity's Last Exam subset: free-form, text-only questions that do not list candidate answers and have a correct generated reference solution (paper Appendix A). |
| `gsm8k_delphi.jsonl` | 256 | The first 256 GSM8K test problems, with the few-shot prompt frame used for Delphi. |

| File | Rows | Per problem | Solved by | Rewritten by |
|---|---|---|---|---|
| `aime_solution_hints.jsonl` | 600 | 10 | Claude Opus 4.6 (573); GPT-5.4, medium thinking (27, on five problems Opus rarely solved) | Claude Opus 4.6, thinking off |
| `hle_solution_hints.jsonl` | 1,760 | 10 | GPT-5.5, medium thinking | Claude Opus 4.7, medium thinking |
| `gsm8k_solution_hints.jsonl` | 256 | 1 | The benchmark's own step-by-step solutions | — (answer line stripped) |

Truncation shows the first `H` fraction of a solution, masking a random `H` fraction of its words,
and logit mixing gives the whole solution to the teacher. A model's rollouts cycle through the
problem's solutions.

The AIME and HLE solutions are model-generated (paper Appendix B): a frontier model's solutions are
kept if graded correct (the official answer on AIME, an o3-mini judge on HLE), then a second model
rewrites each to stop before the final step. `generator_model` is the rewriter,
`metadata.source_generator_model` the solver and `metadata.prompt` the rewriting prompt. 

Loading needs only `problem_id`, `rollout_id` and `full_hint`, which is all the GSM8K file has; the
other fields in the AIME and HLE files are provenance.
