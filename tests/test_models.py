from __future__ import annotations

from hinted.models import MODELS


def test_registry_keys_match_names() -> None:
    for key, config in MODELS.items():
        assert key == config.name


def test_registry_paths_are_consistent() -> None:
    for config in MODELS.values():
        org, _, repo = config.hf_path.partition("/")
        assert org and repo, config
        # meta-llama/Llama-3.1-8B-Instruct; marin-community/delphi-3e18-447Mparams-1.2Btokens
        if "-seed" in config.name:            # delphi-1e21-seed42 -> delphi-1e21-...-seed42
            base, _, seed = config.name.partition("-seed")
            assert repo.startswith(base), config
            # the unsuffixed 5e19 checkpoint is its seed 0
            assert repo.endswith(f"-seed{seed}") or (seed == "0" and "-seed" not in repo), config
            continue
        assert repo == config.name or repo.startswith(config.name), config


def test_registry_values_are_sane() -> None:
    for config in MODELS.values():
        assert 0.0 < config.temperature <= 2.0, config
        assert 0.0 < config.top_p <= 1.0, config
        assert config.top_k >= 0, config  # 0 = top_k disabled (e.g. Mistral)
        assert config.repetition_penalty >= 1.0, config
        assert 0 < config.requested_max_tokens <= config.context_limit, config
