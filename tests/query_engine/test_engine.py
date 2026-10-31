from __future__ import annotations

import pytest

from hinted.models import MODELS
from hinted.query_engine.engine import InferenceTaskServer
from hinted.query_engine.types import InferenceConfig, ResourcesConfig
from hinted.query_engine.workers import (
    JointDecodeInferenceWorker,
    VLLMInferenceWorker,
)

MODEL = MODELS["Qwen3-32B"]


def _resources(tmp_path) -> ResourcesConfig:
    return ResourcesConfig(log_dir=str(tmp_path / "slurm"))


def test_vanilla_gpus_follow_tensor_parallel_size(tmp_path) -> None:
    config = InferenceConfig(model=MODEL, tensor_parallel_size=2)
    server = InferenceTaskServer(VLLMInferenceWorker, config, _resources(tmp_path))
    assert server.gpus_per_worker == 2


def test_vanilla_default_is_one_gpu(tmp_path) -> None:
    config = InferenceConfig(model=MODEL)
    server = InferenceTaskServer(VLLMInferenceWorker, config, _resources(tmp_path))
    assert server.gpus_per_worker == 1


def test_joint_gpus_ignore_tensor_parallel_size(tmp_path) -> None:
    config = InferenceConfig(model=MODEL, joint_gpus_per_worker=2)
    server = InferenceTaskServer(JointDecodeInferenceWorker, config, _resources(tmp_path))
    assert server.gpus_per_worker == 2


def test_tensor_parallel_size_validated() -> None:
    with pytest.raises(ValueError, match="tensor_parallel_size"):
        InferenceConfig(model=MODEL, tensor_parallel_size=0)


def test_joint_worker_rejects_tensor_parallelism() -> None:
    worker = JointDecodeInferenceWorker("http://localhost:1")
    config = InferenceConfig(model=MODEL, tensor_parallel_size=2)
    with pytest.raises(NotImplementedError, match="tensor parallelism"):
        worker.setup(config)
