"""Distributed inference over Slurm: an HTTP task server and pull-based GPU workers.

`run_inference(tasks, inference_config, resources)` runs plain tasks on VLLMInferenceWorker and
tasks with a teacher_prompt (logit mixing) on JointDecodeInferenceWorker.
"""

from hinted.query_engine.engine import (
    InferenceTaskServer,
    get_submitit_executor,
    run_inference,
)
from hinted.query_engine.server import SubmititWorker, TaskServer
from hinted.query_engine.types import (
    InferenceConfig,
    InferenceTask,
    QueryResult,
    ResourcesConfig,
)
from hinted.query_engine.workers import (
    InferenceWorker,
    JointDecodeInferenceWorker,
    VLLMInferenceWorker,
)

__all__ = [
    "InferenceConfig",
    "InferenceTask",
    "InferenceTaskServer",
    "InferenceWorker",
    "JointDecodeInferenceWorker",
    "QueryResult",
    "ResourcesConfig",
    "run_inference",
    "SubmititWorker",
    "TaskServer",
    "VLLMInferenceWorker",
    "get_submitit_executor",
]
