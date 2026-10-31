"""Orchestration: launch a worker fleet on slurm and run a batch of tasks."""

from __future__ import annotations

import os
import time
from typing import Any, Callable, Dict, List, Optional, Type

import submitit

from hinted.query_engine.server import TaskServer
from hinted.query_engine.types import InferenceConfig, InferenceTask, QueryResult, ResourcesConfig
from hinted.query_engine.workers import (
    InferenceWorker,
    JointDecodeInferenceWorker,
    VLLMInferenceWorker,
)


def _time_string_to_minutes(time_str: str) -> int:
    """'HH:MM:SS' or 'D-HH:MM:SS' -> minutes (submitit preemption needs ints)."""
    days = 0
    if "-" in time_str:
        day_part, time_str = time_str.split("-", 1)
        days = int(day_part)
    parts = [int(p) for p in time_str.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    hours, minutes, seconds = parts
    return days * 24 * 60 + hours * 60 + minutes + (1 if seconds else 0)


def get_submitit_executor(
    resources: ResourcesConfig, gpus_per_worker: int
) -> submitit.AutoExecutor:
    # Scrub inherited SLURM_* env vars so nested submission from inside a
    # slurm allocation doesn't confuse the new jobs.
    for key in list(os.environ):
        if key.startswith("SLURM_") and key != "SLURM_JOB_ID":
            os.environ.pop(key, None)
    os.environ["SLURM_CPU_BIND"] = "none"

    executor = submitit.AutoExecutor(folder=resources.log_dir)
    executor.update_parameters(
        slurm_gres=f"gpu:{gpus_per_worker}",
        slurm_mem=resources.mem,
        slurm_time=_time_string_to_minutes(resources.time),
        slurm_cpus_per_task=resources.cpus_per_task,
        slurm_additional_parameters={"export": "ALL,SLURM_CPU_BIND=none"},
    )
    # Site-specific settings are passed only when set, so an unset one falls
    # back to the cluster's own default rather than an explicit empty value.
    optional = {
        "slurm_account": resources.account,
        "slurm_partition": resources.partition,
        "slurm_constraint": resources.constraints,
        "slurm_exclude": resources.exclude,
    }
    executor.update_parameters(**{k: v for k, v in optional.items() if v})
    if resources.node_list is not None:
        executor.update_parameters(slurm_nodelist=resources.node_list)
    if resources.exclusive:
        executor.update_parameters(slurm_exclusive=True)
    return executor


class InferenceTaskServer(TaskServer):
    """Task server that launches InferenceWorker slurm jobs."""

    def __init__(
        self,
        worker_cls: Type[InferenceWorker],
        inference_config: InferenceConfig,
        resources: ResourcesConfig,
    ) -> None:
        super().__init__()
        self.worker_cls = worker_cls
        self.inference_config = inference_config
        self.resources = resources
        self.gpus_per_worker = (
            inference_config.joint_gpus_per_worker
            if worker_cls is JointDecodeInferenceWorker
            else inference_config.tensor_parallel_size
        )

    def launch_workers(self, num_workers: Optional[int] = None) -> None:
        if self._tasks.qsize() == 0 and not self._in_progress:
            raise ValueError("No work to be done; refusing to launch workers.")

        executor = get_submitit_executor(self.resources, self.gpus_per_worker)
        if num_workers is None:
            num_workers = self.resources.num_jobs

        for _ in range(num_workers):
            worker = self.worker_cls(self.server_address)
            job = executor.submit(worker, self.inference_config)
            self.workers.append(job)


def _cleanup_job_files(job: submitit.core.core.Job) -> None:
    """Delete one job's submitit files (logs, pickles, submission script)."""
    for path in [
        job.paths.stderr,
        job.paths.stdout,
        job.paths.submission_file,
        job.paths.submitted_pickle,
        job.paths.result_pickle,
    ]:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


def run_inference(
    tasks: List[InferenceTask],
    inference_config: InferenceConfig,
    resources: ResourcesConfig,
    timeout: Optional[float] = None,
    on_result: Optional[Callable[[QueryResult], None]] = None,
) -> List[QueryResult]:
    """Run tasks on a slurm worker fleet; returns results in task order.

    Worker kind is chosen from the tasks: any task with a teacher_prompt
    routes the whole batch to joint decoding, so callers should split their
    grid into a hint-free batch (hint_level == 0) and a hinted batch.

    If on_result is given it is invoked once per task as its result arrives
    (polled every few seconds) — use it to persist results incrementally so
    an interrupted run keeps everything finished so far.

    Submitit job files are deleted after a fully clean run (all tasks
    completed, no worker deaths); after any crash, preemption, or timeout
    they are all kept as evidence.
    """
    if not tasks:
        return []

    keys = [t.task_key for t in tasks]
    if len(set(keys)) != len(keys):
        raise ValueError("task_key values must be unique within a batch")

    hinted = [t for t in tasks if t.teacher_prompt is not None]
    if hinted and len(hinted) != len(tasks):
        raise ValueError(
            "Mixed batch: run hint-free tasks and hinted tasks as separate "
            "run_inference calls (they use different worker kinds)"
        )
    worker_cls: Type[InferenceWorker] = (
        JointDecodeInferenceWorker if hinted else VLLMInferenceWorker
    )

    with InferenceTaskServer(worker_cls, inference_config, resources) as server:
        task_ids = server.add_tasks([t.to_json() for t in tasks])
        server.launch_workers()

        delivered: set[str] = set()

        def drain_new_results() -> None:
            if on_result is None:
                return
            for task_id, payload in server.results().items():
                if task_id not in delivered:
                    delivered.add(task_id)
                    on_result(QueryResult.from_dict(payload))

        # Wait in slices so on_result fires as results arrive. Slices must
        # exceed wait_until_done's internal 10s done-grace-period, or a
        # finished run would never report as done.
        deadline = None if timeout is None else time.time() + timeout
        while True:
            slice_timeout = 15.0
            if deadline is not None:
                remaining = deadline - time.time()
                if remaining <= 0:
                    finished = False
                    break
                slice_timeout = min(slice_timeout, remaining)
            finished = server.wait_until_done(timeout=slice_timeout)
            drain_new_results()
            if finished:
                break

        raw_results: Dict[str, Any] = server.results()

    if not finished:
        print(f"Warning: timed out with {len(raw_results)}/{len(tasks)} tasks complete")

    clean_run = (
        finished
        and server.num_dead_workers == 0
        and len(raw_results) == len(tasks)
    )
    if clean_run:
        for job in server.workers:
            _cleanup_job_files(job)
    else:
        print(f"Run not clean; keeping submitit logs in {resources.log_dir}")

    results: List[QueryResult] = []
    for task, task_id in zip(tasks, task_ids):
        raw = raw_results.get(task_id)
        if raw is None:
            results.append(
                QueryResult(
                    task_key=task.task_key,
                    response_text="",
                    finish_reason="missing",
                    is_error=True,
                    error="no result returned (timeout or worker failure)",
                )
            )
        else:
            results.append(QueryResult.from_dict(raw))
    return results
