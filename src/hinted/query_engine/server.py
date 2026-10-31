"""HTTP task server and worker client, ported from sgs (sgs/utils/server.py).

Workers pull batches of opaque task strings and push results back. Once the queue is empty, idle
workers get duplicates of in-progress tasks and the first result wins; a dead worker's tasks are
requeued and a replacement job submitted.
"""

from __future__ import annotations

import queue
import socket
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import requests
import submitit
import uvicorn
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

# Shared-secret to avoid cross-talk between concurrent servers on the
# cluster; this does not need to be secure.
AUTH_TOKEN = "hinted-query-engine"

# Death-report reasons with protocol meaning: normal completion, and slurm
# preemption/timeout (submitit checkpoint). Anything else is a crash.
REASON_FINISHED = "finished"
REASON_PREEMPTED = "Preempted"


class TaskItem(BaseModel):
    task_id: str
    task: str


class GetTasksResponse(BaseModel):
    tasks: List[TaskItem]


class SubmitResult(BaseModel):
    task_id: str
    worker_id: str
    result: Any


class SubmitResults(BaseModel):
    results: List[SubmitResult]


class StatusResponse(BaseModel):
    pending: int
    in_progress: int
    completed: int
    is_done: bool


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class TaskServer(ABC):
    """In-memory HTTP task server; subclasses implement worker launching."""

    def __init__(self) -> None:
        self.host = socket.gethostbyname(socket.gethostname())
        self.port = find_free_port()
        self.server_address = f"http://{self.host}:{self.port}"
        self.auth_token = AUTH_TOKEN

        self._tasks: queue.Queue[Tuple[str, str]] = queue.Queue()
        self._in_progress: Dict[str, str] = {}
        self._worker_ids_to_task_ids: Dict[str, List[str]] = defaultdict(list)
        self._task_id_to_worker_ids: Dict[str, Set[str]] = defaultdict(set)
        self._results: Dict[str, Any] = {}

        self._lock = threading.Lock()
        self._done_cv = threading.Condition(self._lock)

        self._server_thread: Optional[threading.Thread] = None
        self._uvicorn_server: Optional[uvicorn.Server] = None
        self._running = False

        self.app = self._create_app()

        self.workers: List[submitit.core.core.Job] = []
        # Crashes only; preemptions are counted separately since they are
        # routine on preemptible partitions and fully handled by requeuing.
        self.num_dead_workers = 0
        self.num_preempted_workers = 0
        self.max_concurrent_workers = 0

        self.num_tasks = 0
        self._current_percent = 0
        self._first_task_time: Optional[float] = None

    # ------------- Public API -------------

    @abstractmethod
    def launch_workers(self, num_workers: Optional[int] = None) -> None:
        """Submit worker jobs. If num_workers is None, use the config default."""

    def kill_workers(self) -> None:
        for job in self.workers:
            # Pace scancel calls so we do not overwhelm the Slurm API
            time.sleep(0.2)
            try:
                job.cancel(check=False)
            except Exception as e:
                print(f"Warning: failed to cancel job {getattr(job, 'job_id', None)}: {e}")

    def start(self) -> None:
        if self._running:
            return
        config = uvicorn.Config(self.app, host=self.host, port=self.port, log_level="error")
        self._uvicorn_server = uvicorn.Server(config=config)
        self._server_thread = threading.Thread(target=self._uvicorn_server.run, daemon=True)
        self._server_thread.start()

        deadline = time.time() + 15
        while time.time() < deadline:
            try:
                with socket.create_connection((self.host, self.port), timeout=1):
                    break
            except OSError:
                time.sleep(0.05)
        else:
            raise RuntimeError(f"Task server failed to bind on {self.server_address}")
        self._running = True

    def stop(self) -> None:
        if not self._running:
            return
        if self._uvicorn_server is not None:
            self._uvicorn_server.should_exit = True
        if self._server_thread is not None:
            self._server_thread.join(timeout=5)
        self._uvicorn_server = None
        self._server_thread = None
        self._running = False
        self.kill_workers()

    def add_tasks(self, tasks: List[str]) -> List[str]:
        """Add opaque task strings; returns generated task_ids in order."""
        ids: List[str] = []
        with self._lock:
            for t in tasks:
                task_id = str(uuid.uuid4())
                self._tasks.put((task_id, t))
                ids.append(task_id)
            self.num_tasks += len(tasks)
            if self._first_task_time is None:
                self._first_task_time = time.time()
        with self._done_cv:
            self._done_cv.notify_all()
        return ids

    def results(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._results)

    def wait_until_done(
        self, poll_interval: float = 0.2, timeout: Optional[float] = None
    ) -> bool:
        """Block until all tasks complete. Returns False on timeout."""
        start = time.time()
        grace_period_start: Optional[float] = None
        grace_period_duration = 10.0

        with self._done_cv:
            while True:
                self.max_concurrent_workers = max(
                    self.max_concurrent_workers, len(self._worker_ids_to_task_ids)
                )

                percent = int(100 * len(self._results) / max(1, self.num_tasks))
                if percent > self._current_percent:
                    self._current_percent = percent
                    print(
                        f"Processing {percent}% complete "
                        f"({len(self._results)}/{self.num_tasks} tasks, "
                        f"avg {self._output_tokens_per_second_locked():.0f} output tok/s)",
                        flush=True,
                    )

                if self._is_done_locked():
                    if grace_period_start is None:
                        grace_period_start = time.time()
                    elif time.time() - grace_period_start > grace_period_duration:
                        return True
                else:
                    grace_period_start = None

                if timeout is not None and (time.time() - start) >= timeout:
                    return False

                self._done_cv.wait(timeout=poll_interval)

    def pending_count(self) -> int:
        return self._tasks.qsize()

    def in_progress_count(self) -> int:
        with self._lock:
            return len(self._in_progress)

    def completed_count(self) -> int:
        with self._lock:
            return len(self._results)

    def __enter__(self) -> "TaskServer":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.stop()

    # ------------- Internal -------------

    def _is_done_locked(self) -> bool:
        return self._tasks.empty() and not self._in_progress

    def _output_tokens_per_second_locked(self) -> float:
        """Cumulative fleet throughput: output tokens across all completed
        results / wall time since tasks were added (includes worker boot and
        queue wait, so it lowballs early and converges as the run matures).
        Results are opaque payloads; count only dicts carrying token counts."""
        if self._first_task_time is None:
            return 0.0
        elapsed = max(1e-6, time.time() - self._first_task_time)
        tokens = sum(
            payload.get("output_token_count") or 0
            for payload in self._results.values()
            if isinstance(payload, dict)
        )
        return tokens / elapsed

    def _check_token(self, authorization: Optional[str]) -> None:
        token = authorization
        if token and token.lower().startswith("bearer "):
            token = token.split(None, 1)[1]
        if token is None:
            raise HTTPException(status_code=401, detail="Missing token")
        if token != self.auth_token:
            raise HTTPException(status_code=403, detail="Invalid token")

    def _create_app(self) -> FastAPI:
        app = FastAPI(title="TaskServer")

        @app.get("/status", response_model=StatusResponse)
        def status(authorization: Optional[str] = Header(None)) -> StatusResponse:
            self._check_token(authorization)
            with self._lock:
                return StatusResponse(
                    pending=self._tasks.qsize(),
                    in_progress=len(self._in_progress),
                    completed=len(self._results),
                    is_done=self._is_done_locked(),
                )

        @app.get("/get_task", response_model=GetTasksResponse)
        def get_task(
            worker_id: str,
            num_tasks: int = 1,
            authorization: Optional[str] = Header(None),
        ) -> GetTasksResponse:
            self._check_token(authorization)
            if num_tasks <= 0:
                return GetTasksResponse(tasks=[])
            items: List[TaskItem] = []
            with self._lock:
                while len(items) < num_tasks:
                    try:
                        task_id, task = self._tasks.get_nowait()
                        if task_id in self._results:
                            continue  # already completed by a duplicate holder
                        self._in_progress.setdefault(task_id, task)
                        items.append(TaskItem(task_id=task_id, task=task))
                        self._worker_ids_to_task_ids[worker_id].append(task_id)
                        self._task_id_to_worker_ids[task_id].add(worker_id)
                    except queue.Empty:
                        # Queue drained: hand out duplicates of in-progress
                        # tasks (fewest current holders first).
                        if not self._in_progress:
                            break
                        owned = set(self._worker_ids_to_task_ids.get(worker_id, []))
                        candidates = [
                            tid for tid in self._in_progress if tid not in owned
                        ]
                        if not candidates:
                            break
                        tid = min(
                            candidates,
                            key=lambda t: len(self._task_id_to_worker_ids.get(t, set())),
                        )
                        items.append(TaskItem(task_id=tid, task=self._in_progress[tid]))
                        self._worker_ids_to_task_ids[worker_id].append(tid)
                        self._task_id_to_worker_ids[tid].add(worker_id)
            with self._done_cv:
                self._done_cv.notify_all()
            return GetTasksResponse(tasks=items)

        @app.get("/report_dead_worker")
        def report_dead_worker(
            worker_id: str,
            reason: Optional[str] = None,
            authorization: Optional[str] = Header(None),
        ) -> dict[str, str]:
            self._check_token(authorization)
            with self._lock:
                if reason != REASON_FINISHED:
                    print(f"Worker {worker_id} died. Reason: {reason}")

                task_ids = self._worker_ids_to_task_ids.pop(worker_id, [])
                for tid in task_ids:
                    holders = self._task_id_to_worker_ids.get(tid)
                    if holders is not None:
                        holders.discard(worker_id)
                        if not holders:
                            self._task_id_to_worker_ids.pop(tid, None)
                            task = self._in_progress.pop(tid, None)
                            if task is not None and tid not in self._results:
                                self._tasks.put((tid, task))

                if reason != REASON_FINISHED:
                    if reason == REASON_PREEMPTED:
                        self.num_preempted_workers += 1
                    else:
                        self.num_dead_workers += 1
                    if not self._is_done_locked():
                        print("Worker died with work remaining; launching a replacement.")
                        self.launch_workers(num_workers=1)
            with self._done_cv:
                self._done_cv.notify_all()
            return {"status": "ok"}

        @app.post("/submit_result")
        def submit_result(
            payload: SubmitResults, authorization: Optional[str] = Header(None)
        ) -> dict[str, Any]:
            self._check_token(authorization)
            item_statuses: List[Dict[str, str]] = []
            with self._lock:
                for res in payload.results:
                    if res.task_id not in self._in_progress:
                        status_str = "duplicate" if res.task_id in self._results else "unknown"
                        item_statuses.append({"task_id": res.task_id, "status": status_str})
                        continue

                    self._in_progress.pop(res.task_id, None)
                    self._results[res.task_id] = res.result
                    item_statuses.append({"task_id": res.task_id, "status": "ok"})

                    holders = self._task_id_to_worker_ids.pop(res.task_id, set())
                    for wid in holders:
                        lst = self._worker_ids_to_task_ids.get(wid)
                        if lst is not None:
                            while res.task_id in lst:
                                lst.remove(res.task_id)
                            if not lst:
                                del self._worker_ids_to_task_ids[wid]
            with self._done_cv:
                self._done_cv.notify_all()
            return {"status": "ok", "items": item_statuses}

        return app


class SubmititWorker(submitit.helpers.Checkpointable, ABC):
    """Worker-side HTTP client. Subclasses implement __call__ (the work loop).

    Checkpointable means slurm preemption calls checkpoint(), which reports
    this worker dead so the server requeues its tasks and replaces it.
    """

    def __init__(self, server_address: str) -> None:
        self.server_address = server_address.rstrip("/")
        self.worker_id = str(uuid.uuid4())
        self.auth_token = AUTH_TOKEN
        self.im_dead = False
        super().__init__()

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> None: ...

    def _headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.auth_token}",
        }

    def get_tasks(self, num_tasks: int = 1) -> GetTasksResponse:
        params: Dict[str, Union[str, int]] = {
            "worker_id": self.worker_id,
            "num_tasks": int(num_tasks),
        }
        r = requests.get(
            f"{self.server_address}/get_task",
            headers=self._headers(),
            params=params,
            timeout=120,
        )
        r.raise_for_status()
        return GetTasksResponse.model_validate(r.json())

    def get_done(self) -> bool:
        r = requests.get(
            f"{self.server_address}/status", headers=self._headers(), timeout=120
        )
        r.raise_for_status()
        return StatusResponse.model_validate(r.json()).is_done

    def submit_results(self, results_model: SubmitResults) -> Dict[str, Any]:
        r = requests.post(
            f"{self.server_address}/submit_result",
            headers=self._headers(),
            json=results_model.model_dump(),
            timeout=120,
        )
        r.raise_for_status()
        return r.json()

    def report_dead(self, reason: Optional[str] = None) -> None:
        if self.im_dead:
            return
        self.im_dead = True
        params = {"worker_id": self.worker_id}
        if reason is not None:
            params["reason"] = reason
        try:
            requests.get(
                f"{self.server_address}/report_dead_worker",
                headers=self._headers(),
                params=params,
                timeout=10,
            )
        except requests.RequestException:
            pass

    def checkpoint(self, *args: Any, **kwargs: Any) -> None:
        # Report death; the server requeues our tasks and submits a fresh
        # worker. Returning None means submitit does NOT requeue this job.
        self.report_dead(reason=REASON_PREEMPTED)
        return None
