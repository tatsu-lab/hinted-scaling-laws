from __future__ import annotations

from typing import List, Optional

import pytest

from hinted.query_engine.server import (
    SubmititWorker,
    SubmitResult,
    SubmitResults,
    TaskServer,
)


class NoLaunchServer(TaskServer):
    """Server that records launch requests instead of touching slurm."""

    def __init__(self) -> None:
        super().__init__()
        self.launch_requests: List[int] = []

    def launch_workers(self, num_workers: Optional[int] = None) -> None:
        self.launch_requests.append(num_workers or 0)


class ClientWorker(SubmititWorker):
    """Bare HTTP client for exercising the protocol from tests."""

    def __call__(self) -> None:  # pragma: no cover - not used
        raise NotImplementedError


@pytest.fixture
def server():
    with NoLaunchServer() as server:
        yield server


@pytest.fixture
def worker(server: NoLaunchServer) -> ClientWorker:
    return ClientWorker(server.server_address)


def _submit(worker: ClientWorker, task_id: str, result: object) -> None:
    worker.submit_results(
        SubmitResults(
            results=[SubmitResult(task_id=task_id, worker_id=worker.worker_id, result=result)]
        )
    )


def test_round_trip(server: NoLaunchServer, worker: ClientWorker) -> None:
    task_ids = server.add_tasks(["task-a", "task-b"])

    pulled = worker.get_tasks(num_tasks=2).tasks
    assert {t.task for t in pulled} == {"task-a", "task-b"}
    assert not worker.get_done()

    for item in pulled:
        _submit(worker, item.task_id, {"echo": item.task})

    assert worker.get_done()
    results = server.results()
    assert results[task_ids[0]] == {"echo": "task-a"}
    assert results[task_ids[1]] == {"echo": "task-b"}
    assert server.wait_until_done(timeout=0.1) is False  # grace period still running


def test_results_preserve_input_order_mapping(server: NoLaunchServer, worker: ClientWorker) -> None:
    task_ids = server.add_tasks([f"task-{i}" for i in range(5)])
    pulled = worker.get_tasks(num_tasks=5).tasks
    for item in reversed(pulled):  # complete out of order
        _submit(worker, item.task_id, item.task)
    results = server.results()
    assert [results[tid] for tid in task_ids] == [f"task-{i}" for i in range(5)]


def test_empty_queue_hands_out_duplicates(server: NoLaunchServer) -> None:
    server.add_tasks(["only-task"])
    first = ClientWorker(server.server_address)
    second = ClientWorker(server.server_address)

    assert len(first.get_tasks(num_tasks=1).tasks) == 1
    duplicated = second.get_tasks(num_tasks=1).tasks
    assert len(duplicated) == 1
    assert duplicated[0].task == "only-task"

    # Same worker asking again gets nothing (it already holds the task).
    assert first.get_tasks(num_tasks=1).tasks == []


def test_first_result_wins_for_duplicates(server: NoLaunchServer) -> None:
    task_ids = server.add_tasks(["dup-task"])
    first = ClientWorker(server.server_address)
    second = ClientWorker(server.server_address)
    item_1 = first.get_tasks(num_tasks=1).tasks[0]
    item_2 = second.get_tasks(num_tasks=1).tasks[0]
    assert item_1.task_id == item_2.task_id

    _submit(first, item_1.task_id, "winner")
    response = second.submit_results(
        SubmitResults(
            results=[
                SubmitResult(task_id=item_2.task_id, worker_id=second.worker_id, result="loser")
            ]
        )
    )
    assert response["items"][0]["status"] == "duplicate"
    assert server.results()[task_ids[0]] == "winner"


def test_dead_worker_requeues_tasks_and_relaunches(server: NoLaunchServer) -> None:
    server.add_tasks(["fragile-task"])
    worker = ClientWorker(server.server_address)
    item = worker.get_tasks(num_tasks=1).tasks[0]
    assert server.pending_count() == 0
    assert server.in_progress_count() == 1

    worker.report_dead(reason="simulated crash")

    assert server.pending_count() == 1  # requeued
    assert server.launch_requests == [1]  # replacement worker requested
    assert server.num_dead_workers == 1

    # A fresh worker can pick the task back up and finish the run.
    rescuer = ClientWorker(server.server_address)
    rescued = rescuer.get_tasks(num_tasks=1).tasks[0]
    assert rescued.task_id == item.task_id
    _submit(rescuer, rescued.task_id, "recovered")
    assert rescuer.get_done()


def test_preemption_relaunches_but_is_not_a_crash(server: NoLaunchServer) -> None:
    server.add_tasks(["task"])
    worker = ClientWorker(server.server_address)
    worker.get_tasks(num_tasks=1)

    worker.checkpoint()  # what submitit calls on slurm preemption

    assert server.pending_count() == 1  # task requeued
    assert server.launch_requests == [1]  # replacement launched
    assert server.num_dead_workers == 0  # preemption is not a crash
    assert server.num_preempted_workers == 1


def test_finished_worker_does_not_trigger_relaunch(server: NoLaunchServer) -> None:
    server.add_tasks(["task"])
    worker = ClientWorker(server.server_address)
    item = worker.get_tasks(num_tasks=1).tasks[0]
    _submit(worker, item.task_id, "done")
    worker.report_dead(reason="finished")
    assert server.launch_requests == []
    assert server.num_dead_workers == 0  # clean exits don't count as deaths


def test_auth_is_required(server: NoLaunchServer) -> None:
    import requests

    r = requests.get(f"{server.server_address}/status", timeout=10)
    assert r.status_code == 401
    r = requests.get(
        f"{server.server_address}/status",
        headers={"Authorization": "Bearer wrong-token"},
        timeout=10,
    )
    assert r.status_code == 403
