from __future__ import annotations

from dataclasses import dataclass
from queue import Empty, Queue
from time import time
from uuid import uuid4


@dataclass(frozen=True)
class Task:
    """Queue task with explicit kind and payload."""

    kind: str
    payload: dict
    id: str = ""
    created_at: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", self.id or str(uuid4()))
        object.__setattr__(self, "created_at", self.created_at or time())


class TaskQueue:
    """Bounded FIFO task queue."""

    def __init__(self, max_size: int = 1000):
        self._queue: Queue[Task] = Queue(maxsize=max_size)

    def put(self, task: Task, block: bool = False) -> bool:
        if self._queue.full():
            return False
        self._queue.put(task, block=block)
        return True

    def get(self, timeout: float | None = None) -> Task | None:
        try:
            return self._queue.get(timeout=timeout)
        except Empty:
            return None

    def task_done(self) -> None:
        self._queue.task_done()

    def size(self) -> int:
        return self._queue.qsize()
