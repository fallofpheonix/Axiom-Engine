from __future__ import annotations

from threading import Event, Thread
from typing import Callable

from infrastructure.queue import Task, TaskQueue


class Worker:
    """Ray-style long-running worker consuming queued tasks."""

    def __init__(self, queue: TaskQueue, handler: Callable[[Task], dict], name: str):
        self.queue = queue
        self.handler = handler
        self.name = name
        self._stop = Event()
        self._thread = Thread(target=self._run, name=name, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            task = self.queue.get(timeout=0.1)
            if task is None:
                continue
            try:
                self.handler(task)
            finally:
                self.queue.task_done()
