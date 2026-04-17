from __future__ import annotations


class CompileError(Exception):
    def __init__(self, message: str, node_id: str | None = None):
        self.message = message
        self.node_id = node_id
        super().__init__(message)
