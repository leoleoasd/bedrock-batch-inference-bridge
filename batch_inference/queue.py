"""Request queue for collecting batch inference requests."""

import asyncio
import uuid
from typing import Callable

from .models import BatchRequest


class RequestQueue:
    """Async queue for collecting inference requests."""

    def __init__(self, on_size_change: Callable[[int], None] | None = None):
        self._queue: list[BatchRequest] = []
        self._lock = asyncio.Lock()
        self._on_size_change = on_size_change

    async def put(
        self,
        body: bytes,
        content_type: str,
        accept: str,
    ) -> asyncio.Future:
        """Add a request to the queue and return a Future for the result."""
        record_id = str(uuid.uuid4())
        loop = asyncio.get_running_loop()
        future = loop.create_future()

        request = BatchRequest(
            record_id=record_id,
            body=body,
            content_type=content_type,
            accept=accept,
            future=future,
        )

        async with self._lock:
            self._queue.append(request)
            size = len(self._queue)
            # Call callback inside lock to ensure size is accurate
            if self._on_size_change:
                self._on_size_change(size)

        return future

    async def drain(self) -> list[BatchRequest]:
        """Remove and return all requests from the queue."""
        async with self._lock:
            requests = self._queue.copy()
            self._queue.clear()
            # Call callback inside lock to ensure consistency
            if self._on_size_change and requests:
                self._on_size_change(0)

        return requests

    async def size(self) -> int:
        """Return the current queue size."""
        async with self._lock:
            return len(self._queue)

    async def is_empty(self) -> bool:
        """Check if the queue is empty."""
        return await self.size() == 0
