"""Batch scheduler for triggering batch submissions."""

import asyncio
import logging
import time

from .models import BatchConfig
from .queue import RequestQueue
from .submitter import BatchSubmitter

logger = logging.getLogger(__name__)


class BatchScheduler:
    """Monitors queue and triggers batch submission based on thresholds."""

    def __init__(
        self,
        config: BatchConfig,
        queue: RequestQueue,
        submitter: BatchSubmitter,
    ):
        self._config = config
        self._queue = queue
        self._submitter = submitter
        self._running = False
        self._task: asyncio.Task | None = None
        self._max_reached_event = asyncio.Event()
        self._last_submit_time = time.monotonic()
        self._pending_submissions: set[asyncio.Task] = set()

    def notify_size_change(self, size: int) -> None:
        """Called when queue size changes to check if max is reached."""
        if size >= self._config.max_batch_size:
            self._max_reached_event.set()

    async def start(self) -> None:
        """Start the scheduler background task."""
        if self._running:
            return

        logger.info("Starting batch scheduler")
        self._running = True
        self._last_submit_time = time.monotonic()
        self._task = asyncio.create_task(self._scheduler_loop())

    async def stop(self) -> None:
        """Stop the scheduler and flush remaining requests."""
        logger.info("Stopping batch scheduler")
        self._running = False
        self._max_reached_event.set()  # Wake up the loop

        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("Scheduler task did not stop gracefully, cancelling")
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass

        # Flush any remaining requests (submit synchronously on stop)
        requests = await self._queue.drain()
        if requests:
            self._last_submit_time = time.monotonic()
            await self._submitter.submit_batch(requests)

        # Wait for all pending batch submissions to complete
        if self._pending_submissions:
            logger.info(f"Waiting for {len(self._pending_submissions)} pending batch submissions...")
            await asyncio.gather(*self._pending_submissions, return_exceptions=True)

        logger.info("Batch scheduler stopped")

    async def _scheduler_loop(self) -> None:
        """Main scheduler loop."""
        logger.debug("Scheduler loop started")
        while self._running:
            try:
                # Wait for either:
                # 1. Max batch size reached (event set)
                # 2. Timeout elapsed
                try:
                    await asyncio.wait_for(
                        self._max_reached_event.wait(),
                        timeout=self._config.batch_timeout,
                    )
                    logger.debug("Scheduler woke up: max_batch_size reached")
                except asyncio.TimeoutError:
                    logger.debug("Scheduler woke up: timeout elapsed")

                self._max_reached_event.clear()

                if not self._running:
                    break

                # Check submission conditions
                queue_size = await self._queue.size()
                time_elapsed = time.monotonic() - self._last_submit_time

                should_submit = False
                trigger_reason = ""

                if queue_size >= self._config.max_batch_size:
                    # Max reached - submit immediately
                    should_submit = True
                    trigger_reason = f"max_batch_size reached ({queue_size} >= {self._config.max_batch_size})"
                elif queue_size > 0 and time_elapsed >= self._config.batch_timeout:
                    # Timeout elapsed with pending requests - submit with padding if needed
                    # The submitter will pad to bedrock_min_records (100) if queue_size < 100
                    should_submit = True
                    if queue_size >= self._config.min_batch_size:
                        trigger_reason = f"timeout with min_batch_size ({queue_size} >= {self._config.min_batch_size}, {time_elapsed:.1f}s elapsed)"
                    else:
                        trigger_reason = f"timeout with padding ({queue_size} requests, will pad to {self._config.bedrock_min_records}, {time_elapsed:.1f}s elapsed)"

                if should_submit:
                    logger.info(f"Triggering batch submission: {trigger_reason}")
                    self._start_submission()

            except asyncio.CancelledError:
                logger.debug("Scheduler loop cancelled")
                break
            except Exception:
                logger.exception("Error in scheduler loop")
                await asyncio.sleep(1.0)

    def _start_submission(self) -> None:
        """Drain the queue and start batch submission in background."""
        # Use synchronous drain to avoid race conditions
        # The queue.drain() is async, so we need to schedule it
        task = asyncio.create_task(self._run_submission())
        self._pending_submissions.add(task)
        task.add_done_callback(self._pending_submissions.discard)

    async def _run_submission(self) -> None:
        """Run a single batch submission."""
        requests = await self._queue.drain()
        if requests:
            self._last_submit_time = time.monotonic()
            logger.info(f"Starting background submission with {len(requests)} requests")
            try:
                await self._submitter.submit_batch(requests)
            except Exception:
                logger.exception("Background batch submission failed")
