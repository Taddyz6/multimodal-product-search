"""Bounded FIFO admission and in-flight deduplication.

One executor owns the non-thread-safe inference core. HTTP timeout does not
cancel an already running CPU job; shutdown waits for that job to finish.
"""

import asyncio
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import copy
from dataclasses import dataclass
import math
from .errors import ServiceError, Unavailable


class QueueFull(ServiceError):
    code = "queue_full"
    http_status = 429


class QueueExpired(ServiceError):
    code = "queue_timeout"
    http_status = 504


class RequestExpired(ServiceError):
    code = "request_timeout"
    http_status = 504


@dataclass(frozen=True)
class QueueConfig:
    capacity: int = 64
    queue_timeout: float = 9.5
    request_timeout: float = 10.0
    max_waiters: int = 64

    def __post_init__(self):
        for n in ["capacity", "max_waiters"]:
            v = getattr(self, n)
            if type(v) is not int or v < 1:
                raise ValueError(n + " must be a positive integer")
        for n in ["queue_timeout", "request_timeout"]:
            v = getattr(self, n)
            if (
                isinstance(v, bool)
                or not isinstance(v, (int, float))
                or (not math.isfinite(v))
                or (v <= 0)
            ):
                raise ValueError(n + " must be positive and finite")


@dataclass
class DispatchResult:
    payload: dict
    coalesced: bool
    queue_wait_ms: float
    scheduler_ms: float
    shared_job_queue_ms: float


@dataclass(eq=False)
class _Job:
    key: tuple
    call: object
    future: asyncio.Future
    arrived: float
    started: float | None = None
    waiters: int = 0
    expiry: asyncio.TimerHandle | None = None


class Dispatcher:
    def __init__(self, config):
        self.config = config
        self._pending = deque()
        self._flights = {}
        self._waiters = 0
        self._active = None
        self._accepting = False
        self._worker = None
        self._counts = {
            "submitted": 0,
            "executed": 0,
            "coalesced": 0,
            "rejected": 0,
            "queue_expired": 0,
            "request_expired": 0,
            "cancelled": 0,
            "failed": 0,
        }

    async def __aenter__(self):
        self._loop = asyncio.get_running_loop()
        self._wake = asyncio.Event()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="product-inference")
        self._accepting = True
        self._worker = asyncio.create_task(self._run(), name="product-dispatch")
        return self

    async def __aexit__(self, *exc):
        await self.close()

    def snapshot(self):
        return {
            "accepting": self._accepting and self._worker is not None and (not self._worker.done()),
            "pending_jobs": len(self._pending),
            "active_jobs": int(self._active is not None),
            "waiters": self._waiters,
            "counts": dict(self._counts),
        }

    def _remove_pending(self, job, error=None):
        self._pending.remove(job)
        if self._flights.get(job.key) is job:
            del self._flights[job.key]
        if job.expiry:
            job.expiry.cancel()
        if not job.future.done():
            if error is None:
                job.future.cancel()
            else:
                job.future.set_exception(error)

    def _expire(self, job):
        if job.started is None and job in self._pending:
            self._counts["queue_expired"] += 1
            self._remove_pending(job, QueueExpired("Search waited too long in the queue."))

    async def submit(self, key, call):
        if not self.snapshot()["accepting"]:
            raise Unavailable("Search admission is unavailable.")
        arrived = self._loop.time()
        job = self._flights.get(key)
        coalesced = job is not None
        if self._waiters >= self.config.max_waiters or (
            job is None and len(self._pending) >= self.config.capacity
        ):
            self._counts["rejected"] += 1
            raise QueueFull("Search queue is full; retry later.")
        if job is None:
            future = self._loop.create_future()
            future.add_done_callback(lambda f: None if f.cancelled() else f.exception())
            job = _Job(key, call, future, arrived)
            self._flights[key] = job
            self._pending.append(job)
            job.expiry = self._loop.call_later(self.config.queue_timeout, self._expire, job)
            self._wake.set()
        else:
            self._counts["coalesced"] += 1
        self._waiters += 1
        job.waiters += 1
        self._counts["submitted"] += 1
        try:
            deadline = asyncio.timeout(self.config.request_timeout)
            try:
                async with deadline:
                    payload = await asyncio.shield(job.future)
            except TimeoutError as exc:
                if not deadline.expired():
                    raise
                self._counts["request_expired"] += 1
                raise RequestExpired("Search response deadline exceeded.") from exc
            return DispatchResult(
                copy.deepcopy(payload),
                coalesced,
                max(0.0, job.started - arrived) * 1000,
                (self._loop.time() - arrived) * 1000,
                (job.started - job.arrived) * 1000,
            )
        except asyncio.CancelledError:
            self._counts["cancelled"] += 1
            raise
        finally:
            self._waiters -= 1
            job.waiters -= 1
            if job.waiters == 0 and job in self._pending:
                self._remove_pending(job)

    async def _run(self):
        while True:
            if not self._pending:
                if not self._accepting:
                    return
                self._wake.clear()
                await self._wake.wait()
                continue
            job = self._pending[0]
            if self._loop.time() - job.arrived >= self.config.queue_timeout:
                self._expire(job)
                continue
            self._pending.popleft()
            if job.expiry:
                job.expiry.cancel()
            job.started = self._loop.time()
            self._active = job
            self._counts["executed"] += 1
            try:
                result = await self._loop.run_in_executor(self._executor, job.call)
                if not job.future.done():
                    job.future.set_result(result)
            except Exception as exc:
                self._counts["failed"] += 1
                if not job.future.done():
                    job.future.set_exception(exc)
            finally:
                self._active = None
                if self._flights.get(job.key) is job:
                    del self._flights[job.key]

    async def close(self):
        if self._worker is None:
            return
        self._accepting = False
        for job in list(self._pending):
            self._remove_pending(job, Unavailable("Search service is stopping."))
        self._wake.set()
        await self._worker
        self._executor.shutdown(wait=True)
        self._worker = None
