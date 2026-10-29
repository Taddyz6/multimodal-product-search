"""Real scheduler behavior with deterministic, gated work instead of ML assets."""

import asyncio
import threading

import pytest


async def until(predicate):
    async with asyncio.timeout(2):
        while not predicate():
            await asyncio.sleep(0.001)


def test_distinct_jobs_run_fifo_without_overlapping_and_keep_original_payload():
    from product_search.dispatch import Dispatcher, QueueConfig

    async def scenario():
        started = threading.Event()
        release = threading.Event()
        order = []

        def first():
            order.append("a")
            started.set()
            assert release.wait(2)
            return {
                "query": "a",
                "timings_ms": {"service_total": 7.0},
                "results": [{"score": 0.125}],
            }

        async with Dispatcher(QueueConfig()) as d:
            a = asyncio.create_task(d.submit(("online", "a", 10), first))
            await until(started.is_set)
            b = asyncio.create_task(
                d.submit(("online", "b", 10), lambda: order.append("b") or {"query": "b"})
            )
            await until(lambda: d.snapshot()["pending_jobs"] == 1)
            assert order == ["a"]
            release.set()
            ra, rb = await asyncio.gather(a, b)
            assert order == ["a", "b"] and ra.payload["results"] == [{"score": 0.125}]
            assert ra.payload["timings_ms"]["service_total"] == 7.0 and rb.payload["query"] == "b"
            assert rb.queue_wait_ms > 0 and not rb.coalesced

    asyncio.run(scenario())


def test_same_inflight_key_computes_once_and_each_caller_owns_its_response():
    from product_search.dispatch import Dispatcher, QueueConfig

    async def scenario():
        started = threading.Event()
        release = threading.Event()
        calls = []

        def work():
            calls.append("one")
            started.set()
            assert release.wait(2)
            return {"results": [{"rank": 1}]}

        async with Dispatcher(QueueConfig()) as d:
            a = asyncio.create_task(d.submit(("online", " original ", 10), work))
            await until(started.is_set)
            b = asyncio.create_task(d.submit(("online", " original ", 10), work))
            await until(lambda: d.snapshot()["waiters"] == 2)
            release.set()
            ra, rb = await asyncio.gather(a, b)
            assert calls == ["one"] and not ra.coalesced and rb.coalesced
            ra.payload["results"][0]["rank"] = 99
            assert rb.payload["results"] == [{"rank": 1}]
            await d.submit(("online", " original ", 10), work)
            assert calls == ["one", "one"], (
                "Completed results must not turn into a persistent cache"
            )

    asyncio.run(scenario())


def test_full_queue_refuses_new_work_but_accepts_an_existing_duplicate():
    from product_search.dispatch import Dispatcher, QueueConfig, QueueFull

    async def scenario():
        started = threading.Event()
        release = threading.Event()

        def work():
            started.set()
            assert release.wait(2)
            return {"query": "a"}

        async with Dispatcher(QueueConfig(capacity=1)) as d:
            a = asyncio.create_task(d.submit(("online", "a", 10), work))
            await until(started.is_set)
            b = asyncio.create_task(d.submit(("online", "b", 10), lambda: {"query": "b"}))
            await until(lambda: d.snapshot()["pending_jobs"] == 1)
            with pytest.raises(QueueFull):
                await d.submit(("online", "c", 10), lambda: {})
            duplicate = asyncio.create_task(d.submit(("online", "a", 10), work))
            await until(lambda: d.snapshot()["waiters"] == 3)
            release.set()
            _, _, r = await asyncio.gather(a, b, duplicate)
            assert r.coalesced and r.payload["query"] == "a"

    asyncio.run(scenario())


def test_expired_queued_job_is_removed_without_running():
    from product_search.dispatch import Dispatcher, QueueConfig, QueueExpired

    async def scenario():
        started = threading.Event()
        release = threading.Event()
        calls = []

        def work():
            started.set()
            assert release.wait(2)
            return {}

        async with Dispatcher(QueueConfig(capacity=1, queue_timeout=0.03)) as d:
            a = asyncio.create_task(d.submit(("a",), work))
            await until(started.is_set)
            with pytest.raises(QueueExpired):
                await d.submit(("expired",), lambda: calls.append("bad") or {})
            assert d.snapshot()["pending_jobs"] == 0 and not calls
            release.set()
            await a
            assert (await d.submit(("ok",), lambda: {"ok": True})).payload == {"ok": True}

    asyncio.run(scenario())


def test_cancelled_queued_caller_frees_its_capacity():
    from product_search.dispatch import Dispatcher, QueueConfig

    async def scenario():
        started = threading.Event()
        release = threading.Event()
        calls = []

        def work():
            started.set()
            assert release.wait(2)
            return {}

        async with Dispatcher(QueueConfig(capacity=1)) as d:
            a = asyncio.create_task(d.submit(("a",), work))
            await until(started.is_set)
            b = asyncio.create_task(d.submit(("b",), lambda: calls.append("bad") or {}))
            await until(lambda: d.snapshot()["pending_jobs"] == 1)
            b.cancel()
            with pytest.raises(asyncio.CancelledError):
                await b
            assert d.snapshot()["pending_jobs"] == 0
            c = asyncio.create_task(d.submit(("c",), lambda: {"ok": True}))
            await until(lambda: d.snapshot()["pending_jobs"] == 1)
            release.set()
            await a
            assert (await c).payload["ok"] and not calls

    asyncio.run(scenario())


def test_cancelled_duplicate_does_not_cancel_other_waiters():
    from product_search.dispatch import Dispatcher, QueueConfig

    async def scenario():
        started = threading.Event()
        release = threading.Event()

        def work():
            started.set()
            assert release.wait(2)
            return {"query": "shared"}

        async with Dispatcher(QueueConfig()) as d:
            a = asyncio.create_task(d.submit(("same",), work))
            await until(started.is_set)
            b = asyncio.create_task(d.submit(("same",), work))
            await until(lambda: d.snapshot()["waiters"] == 2)
            a.cancel()
            with pytest.raises(asyncio.CancelledError):
                await a
            release.set()
            assert (await b).payload == {"query": "shared"}

    asyncio.run(scenario())


def test_request_timeout_does_not_allow_overlapping_uncancellable_inference():
    from product_search.dispatch import Dispatcher, QueueConfig, RequestExpired

    async def scenario():
        started = threading.Event()
        release = threading.Event()
        order = []

        def work():
            started.set()
            assert release.wait(2)
            order.append("first finished")
            return {}

        async with Dispatcher(QueueConfig(request_timeout=0.05)) as d:
            a = asyncio.create_task(d.submit(("a",), work))
            await until(started.is_set)
            with pytest.raises(RequestExpired):
                await a
            b = asyncio.create_task(d.submit(("b",), lambda: order.append("second") or {}))
            await until(lambda: d.snapshot()["pending_jobs"] == 1)
            assert order == [] and d.snapshot()["active_jobs"] == 1
            release.set()
            await b
            assert order == ["first finished", "second"]

    asyncio.run(scenario())


def test_work_error_reaches_all_waiters_and_scheduler_recovers():
    from product_search.dispatch import Dispatcher, QueueConfig

    async def scenario():
        async with Dispatcher(QueueConfig()) as d:

            def broken():
                raise ValueError("deliberate test failure")

            with pytest.raises(ValueError):
                await d.submit(("bad",), broken)
            assert (await d.submit(("good",), lambda: {"ok": True})).payload["ok"]
            assert d.snapshot()["pending_jobs"] == 0 and d.snapshot()["waiters"] == 0

    asyncio.run(scenario())


def test_work_timeout_error_is_not_misreported_as_request_deadline():
    from product_search.dispatch import Dispatcher, QueueConfig, RequestExpired

    async def scenario():
        async with Dispatcher(QueueConfig()) as d:

            def work():
                raise TimeoutError("upstream operation timed out")

            with pytest.raises(TimeoutError, match="upstream") as caught:
                await d.submit(("bad",), work)
            assert not isinstance(caught.value, RequestExpired)

    asyncio.run(scenario())


def test_dequeue_rechecks_absolute_queue_deadline(monkeypatch):
    import time
    from product_search.dispatch import Dispatcher, QueueConfig, QueueExpired

    async def scenario():
        loop = asyncio.get_running_loop()
        first = loop.create_future()
        first.add_done_callback(lambda _: time.sleep(0.08))
        calls = []

        def completion_boundary(executor, call):
            calls.append(call)
            if len(calls) == 1:
                return first
            completed = loop.create_future()
            completed.set_result(call())
            return completed

        monkeypatch.setattr(loop, "run_in_executor", completion_boundary)
        async with Dispatcher(QueueConfig(queue_timeout=0.04)) as d:
            a = asyncio.create_task(d.submit(("a",), lambda: {}))
            await until(lambda: len(calls) == 1)
            b = asyncio.create_task(d.submit(("b",), lambda: {"bad": True}))
            await until(lambda: d.snapshot()["pending_jobs"] == 1)
            first.set_result({})
            await a
            with pytest.raises(QueueExpired):
                await b
            assert len(calls) == 1

    asyncio.run(scenario())


def test_topk_whitespace_and_operation_are_not_silently_merged():
    from product_search.dispatch import Dispatcher, QueueConfig

    async def scenario():
        calls = []
        async with Dispatcher(QueueConfig()) as d:
            keys = [
                ("search", "q", 10),
                ("search", "q", 20),
                ("search", " q", 10),
                ("replay", "q", 10),
            ]
            tasks = [
                asyncio.create_task(
                    d.submit(key, lambda key=key: calls.append(key) or {"key": key})
                )
                for key in keys
            ]
            results = await asyncio.gather(*tasks)
            assert calls == keys and all(not x.coalesced for x in results)

    asyncio.run(scenario())


def test_max_waiters_also_bounds_duplicate_requests():
    from product_search.dispatch import Dispatcher, QueueConfig, QueueFull

    async def scenario():
        started = threading.Event()
        release = threading.Event()

        def work():
            started.set()
            assert release.wait(2)
            return {}

        async with Dispatcher(QueueConfig(max_waiters=1)) as d:
            a = asyncio.create_task(d.submit(("a",), work))
            await until(started.is_set)
            with pytest.raises(QueueFull):
                await d.submit(("a",), work)
            release.set()
            await a

    asyncio.run(scenario())


def test_shutdown_refuses_new_and_pending_work_then_drains_active():
    from product_search.dispatch import Dispatcher, QueueConfig
    from product_search.errors import Unavailable

    async def scenario():
        started = threading.Event()
        release = threading.Event()

        def work():
            started.set()
            assert release.wait(2)
            return {"done": True}

        d = Dispatcher(QueueConfig())
        await d.__aenter__()
        a = asyncio.create_task(d.submit(("a",), work))
        await until(started.is_set)
        b = asyncio.create_task(d.submit(("b",), lambda: {}))
        await until(lambda: d.snapshot()["pending_jobs"] == 1)
        closing = asyncio.create_task(d.close())
        await until(lambda: not d.snapshot()["accepting"])
        with pytest.raises(Unavailable):
            await b
        with pytest.raises(Unavailable):
            await d.submit(("c",), lambda: {})
        release.set()
        assert (await a).payload["done"]
        await closing

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"capacity": 0},
        {"capacity": True},
        {"queue_timeout": float("nan")},
        {"request_timeout": -1},
        {"max_waiters": 0},
    ],
)
def test_invalid_limits_cannot_disable_resource_bounds(kwargs):
    from product_search.dispatch import QueueConfig

    with pytest.raises(ValueError):
        QueueConfig(**kwargs)
