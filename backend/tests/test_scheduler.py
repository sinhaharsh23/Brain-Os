import asyncio
import threading
import time
from types import SimpleNamespace

import pytest

from app.events.bus import EventBus
from app.events.types import Event
from app.inference.scheduler import InferenceScheduler, SchedulerQueueFull


class FakeEngine:
    def __init__(self):
        self.current_session = None
        self.cancelled = set()
        self.cancel_calls = []

    def run(self, prompt, _params, on_event, owner=None, session_id=None):
        self.current_session = SimpleNamespace(session_id=session_id, owner=owner)
        try:
            on_event(Event("token.generated", {"text": prompt, "step": 0}, session_id=session_id))
            if prompt.startswith("error"):
                raise RuntimeError("intentional test failure")
            while prompt.startswith("slow") and session_id not in self.cancelled:
                time.sleep(0.005)
            if prompt.startswith("slow"):
                on_event(Event("inference.cancelled", {"reason": "test cancellation"}, session_id=session_id))
            else:
                on_event(Event("inference.complete", {"response": prompt}, session_id=session_id))
        finally:
            self.current_session = None

    def cancel(self, owner=None):
        self.cancel_calls.append(owner)
        if self.current_session is not None and (owner is None or self.current_session.owner is owner):
            self.cancelled.add(self.current_session.session_id)
            return True
        return False


async def wait_for_status(run, status, timeout=2.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while run.status != status and asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(0.005)
    assert run.status == status


async def submit(scheduler, prompt, owner, events):
    return await scheduler.submit(
        prompt,
        {},
        owner=owner,
        user_id=f"user-{owner}",
        connection_id=f"connection-{owner}",
        on_event=events.append,
    )


def test_bounded_queue_isolated_runs_and_queued_cancellation():
    async def scenario():
        engine = FakeEngine()
        scheduler = InferenceScheduler(engine, EventBus(), max_queue_size=1, max_active=1)
        await scheduler.start()
        events_a, events_b = [], []
        try:
            owner_a, owner_b = object(), object()
            first = await submit(scheduler, "slow-a", owner_a, events_a)
            await asyncio.sleep(0.02)
            second = await submit(scheduler, "quick-b", owner_b, events_b)
            with pytest.raises(SchedulerQueueFull):
                await submit(scheduler, "overflow", object(), [])

            assert second.queue_position == 1
            assert await scheduler.cancel_run(second.run_id, owner_a) is False
            assert await scheduler.cancel_run(second.run_id, owner_b) is True
            assert second.status == "CANCELLED"
            assert await scheduler.cancel_run(first.run_id, owner_a) is True
            await wait_for_status(first, "CANCELLED")
            await asyncio.sleep(0.02)

            assert {event.session_id for event in events_a if event.session_id} == {first.session_id}
            assert {event.session_id for event in events_b if event.session_id} == {second.session_id}
            assert all(event.data.get("run_id") == first.run_id for event in events_a)
            assert all(event.data.get("run_id") == second.run_id for event in events_b)
            assert any(event.type == "inference.queued" for event in events_b)
            assert any(event.type == "inference.cancelled" for event in events_b)
        finally:
            await scheduler.shutdown()

    asyncio.run(scenario())


def test_timeout_failure_cleanup_and_sequential_runs():
    async def scenario():
        engine = FakeEngine()
        scheduler = InferenceScheduler(engine, EventBus(), max_queue_size=2, max_active=1, timeout_s=0.03)
        await scheduler.start()
        events = []
        owner = object()
        try:
            timed_out = await submit(scheduler, "slow-timeout", owner, events)
            await wait_for_status(timed_out, "TIMED_OUT")
            failed = await submit(scheduler, "error-run", owner, events)
            await wait_for_status(failed, "FAILED")
            one = await submit(scheduler, "one", owner, events)
            await wait_for_status(one, "COMPLETED")
            two = await submit(scheduler, "two", owner, events)
            await wait_for_status(two, "COMPLETED")

            assert timed_out.run_id != failed.run_id != one.run_id != two.run_id
            assert any(event.type == "inference.timeout" for event in events)
            assert any(event.type == "inference.failed" for event in events)
            assert scheduler.snapshot()["active"] == 0
            assert scheduler.snapshot()["queue_size"] == 0
        finally:
            await scheduler.shutdown()

    asyncio.run(scenario())


def test_queue_positions_update_after_cancellation_and_shutdown_cleanup():
    async def scenario():
        engine = FakeEngine()
        scheduler = InferenceScheduler(engine, EventBus(), max_queue_size=3, max_active=1, timeout_s=1)
        await scheduler.start()
        owner_a, owner_b, owner_c = object(), object(), object()
        events_a, events_b, events_c = [], [], []
        try:
            first = await submit(scheduler, "slow-a", owner_a, events_a)
            await asyncio.sleep(0.02)
            second = await submit(scheduler, "quick-b", owner_b, events_b)
            third = await submit(scheduler, "quick-c", owner_c, events_c)
            assert second.queue_position == 1
            assert third.queue_position == 2
            assert await scheduler.cancel_run(second.run_id, owner_b)
            assert third.queue_position == 1
            assert scheduler.snapshot()["queue_size"] == 1
            await scheduler.cancel_run(first.run_id, owner_a)
            await wait_for_status(first, "CANCELLED")
            await wait_for_status(third, "COMPLETED")
            assert third.finished_at is not None
            assert scheduler.snapshot()["queue_size"] == 0
        finally:
            await scheduler.shutdown()

        engine2 = FakeEngine()
        scheduler2 = InferenceScheduler(engine2, EventBus(), max_queue_size=2, max_active=1, timeout_s=1)
        await scheduler2.start()
        try:
            active = await submit(scheduler2, "slow-shutdown", object(), [])
            await asyncio.sleep(0.01)
            queued = await submit(scheduler2, "queued-shutdown", object(), [])
            await scheduler2.shutdown()
            assert active.status == "CANCELLED"
            assert queued.status == "CANCELLED"
            assert active.finished_at is not None
            assert queued.finished_at is not None
            assert scheduler2.snapshot()["active"] == 0
            assert scheduler2.snapshot()["queue_size"] == 0
        finally:
            if scheduler2._workers:
                await scheduler2.shutdown()

    asyncio.run(scenario())
