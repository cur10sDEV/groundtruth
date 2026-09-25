import asyncio

from app.ingestion import worker as worker_module


async def test_worker_main_runs_all_loops_concurrently(monkeypatch):
    started = []

    async def fake_consume_loop():
        started.append("consume-start")
        await asyncio.sleep(0)
        started.append("consume-end")

    async def fake_translate_loop():
        started.append("translate-start")
        await asyncio.sleep(0)
        started.append("translate-end")

    async def fake_cleanup_job_loop(interval_seconds=3600):
        started.append(("cleanup-start", interval_seconds))
        await asyncio.sleep(0)
        started.append("cleanup-end")

    monkeypatch.setattr(worker_module, "consume_loop", fake_consume_loop)
    monkeypatch.setattr(worker_module, "translate_loop", fake_translate_loop)
    monkeypatch.setattr(worker_module, "cleanup_job_loop", fake_cleanup_job_loop)

    await worker_module.main()

    # interleaved execution proves gather concurrency, not sequential awaiting
    assert started == [
        "consume-start",
        "translate-start",
        ("cleanup-start", 3600),
        "consume-end",
        "translate-end",
        "cleanup-end",
    ]
