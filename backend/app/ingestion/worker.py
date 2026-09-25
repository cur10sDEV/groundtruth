import asyncio

from app.core.logging import setup_logging
from app.ingestion.cleanup_job import cleanup_job_loop
from app.ingestion.consumer import consume_loop
from app.ingestion.event_translator import translate_loop


async def main() -> None:
    setup_logging()
    await asyncio.gather(consume_loop(), translate_loop(), cleanup_job_loop())


def run_worker() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    run_worker()
