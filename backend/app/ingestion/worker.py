import asyncio

from app.core.logging import setup_logging
from app.ingestion.consumer import consume_loop


async def main() -> None:
    setup_logging()
    await consume_loop()


def run_worker() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    run_worker()
