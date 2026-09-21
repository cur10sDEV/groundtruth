import asyncio

from app.core.logging import setup_logging
from app.ingestion.consumer import consume_loop


async def main() -> None:
    setup_logging()
    await consume_loop()


if __name__ == "__main__":
    asyncio.run(main())
